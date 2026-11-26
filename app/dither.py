"""Floyd–Steinberg 误差扩散引擎（严格规格实现）。

规则：
- 扫描顺序：从左到右、从上到下。
- 校正灰度 corrected = 原灰度 + 累积误差，中间灰度保持浮点不截断。
- corrected < 128 落黑(0)，否则落白(255)。
- 误差 = corrected - 实际落色值（黑 0 / 白 255），按 7/16、3/16、5/16、1/16 扩散。
- 图外误差舍弃（越界写入直接丢弃）。
- 强制色：像素被强制黑/白时输出强制色，误差仍按“实际落色”计算扩散，
  因此强制改动会像真实落色一样把影响向后传播。

缓存预算（每稿）：
- ⌈H/32⌉ 条全宽误差行作为检查点，固定在第 0、32、64、… 行；
- 另加 2 行工作区，供处理当前行/下一行时使用；
- 续算起始行 = 不早于最早改动行的最近检查点（floor32(e)），
  距最早改动行至多 31 行，满足“最多回到最早改动行之前 31 行”。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BLACK = 0
WHITE = 255
THRESHOLD = 128
BLOCK = 32
ROLLBACK = 31

# Floyd–Steinberg 扩散权重
W_RIGHT = 7.0 / 16.0
W_BELOW_LEFT = 3.0 / 16.0
W_BELOW = 5.0 / 16.0
W_BELOW_RIGHT = 1.0 / 16.0


def cache_slots(height: int) -> int:
    """每稿缓存的全宽误差行数量：ceil(图高 / 32)。"""

    return math.ceil(height / BLOCK)


def checkpoint_rows(height: int) -> list[int]:
    """检查点逻辑行号：0, 32, 64, ...（共 ceil(H/32) 个）。"""

    return list(range(0, height, BLOCK))


@dataclass
class BufferStats:
    slots: int          # 检查点缓存槽位
    workspace: int      # 工作区行数（固定 2）
    capacity: int       # 物理全宽误差行总数
    resident_rows: int  # 当前驻留的逻辑行数量


class RollingErrorBuffer:
    """全宽误差行滚动缓存。

    物理布局：前 ``slots`` 行固定存放检查点（逻辑行 32k 存在槽位 k），
    其余 2 行为工作区，按 LRU 方式借给非检查点的当前行/下一行。
    任何新借出的工作区槽位先清零；检查点行的清零由续算流程显式完成。
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.slots = cache_slots(height)
        self.capacity = self.slots + 2
        self._buf = np.zeros((self.capacity, width), dtype=np.float64)
        # 工作区逻辑行 -> 物理槽位（至多 2 项），列表头为最旧
        self._ws: dict[int, int] = {}
        self._ws_order: list[int] = []

    # ---- 检查点 -------------------------------------------------------
    @staticmethod
    def is_checkpoint(row: int) -> bool:
        return row % BLOCK == 0

    def _pin_slot(self, row: int) -> int:
        return row // BLOCK

    # ---- 行访问 -------------------------------------------------------
    def row(self, row: int) -> np.ndarray:
        """取得逻辑行 ``row`` 的误差行（numpy 视图）。

        检查点行返回其固定 pin slot（入流快照，只读语义）；
        非检查点行占用一个工作区槽位，槽位不足时淘汰最旧工作区行。
        """

        if row < 0 or row >= self.height:
            raise IndexError(row)
        if self.is_checkpoint(row):
            return self._buf[self._pin_slot(row)]
        return self._alloc_workspace(row)

    def _alloc_workspace(self, logical: int) -> np.ndarray:
        """在工作区为非检查点 ``logical`` 行分配（或复用）物理槽位。"""

        slot = self._ws.get(logical)
        if slot is not None:
            return self._buf[slot]
        if len(self._ws_order) >= 2:
            old = self._ws_order.pop(0)
            slot = self._ws.pop(old)
            self._buf[slot].fill(0.0)
        else:
            slot = self.slots + len(self._ws_order)
        self._ws[logical] = slot
        self._ws_order.append(logical)
        return self._buf[slot]

    def working_cur(self, row: int) -> np.ndarray:
        """处理 ``row`` 行时的“当前行工作数组”。

        - 非检查点行：即其工作区行；
        - 检查点行：在工作区放置 pin slot 入流快照的副本，处理中的
          右扩散只改写副本，pin slot 保持纯入流，供后续续算复用。
        """

        if not self.is_checkpoint(row):
            return self.row(row)
        work = self._alloc_workspace(row)
        work[:] = self._buf[self._pin_slot(row)]
        return work

    def clear_all_after(self, keep_row: int) -> None:
        """续算前重置：保留 ``keep_row`` 的入流误差，其余行全部清零。

        ``keep_row`` 必须是检查点行。当 keep_row > 0 时，其内容是上一轮
        由未改动的上方行扩散下来的有效入流，予以保留；keep_row == 0 时
        图顶没有上方行，入流必须为零，同样清零。其下所有检查点与工作区
        都将在续算中重建。
        """

        for k in range(self.slots):
            logical = k * BLOCK
            if logical == keep_row and keep_row != 0:
                continue
            self._buf[k].fill(0.0)
        for slot in (self.slots, self.slots + 1):
            self._buf[slot].fill(0.0)
        self._ws.clear()
        self._ws_order.clear()

    def resident_rows(self) -> list[int]:
        rows = [k * BLOCK for k in range(self.slots)]
        rows.extend(self._ws_order)
        return rows

    def stats(self) -> BufferStats:
        return BufferStats(
            slots=self.slots,
            workspace=2,
            capacity=self.capacity,
            resident_rows=len(self.resident_rows()),
        )


def dither_range(
    gray: np.ndarray,
    force: np.ndarray | None,
    err: RollingErrorBuffer,
    start_row: int,
    end_row: int,
    out: np.ndarray,
) -> np.ndarray:
    """对 [start_row, end_row) 行做 Floyd–Steinberg 抖动，写入 ``out``。

    热循环使用 Python float 列表（numpy 仅作存储），保证运算顺序与
    IEEE double 结果在从头算/续算之间逐位一致。
    """

    height, width = gray.shape
    for r in range(start_row, end_row):
        # 当前行：检查点行走 pin slot 的工作区副本，避免污染入流快照
        cur = err.working_cur(r)
        cur_vals = cur.tolist()
        if r + 1 < height:
            nxt = err.row(r + 1)
            next_vals = nxt.tolist()
        else:
            nxt = None
            next_vals = None
        src_vals = gray[r].tolist()
        force_vals = force[r].tolist() if force is not None else None
        out_row = out[r]
        last = width - 1
        for c in range(width):
            corrected = src_vals[c] + cur_vals[c]
            if force_vals is not None:
                code = force_vals[c]
                if code == 1:
                    chosen = BLACK
                elif code == 2:
                    chosen = WHITE
                else:
                    chosen = BLACK if corrected < THRESHOLD else WHITE
            else:
                chosen = BLACK if corrected < THRESHOLD else WHITE
            out_row[c] = chosen
            quant_err = corrected - chosen
            if quant_err == 0.0:
                continue
            if c < last:
                cur_vals[c + 1] += quant_err * W_RIGHT
            if next_vals is not None:
                if c > 0:
                    next_vals[c - 1] += quant_err * W_BELOW_LEFT
                next_vals[c] += quant_err * W_BELOW
                if c < last:
                    next_vals[c + 1] += quant_err * W_BELOW_RIGHT
        cur[:] = cur_vals
        if nxt is not None:
            nxt[:] = next_vals
    return out


def dither_full(
    gray: np.ndarray, force: np.ndarray | None = None
) -> tuple[np.ndarray, RollingErrorBuffer]:
    """从头完整抖动，返回二值图（0/255 的 uint8）与误差缓存。"""

    height, width = gray.shape
    err = RollingErrorBuffer(width, height)
    out = np.zeros((height, width), dtype=np.uint8)
    dither_range(gray, force, err, 0, height, out)
    return out, err


def resume_checkpoint(changed_rows: set[int]) -> int:
    """续算检查点：floor32(最早改动行)，距改动行 0~31 行。"""

    earliest = min(changed_rows)
    return (earliest // BLOCK) * BLOCK


def dither_resume(
    gray: np.ndarray,
    new_force: np.ndarray,
    changed_rows: set[int],
    cached_err: RollingErrorBuffer,
    out: np.ndarray,
) -> tuple[int, np.ndarray]:
    """按新强制色从检查点续算。

    1. 取 floor32(最早改动行) 作为续算起始检查点（回退不超过 31 行）；
    2. 保留该检查点的入流误差（由未改动的上方行扩散而来，仍然有效），
       清空其下所有误差行；
    3. 从检查点重新抖动到图底，输出与从头重算逐像素一致。
    """

    height = gray.shape[0]
    start = resume_checkpoint(changed_rows)
    cached_err.clear_all_after(keep_row=start)
    dither_range(gray, new_force, cached_err, start, height, out)
    return start, out


def first_changed_point(
    old_out: np.ndarray, new_out: np.ndarray, top: int, bottom: int
) -> tuple[int, int] | None:
    """条带行区间 [top, bottom) 内首个网点变色点（行优先扫描）。"""

    height = old_out.shape[0]
    bottom = min(bottom, height)
    for r in range(max(0, top), bottom):
        diff = np.nonzero(old_out[r] != new_out[r])[0]
        if diff.size:
            return r, int(diff[0])
    return None


def strip_changed(
    old_out: np.ndarray, new_out: np.ndarray, top: int, bottom: int
) -> bool:
    """条带内是否存在任何网点变色。"""

    return first_changed_point(old_out, new_out, top, bottom) is not None
