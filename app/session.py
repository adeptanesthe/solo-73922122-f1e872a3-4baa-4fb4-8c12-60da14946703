"""会话与稿件管理。

业务流程：
- 上传灰度 PNG 后创建会话，生成第 0 稿（无强制色，从头抖动）。
- 制版员可在预览上框选区域设置强制黑/白或清除强制色，产生“候选新稿”：
  系统用滚动缓存续算，结果与从头重算逐像素一致（引擎测试保证）。
- 可按行号标记一段“已打样条带” [top, bottom)。对候选稿逐条带比对：
  * 条带内有网点变色 → 标出首个变色点，拒绝换稿，旧稿及其强制色保留；
  * 所有已打样条带内均无变色 → 接受换稿，候选稿成为当前稿。
- 每个稿件独立持有一份 ⌈H/32⌉+2 行的误差缓存。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

import numpy as np

from . import dither as D

MAX_SIDE = 2048


class ValidationError(ValueError):
    pass


@dataclass
class Strip:
    """已打样条带（行区间 [top, bottom)，1 基行号在 API 层转换）。"""

    top: int
    bottom: int
    label: str = ""


@dataclass
class Revision:
    revision_id: str
    force: np.ndarray                 # 0 无 / 1 强制黑 / 2 强制白
    out: np.ndarray                   # 二值结果 0/255
    err: D.RollingErrorBuffer
    note: str = ""
    # 本稿相对上一稿的元信息（接受换稿时填写）
    resume_start: int | None = None
    changed_rows: list[int] = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    gray: np.ndarray                  # 8 位灰度原图
    revisions: list[Revision]
    strips: list[Strip]
    lock: threading.Lock
    # 最近一次被拒绝的候选稿（用于界面展示首个变色点），不纳入正式稿件链
    pending: dict | None = None


# 进程内会话表（演示规模足够；键为 session_id）
_SESSIONS: dict[str, Session] = {}
_GUARD = threading.Lock()


def create_session(gray: np.ndarray) -> Session:
    if gray.ndim != 2:
        raise ValidationError("只接受单通道 8 位灰度 PNG")
    if gray.dtype != np.uint8:
        raise ValidationError("只接受 8 位灰度 PNG")
    h, w = gray.shape
    if not (1 <= h <= MAX_SIDE and 1 <= w <= MAX_SIDE):
        raise ValidationError(f"宽高须在 1..{MAX_SIDE} 像素之间，实际 {w}x{h}")
    sid = uuid.uuid4().hex
    out, err = D.dither_full(gray, None)
    rev = Revision(
        revision_id=uuid.uuid4().hex,
        force=np.zeros((h, w), dtype=np.uint8),
        out=out,
        err=err,
        note="第0稿（无强制色，从头抖动）",
    )
    sess = Session(
        session_id=sid,
        gray=gray,
        revisions=[rev],
        strips=[],
        lock=threading.Lock(),
    )
    with _GUARD:
        _SESSIONS[sid] = sess
    return sess


def get_session(sid: str) -> Session:
    with _GUARD:
        sess = _SESSIONS.get(sid)
    if sess is None:
        raise ValidationError("会话不存在或已过期")
    return sess


def current_revision(sess: Session) -> Revision:
    return sess.revisions[-1]


def apply_force_rect(
    force: np.ndarray, x0: int, y0: int, x1: int, y1: int, mode: str
) -> tuple[np.ndarray, set[int]]:
    """在框选矩形上应用强制色，返回 (新 force 数组, 真实改动行集合)。

    mode: "black" 强制黑 / "white" 强制白 / "clear" 清除强制色。
    只统计值真正发生变化的像素所在行。
    """

    h, w = force.shape
    xa, xb = sorted((max(0, x0), min(w, x1)))
    ya, yb = sorted((max(0, y0), min(h, y1)))
    if xa >= xb or ya >= yb:
        raise ValidationError("框选区域为空或越界")
    code = {"black": 1, "white": 2, "clear": 0}.get(mode)
    if code is None:
        raise ValidationError("mode 必须是 black / white / clear")
    new_force = force.copy()
    region = new_force[ya:yb, xa:xb]
    diff_mask = region != code
    changed_rows = set()
    if diff_mask.any():
        local_rows = np.nonzero(diff_mask)[0]
        changed_rows = {int(ya + r) for r in np.unique(local_rows)}
        region[diff_mask] = code
    return new_force, changed_rows


def evaluate_candidate(
    sess: Session, x0: int, y0: int, x1: int, y1: int, mode: str
) -> dict:
    """应用框选强制色并续算，按已打样条带决定接受/拒绝。

    返回结构：
    {
      accepted: bool,
      resume_start: int,
      changed_rows: [...],
      violations: [{top,bottom,label,first_point:[r,c]}, ...],
      candidate_revision_id: str (仅 accepted 时有意义),
      buffer_stats: {...},
    }
    """

    with sess.lock:
        rev = current_revision(sess)
        new_force, changed_rows = apply_force_rect(
            rev.force, x0, y0, x1, y1, mode
        )
        result: dict = {
            "accepted": False,
            "changed_pixels": len(changed_rows),
            "violations": [],
        }
        if not changed_rows:
            result["note"] = "框选内强制色无实际变化，未产生新稿"
            sess.pending = result
            return result

        # 在当前稿缓存的深拷贝上续算：原缓存保持不动，
        # 接受则新稿持有工作缓存、旧稿保留自己的缓存；拒绝则直接丢弃拷贝。
        working_err = _clone_buffer(rev.err)
        candidate_out = rev.out.copy()
        start, _ = D.dither_resume(
            sess.gray, new_force, changed_rows, working_err, candidate_out
        )

        # 逐已打样条带检查
        violations = []
        for strip in sess.strips:
            pt = D.first_changed_point(rev.out, candidate_out, strip.top, strip.bottom)
            if pt is not None:
                violations.append(
                    {
                        "top": strip.top,
                        "bottom": strip.bottom,
                        "label": strip.label,
                        "first_point": [pt[0], pt[1]],
                    }
                )

        result.update(
            resume_start=start,
            changed_rows=sorted(changed_rows),
            buffer_stats=working_err.stats().__dict__,
        )

        if violations:
            # 拒绝：旧稿（含其缓存与强制色）从未被修改，原样保留；
            # 仅暂存候选结果用于前端预览。
            sess.pending = {
                **result,
                "accepted": False,
                "violations": violations,
                "preview_out": candidate_out,
                "new_force": new_force,
                "mode": mode,
                "rect": [x0, y0, x1, y1],
            }
            result["violations"] = violations
            return result

        # 接受：候选稿成为正式新稿，持有续算后的工作缓存；
        # 旧稿的缓存从未被改动，继续与自身 force/out 对应。
        new_rev = Revision(
            revision_id=uuid.uuid4().hex,
            force=new_force,
            out=candidate_out,
            err=working_err,
            note=f"强制色 {mode} 区域 ({x0},{y0})-({x1},{y1})",
            resume_start=start,
            changed_rows=sorted(changed_rows),
        )
        sess.revisions.append(new_rev)
        sess.pending = None
        result["accepted"] = True
        result["candidate_revision_id"] = new_rev.revision_id
        return result


def _clone_buffer(buf: D.RollingErrorBuffer) -> D.RollingErrorBuffer:
    clone = D.RollingErrorBuffer(buf.width, buf.height)
    clone._buf = buf._buf.copy()
    clone._ws = dict(buf._ws)
    clone._ws_order = list(buf._ws_order)
    return clone


def add_strip(sess: Session, top: int, bottom: int, label: str = "") -> Strip:
    """登记已打样条带，行号为 0 基闭开区间（API 层负责 1 基转换）。"""

    h = sess.gray.shape[0]
    if not (0 <= top < bottom <= h):
        raise ValidationError(f"条带行号非法：top={top}, bottom={bottom}, 图高={h}")
    with sess.lock:
        strip = Strip(top=top, bottom=bottom, label=label or f"条带{len(sess.strips) + 1}")
        sess.strips.append(strip)
        return strip


def remove_strip(sess: Session, index: int) -> None:
    with sess.lock:
        if not (0 <= index < len(sess.strips)):
            raise ValidationError("条带序号不存在")
        sess.strips.pop(index)


def force_summary(force: np.ndarray) -> dict:
    black = int(np.count_nonzero(force == 1))
    white = int(np.count_nonzero(force == 2))
    return {"black": black, "white": white, "total": black + white}
