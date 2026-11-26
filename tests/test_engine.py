"""引擎正确性测试：独立参考实现 + 续算一致性 + 缓存预算。

运行：python -m tests.test_engine   （或 python tests/test_engine.py）
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import dither as D  # noqa: E402


def reference_dither(gray: np.ndarray, force: np.ndarray | None) -> np.ndarray:
    """完全独立的朴素参考实现：普通二维数组，无任何滚动/复用。"""

    h, w = gray.shape
    err = [[0.0] * w for _ in range(h)]
    out = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            corrected = float(gray[r, c]) + err[r][c]
            if force is not None and int(force[r, c]) == 1:
                chosen = D.BLACK
            elif force is not None and int(force[r, c]) == 2:
                chosen = D.WHITE
            else:
                chosen = D.BLACK if corrected < D.THRESHOLD else D.WHITE
            out[r, c] = chosen
            e = corrected - chosen
            # 图外写入直接舍弃（用边界判断）
            if c + 1 < w:
                err[r][c + 1] += e * 7.0 / 16.0
            if r + 1 < h:
                if c - 1 >= 0:
                    err[r + 1][c - 1] += e * 3.0 / 16.0
                err[r + 1][c] += e * 5.0 / 16.0
                if c + 1 < w:
                    err[r + 1][c + 1] += e * 1.0 / 16.0
    return out


def random_gray(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 多取中间灰度，制造大量抖动决策
    base = rng.integers(0, 256, size=(h, w), dtype=np.int16)
    wave = (60 * np.sin(np.arange(w)[None, :] / 7.0 + seed)).astype(np.int16)
    g = np.clip(base + wave, 0, 255).astype(np.uint8)
    return g


def random_force(h: int, w: int, seed: int, density: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed + 999)
    f = np.zeros((h, w), dtype=np.uint8)
    mask = rng.random((h, w)) < density
    f[mask] = rng.choice([1, 2], size=int(mask.sum()))
    return f


def check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def test_matches_reference() -> None:
    print("test_matches_reference")
    for seed, (h, w) in enumerate([(1, 1), (1, 7), (7, 1), (33, 48), (64, 64), (100, 130)]):
        gray = random_gray(h, w, seed)
        ref = reference_dither(gray, None)
        got, _ = D.dither_full(gray, None)
        check(f"{h}x{w} 无强制色逐像素一致", np.array_equal(ref, got))
        force = random_force(h, w, seed)
        ref_f = reference_dither(gray, force)
        got_f, _ = D.dither_full(gray, force)
        check(f"{h}x{w} 带强制色逐像素一致", np.array_equal(ref_f, got_f))


def test_resume_identity() -> None:
    print("test_resume_identity")
    rng = np.random.default_rng(7)
    for trial, (h, w) in enumerate([(64, 80), (96, 96), (200, 120), (33, 33), (512, 256)]):
        gray = random_gray(h, w, trial + 20)
        old_force = random_force(h, w, trial, density=0.01)
        old_out, buf = D.dither_full(gray, old_force)

        # 新稿：在旧强制色基础上增删若干强制色（只统计真实变化，
        # 且保证检查点 floor32(最早改动行) 之上没有任何改动——
        # 应用层同样按“最早真实改动行”决定续算检查点）
        new_force = old_force.copy()
        rows_changed = set()
        tries = 0
        while len(rows_changed) < 5 and tries < 100:
            tries += 1
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            action = int(rng.integers(0, 3))  # 0 黑 1 白 2 清除
            nv = 0 if action == 2 else (1 if action == 0 else 2)
            if int(new_force[r, c]) == nv:
                continue
            new_force[r, c] = nv
            rows_changed.add(r)

        # 期望：同一 force 从头算
        expect, _ = D.dither_full(gray, new_force)

        # 实际：从缓存续算
        work_out = old_out.copy()
        start, got = D.dither_resume(gray, new_force, rows_changed, buf, work_out)
        earliest = min(rows_changed)
        check(
            f"trial{trial} 回退不超过31行 (start={start}, earliest={earliest})",
            0 <= earliest - start <= 31,
        )
        check(f"trial{trial} 续算与从头重算全图逐像素一致", np.array_equal(expect, got))
        # 续算后缓存再续算一次（模拟多轮改稿）仍一致
        r2 = int(rng.integers(0, h))
        c2 = int(rng.integers(0, w))
        if int(new_force[r2, c2]) == 1:
            c2 = (c2 + 1) % w
        new_force2 = new_force.copy()
        new_force2[r2, c2] = 1
        changed2 = {r2}
        expect2, _ = D.dither_full(gray, new_force2)
        start2, got2 = D.dither_resume(gray, new_force2, changed2, buf, got)
        check(f"trial{trial} 二次续算仍逐像素一致", np.array_equal(expect2, got2))
        check(f"trial{trial} 二次回退不超过31行", 0 <= r2 - start2 <= 31)


def test_buffer_budget() -> None:
    print("test_buffer_budget")
    for h in (1, 31, 32, 33, 64, 100, 2048):
        w = 16
        gray = random_gray(h, w, h)
        _, buf = D.dither_full(gray, None)
        import math

        expect_slots = math.ceil(h / 32)
        check(f"h={h} 检查点槽位=ceil(h/32)={expect_slots}", buf.slots == expect_slots)
        check(f"h={h} 物理行总数=槽位+2={expect_slots + 2}", buf.capacity == expect_slots + 2)
        check(
            f"h={h} 驻留行不超过物理容量",
            buf.stats().resident_rows <= buf.capacity,
        )
        # 工作区在任意时刻至多 2 行
        check(f"h={h} 工作区至多2行", len(buf._ws_order) <= 2)


def test_forced_propagation() -> None:
    print("test_forced_propagation")
    # 全 128 灰度：无强制色时首像素 128>=128 落白，误差扩散形成经典图案
    gray = np.full((40, 40), 128, dtype=np.uint8)
    base, _ = D.dither_full(gray, None)
    force = np.zeros((40, 40), dtype=np.uint8)
    force[0, 0] = 1  # 强制首像素落黑
    forced, _ = D.dither_full(gray, force)
    ref_forced = reference_dither(gray, force)
    check("强制黑与参考实现一致", np.array_equal(forced, ref_forced))
    check("强制色确实改变了后续网点", not np.array_equal(base, forced))

    # 续算：改动第 10 行某像素
    force2 = force.copy()
    force2[10, 5] = 2
    expect, _ = D.dither_full(gray, force2)
    got_buf_out, buf = D.dither_full(gray, force)
    start, got = D.dither_resume(gray, force2, {10}, buf, got_buf_out)
    check("强制色续算起点=0（floor32(10)）", start == 0)
    check("强制色续算与从头算一致", np.array_equal(expect, got))


def test_strip_detection() -> None:
    print("test_strip_detection")
    gray = random_gray(128, 64, 3)
    f1 = np.zeros((128, 64), dtype=np.uint8)
    out1, buf = D.dither_full(gray, f1)
    f2 = f1.copy()
    # 选一个原本落白的像素强制落黑，保证改动真实生效并产生误差扰动
    white_points = np.argwhere(out1 == D.WHITE)
    pr, pc = (int(x) for x in white_points[len(white_points) // 2])
    f2[pr, pc] = 1
    expect, _ = D.dither_full(gray, f2)
    out2 = out1.copy()
    D.dither_resume(gray, f2, {pr}, buf, out2)
    check("续算结果与从头算一致", np.array_equal(expect, out2))
    all_diff = np.argwhere(out1 != expect)
    check("该改动确实引发了网点变色", all_diff.size > 0)
    first_diff_row = int(all_diff[0][0])
    # 以真实首个差异行为条带上界，构造必含变色与必不含变色的条带
    top = (first_diff_row // 32) * 32
    bottom = min(128, top + 32)
    check(f"条带[{top},{bottom})检出变色", D.strip_changed(out1, expect, top, bottom))
    pt = D.first_changed_point(out1, expect, top, bottom)
    check("首个变色点行优先且确为差异点", pt is not None and expect[pt] != out1[pt])
    # 行优先：首个变色点必须是该行最左差异
    r0, c0 = pt
    check(
        "首个变色点为该行最左差异",
        c0 == int(np.nonzero(out1[r0] != expect[r0])[0][0]),
    )
    # 首个变色点之上的区域必然无变化；其下可能因误差扩散继续变色（正常现象）
    if top > 0:
        check(f"条带[0,{top})无变色", not D.strip_changed(out1, expect, 0, top))
    print(f"    （强制点({pr},{pc}) 首个变色行={first_diff_row}，"
          f"变色传播至第{int(all_diff[-1][0])}行）")

    # 必定波及的用例：全 128 灰图，条带内强制黑会持续向后扩散
    g2 = np.full((128, 64), 128, dtype=np.uint8)
    ff1 = np.zeros((128, 64), dtype=np.uint8)
    o1, b2 = D.dither_full(g2, ff1)
    ff2 = ff1.copy()
    ff2[70, 10] = 1
    o2_expect, _ = D.dither_full(g2, ff2)
    check("全128灰图条带[64,96)检出变色", D.strip_changed(o1, o2_expect, 64, 96))
    check("全128灰图条带[0,64)无变色", not D.strip_changed(o1, o2_expect, 0, 64))
    pt2 = D.first_changed_point(o1, o2_expect, 64, 96)
    check("首个变色点落在条带内", pt2 is not None and 64 <= pt2[0] < 96)



def test_performance() -> None:
    print("test_performance")
    for h, w in ((512, 512), (1024, 1024)):
        gray = random_gray(h, w, 1)
        t0 = time.perf_counter()
        out, buf = D.dither_full(gray, None)
        dt = time.perf_counter() - t0
        print(f"  全图 {h}x{w} 耗时 {dt:.2f}s（{h * w / dt / 1e6:.2f} Mpix/s）")
        force = np.zeros((h, w), dtype=np.uint8)
        force[h - 5, w // 2] = 1
        changed = {h - 5}
        out2 = out.copy()
        t0 = time.perf_counter()
        start, _ = D.dither_resume(gray, force, changed, buf, out2)
        dt2 = time.perf_counter() - t0
        expect, _ = D.dither_full(gray, force)
        check(f"{h}x{w} 续算结果一致", np.array_equal(expect, out2))
        print(f"  续算起点 {start}，耗时 {dt2:.2f}s")


if __name__ == "__main__":
    test_matches_reference()
    test_resume_identity()
    test_buffer_budget()
    test_forced_propagation()
    test_strip_detection()
    test_performance()
    print("\n全部测试通过 ✔")
