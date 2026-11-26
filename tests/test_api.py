"""API 端到端测试：通过 Flask test client 走完整业务流程。

运行：python -I tests/test_api.py
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.server import app  # noqa: E402
from app import session as S  # noqa: E402
from tests.test_engine import reference_dither  # noqa: E402

client = app.test_client()
PASS = 0
FAIL = 0


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")
        raise AssertionError(name)


def png_upload_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def test_upload_validation() -> None:
    print("test_upload_validation")
    # 非 PNG
    r = client.post("/api/upload", data={"image": (io.BytesIO(b"not png"), "x.txt")},
                    content_type="multipart/form-data")
    check("非 PNG 被拒（400）", r.status_code == 400)

    # RGB 图被拒
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    r = client.post("/api/upload", data={"image": (io.BytesIO(buf.getvalue()), "rgb.png")},
                    content_type="multipart/form-data")
    check("RGB PNG 被拒", r.status_code == 400 and "灰度" in r.get_json()["error"])

    # 超限尺寸被拒
    big = np.zeros((2049, 8), dtype=np.uint8)
    r = client.post("/api/upload",
                    data={"image": (io.BytesIO(png_upload_bytes(big)), "big.png")},
                    content_type="multipart/form-data")
    check("超 2048 尺寸被拒", r.status_code == 400 and "2048" in r.get_json()["error"])

    # 合法灰度上传
    g = np.full((64, 64), 128, dtype=np.uint8)
    r = client.post("/api/upload",
                    data={"image": (io.BytesIO(png_upload_bytes(g)), "ok.png")},
                    content_type="multipart/form-data")
    check("合法灰度上传 200", r.status_code == 200)
    payload = r.get_json()
    check("返回会话与尺寸", payload["width"] == 64 and payload["height"] == 64)
    check("预算=ceil(64/32)+2=4 物理行",
          payload["buffer_budget"]["physical_rows"] == 4)
    check("初始只有第0稿", len(payload["revisions"]) == 1)


def test_full_workflow() -> None:
    print("test_full_workflow")
    rng = np.random.default_rng(11)
    h = w = 128
    gray = np.clip(
        rng.integers(40, 220, (h, w)) + 30 * np.sin(np.arange(w)[None, :] / 5.0),
        0, 255,
    ).astype(np.uint8)

    r = client.post("/api/upload",
                    data={"image": (io.BytesIO(png_upload_bytes(gray)), "g.png")},
                    content_type="multipart/form-data")
    sid = r.get_json()["session_id"]

    # 第 0 稿输出必须与参考实现一致
    out_resp = client.get(f"/api/session/{sid}/image/out")
    out0 = np.array(Image.open(io.BytesIO(out_resp.data)).convert("L"))
    ref0 = reference_dither(gray, None)
    check("第0稿与独立参考实现逐像素一致", np.array_equal(out0, ref0))
    check("输出仅含 0/255", set(np.unique(out0)).issubset({0, 255}))

    # 登记条带：第 1–40 行（覆盖 0 基 0..39）
    r = client.post(f"/api/session/{sid}/strips",
                    json={"top_line": 1, "bottom_line": 40, "label": "首条"})
    check("登记条带 200", r.status_code == 200 and len(r.get_json()["strips"]) == 1)

    # 找一个条带内、当前落白的像素强制落黑 → 必然在条带内产生变色 → 拒绝
    white_in_strip = np.argwhere(out0[:40] == 255)
    pr, pc = white_in_strip[len(white_in_strip) // 2]
    r = client.post(f"/api/session/{sid}/force",
                    json={"x0": int(pc), "y0": int(pr), "x1": int(pc) + 1,
                          "y1": int(pr) + 1, "mode": "black"})
    data = r.get_json()
    check("条带内改色被拒绝", r.status_code == 200 and data["result"]["accepted"] is False)
    viol = data["result"]["violations"]
    check("给出首个变色点且在条带内",
          len(viol) == 1 and viol[0]["first_point"][0] < 40)
    check("拒绝后仍只有第0稿（旧稿保留）",
          len(data["session"]["revisions"]) == 1)
    # 旧稿强制色仍为空
    check("旧稿强制色保留（仍为0个）",
          data["session"]["revisions"][0]["force"]["total"] == 0)
    # 被拒候选稿预览可取
    pr_resp = client.get(f"/api/session/{sid}/pending/image")
    check("被拒候选稿预览可下载", pr_resp.status_code == 200)

    # 缓存可回退验证：再做一次安全区改色，续算结果仍须等于从头重算
    # 安全区：第 90 行附近（条带之外），且挑一个实际翻转的像素
    # 先模拟候选 force 计算从头结果以挑选安全点
    test_force = np.zeros((h, w), np.uint8)
    # 选第 95 行第一个白点强制黑
    row_white = np.nonzero(out0[95] == 255)[0]
    pc2 = int(row_white[0])
    test_force[95, pc2] = 1
    expect = reference_dither(gray, test_force)
    # 确认该改动不波及条带 [0,40)
    check("安全区改动不影响已打样条带",
          np.array_equal(expect[:40], out0[:40]))

    r = client.post(f"/api/session/{sid}/force",
                    json={"x0": pc2, "y0": 95, "x1": pc2 + 1, "y1": 96,
                          "mode": "black"})
    data = r.get_json()
    check("安全区改色被接受", data["result"]["accepted"] is True)
    check("接受后产生第1稿", len(data["session"]["revisions"]) == 2)
    check("续算起点为检查点 floor32(95)=64", data["result"]["resume_start"] == 64)
    check("回退距离 95-64=31 ≤ 31", 95 - data["result"]["resume_start"] <= 31)

    out_resp = client.get(f"/api/session/{sid}/image/out")
    out1 = np.array(Image.open(io.BytesIO(out_resp.data)).convert("L"))
    check("接受稿与从头重算逐像素一致", np.array_equal(out1, expect))

    # 强制白与清除模式
    r = client.post(f"/api/session/{sid}/force",
                    json={"x0": 100, "y0": 100, "x1": 104, "y1": 104,
                          "mode": "white"})
    check("强制白请求 200", r.status_code == 200)
    r = client.post(f"/api/session/{sid}/force",
                    json={"x0": 100, "y0": 100, "x1": 104, "y1": 104,
                          "mode": "clear"})
    check("清除强制色请求 200", r.status_code == 200)

    # 多轮续算后仍与从头一致
    sess = S.get_session(sid)
    latest = S.current_revision(sess)
    full_again, _ = __import__("app.dither", fromlist=["dither_full"]).dither_full(
        gray, latest.force
    )
    check("多轮改稿后当前稿仍等于同 force 从头重算",
          np.array_equal(full_again, latest.out))

    # 下载
    dl = client.get(f"/api/session/{sid}/download")
    check("下载 PNG 200 且为合法灰度图",
          dl.status_code == 200 and
          Image.open(io.BytesIO(dl.data)).mode == "L")

    # 删除条带
    r = client.delete(f"/api/session/{sid}/strips/0")
    check("删除条带 200", r.status_code == 200 and r.get_json()["strips"] == [])


def test_sample_endpoint() -> None:
    print("test_sample_endpoint")
    r = client.post("/api/sample")
    check("样图载入 200", r.status_code == 200)
    sid = r.get_json()["session_id"]
    check("样图尺寸 256x256", r.get_json()["width"] == 256)
    img = client.get(f"/api/session/{sid}/image/gray")
    check("样图灰度可访问", img.status_code == 200)


def test_bad_session() -> None:
    print("test_bad_session")
    r = client.get("/api/session/nope/image/out")
    check("未知会话 400", r.status_code == 400)


if __name__ == "__main__":
    test_upload_validation()
    test_full_workflow()
    test_sample_endpoint()
    test_bad_session()
    print(f"\nAPI 测试完成：{PASS} 通过，{FAIL} 失败")
    sys.exit(1 if FAIL else 0)
