"""Flask 网页服务：上传、预览、框选强制色、条带管理、换稿与下载。"""

from __future__ import annotations

import io
import os
import math

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory, abort
from PIL import Image

from . import dither as D
from . import session as S

MAX_SIDE = S.MAX_SIDE
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample", "sample.png")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB 上传上限


# ---------------------------------------------------------------- 图片编码
def _png_bytes(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    return buf


def _binary_png(arr: np.ndarray) -> io.BytesIO:
    return _png_bytes(Image.fromarray(arr, mode="L"))


def _force_overlay(force: np.ndarray, base: np.ndarray) -> io.BytesIO:
    """强制色叠加图：强制黑=蓝框色，强制白=红，半透明覆盖在二值稿上。"""

    rgba = np.zeros((force.shape[0], force.shape[1], 4), dtype=np.uint8)
    rgb = np.repeat(base[:, :, None], 3, axis=2)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    # 强制黑：蓝色 (64,128,255)；强制白：红色 (255,80,80)
    rgba[force == 1] = (64, 128, 255, 220)
    rgba[force == 2] = (255, 80, 80, 220)
    return _png_bytes(Image.fromarray(rgba, mode="RGBA"))


def _diff_overlay(base_old: np.ndarray, base_new: np.ndarray,
                  points: list[tuple[int, int]]) -> io.BytesIO:
    """候选稿叠加图：变色像素用绿色描出，首个变色点用橙色十字标记。"""

    rgba = np.zeros((base_new.shape[0], base_new.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = np.repeat(base_new[:, :, None], 3, axis=2)
    rgba[:, :, 3] = 255
    changed = base_old != base_new
    rgba[changed] = (0, 220, 90, 200)
    for r, c in points:
        for dr in range(-3, 4):
            rr = r + dr
            if 0 <= rr < rgba.shape[0]:
                rgba[rr, c] = (255, 170, 0, 255)
        for dc in range(-3, 4):
            cc = c + dc
            if 0 <= cc < rgba.shape[1]:
                rgba[r, cc] = (255, 170, 0, 255)
    return _png_bytes(Image.fromarray(rgba, mode="RGBA"))


def _load_gray_png(stream) -> np.ndarray:
    try:
        img = Image.open(stream)
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise S.ValidationError(f"无法解析 PNG：{exc}") from exc
    if img.format != "PNG":
        raise S.ValidationError("只接受 PNG 文件")
    if img.mode != "L":
        raise S.ValidationError(
            f"只接受 8 位灰度 PNG（PIL mode=L），当前 mode={img.mode}，"
            "请先转换为灰度再上传"
        )
    w, h = img.size
    if w > MAX_SIDE or h > MAX_SIDE:
        raise S.ValidationError(f"宽高各至多 {MAX_SIDE} 像素，当前 {w}x{h}")
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------- 页面
@app.get("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ---------------------------------------------------------------- 上传/样图
@app.post("/api/upload")
def upload():
    file = request.files.get("image")
    if file is None:
        raise S.ValidationError("缺少 image 字段")
    gray = _load_gray_png(file.stream)
    sess = S.create_session(gray)
    return jsonify(_session_payload(sess))


@app.post("/api/sample")
def load_sample():
    if not os.path.exists(SAMPLE_PATH):
        abort(404, "样图不存在")
    with open(SAMPLE_PATH, "rb") as f:
        gray = _load_gray_png(f)
    sess = S.create_session(gray)
    return jsonify(_session_payload(sess))


# ---------------------------------------------------------------- 会话信息
def _session_payload(sess: S.Session) -> dict:
    with sess.lock:
        h, w = sess.gray.shape
        rev = S.current_revision(sess)
        return {
            "session_id": sess.session_id,
            "width": w,
            "height": h,
            "buffer_budget": {
                "checkpoint_rows": D.cache_slots(h),
                "workspace_rows": 2,
                "physical_rows": D.cache_slots(h) + 2,
                "block": D.BLOCK,
                "max_rollback": D.ROLLBACK,
            },
            "strips": [
                {"index": i, "top": s.top, "bottom": s.bottom,
                 "top_line": s.top + 1, "bottom_line": s.bottom, "label": s.label}
                for i, s in enumerate(sess.strips)
            ],
            "revisions": [
                {
                    "revision_id": r.revision_id,
                    "note": r.note,
                    "force": S.force_summary(r.force),
                    "resume_start": r.resume_start,
                    "changed_rows": r.changed_rows,
                    "buffer_stats": r.err.stats().__dict__,
                }
                for r in sess.revisions
            ],
            "current_revision_id": rev.revision_id,
            "pending": _pending_payload(sess),
        }


def _pending_payload(sess: S.Session) -> dict | None:
    p = sess.pending
    if p is None:
        return None
    return {
        "accepted": p.get("accepted", False),
        "violations": p.get("violations", []),
        "resume_start": p.get("resume_start"),
        "changed_rows": p.get("changed_rows", []),
        "mode": p.get("mode"),
        "rect": p.get("rect"),
        "note": p.get("note"),
    }


@app.get("/api/session/<sid>")
def session_info(sid: str):
    sess = S.get_session(sid)
    return jsonify(_session_payload(sess))


# ---------------------------------------------------------------- 图片
@app.get("/api/session/<sid>/image/gray")
def image_gray(sid: str):
    sess = S.get_session(sid)
    with sess.lock:
        return send_file(_png_bytes(Image.fromarray(sess.gray, mode="L")),
                         mimetype="image/png")


@app.get("/api/session/<sid>/image/out")
def image_out(sid: str):
    sess = S.get_session(sid)
    with sess.lock:
        rev = S.current_revision(sess)
        return send_file(_binary_png(rev.out), mimetype="image/png")


@app.get("/api/session/<sid>/image/force")
def image_force(sid: str):
    sess = S.get_session(sid)
    with sess.lock:
        rev = S.current_revision(sess)
        return send_file(_force_overlay(rev.force, rev.out), mimetype="image/png")


@app.get("/api/session/<sid>/pending/image")
def pending_image(sid: str):
    """被拒绝候选稿的叠加预览（绿=变色点，橙十字=条带内首个变色点）。"""

    sess = S.get_session(sid)
    with sess.lock:
        p = sess.pending
        if p is None:
            abort(404, "没有待处理的候选稿")
        rev = S.current_revision(sess)
        points = [tuple(v["first_point"]) for v in p.get("violations", [])]
        return send_file(
            _diff_overlay(rev.out, p["preview_out"], points), mimetype="image/png"
        )


@app.get("/api/session/<sid>/download")
def download(sid: str):
    sess = S.get_session(sid)
    with sess.lock:
        rev = S.current_revision(sess)
        buf = _binary_png(rev.out)
        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"halftone_{sid[:8]}_{rev.revision_id[:8]}.png",
        )


# ---------------------------------------------------------------- 强制色/条带
@app.post("/api/session/<sid>/force")
def apply_force(sid: str):
    sess = S.get_session(sid)
    data = request.get_json(force=True)
    try:
        x0 = int(data["x0"]); y0 = int(data["y0"])
        x1 = int(data["x1"]); y1 = int(data["y1"])
        mode = str(data["mode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise S.ValidationError(f"参数缺失或非法：{exc}") from exc
    result = S.evaluate_candidate(sess, x0, y0, x1, y1, mode)
    return jsonify({"result": result, "session": _session_payload(sess)})


@app.post("/api/session/<sid>/strips")
def add_strip(sid: str):
    sess = S.get_session(sid)
    data = request.get_json(force=True)
    try:
        # 网页使用 1 基行号：[top_line, bottom_line]
        top_line = int(data["top_line"])
        bottom_line = int(data["bottom_line"])
        label = str(data.get("label", ""))
    except (KeyError, TypeError, ValueError) as exc:
        raise S.ValidationError(f"参数缺失或非法：{exc}") from exc
    S.add_strip(sess, top_line - 1, bottom_line, label)
    return jsonify(_session_payload(sess))


@app.delete("/api/session/<sid>/strips/<int:index>")
def delete_strip(sid: str, index: int):
    sess = S.get_session(sid)
    S.remove_strip(sess, index)
    return jsonify(_session_payload(sess))


# ---------------------------------------------------------------- 错误处理
@app.errorhandler(S.ValidationError)
def _on_validation_error(err):
    return jsonify({"error": str(err)}), 400


@app.errorhandler(404)
def _on_404(err):
    return jsonify({"error": getattr(err, "description", "not found")}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
