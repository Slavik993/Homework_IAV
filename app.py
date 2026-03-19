# -*- coding: utf-8 -*-
"""
Веб-приложение: аннотирование видео — поиск по фото или тексту, результат в плеере с гиперссылками на моменты.
"""
import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from video_search import load_clip, search_moments

app = Flask(__name__, static_folder="static")
CORS(app)

UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB для видео

ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "gif", "webp"}


def allowed_file(filename, ext_set):
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ext_set


# Глобальная загрузка модели при старте (ленивая при первом поиске)
_clip_state = {"model": None, "processor": None, "device": None}


def get_clip():
    if _clip_state["model"] is None:
        _clip_state["model"], _clip_state["processor"], _clip_state["device"] = load_clip()
    return _clip_state["model"], _clip_state["processor"], _clip_state["device"]


@app.route("/")
def index():
    return send_file(Path(__file__).resolve().parent / "static" / "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.route("/api/upload-video", methods=["POST"])
def upload_video():
    if "video" not in request.files and "file" not in request.files:
        return jsonify({"error": "Нет файла видео"}), 400
    f = request.files.get("video") or request.files.get("file")
    if not f or f.filename == "":
        return jsonify({"error": "Файл не выбран"}), 400
    if not allowed_file(f.filename, ALLOWED_VIDEO):
        return jsonify({"error": "Разрешены форматы: " + ", ".join(ALLOWED_VIDEO)}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower()
    name = f"{uuid.uuid4().hex}.{ext}"
    path = Path(app.config["UPLOAD_FOLDER"]) / name
    f.save(path)
    return jsonify({"filename": name, "url": f"/api/video/{name}"})


@app.route("/api/upload-image", methods=["POST"])
def upload_image():
    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "Нет файла изображения"}), 400
    f = request.files.get("image") or request.files.get("file")
    if not f or f.filename == "":
        return jsonify({"error": "Файл не выбран"}), 400
    if not allowed_file(f.filename, ALLOWED_IMAGE):
        return jsonify({"error": "Разрешены форматы: " + ", ".join(ALLOWED_IMAGE)}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower()
    name = f"{uuid.uuid4().hex}.{ext}"
    path = Path(app.config["UPLOAD_FOLDER"]) / name
    f.save(path)
    return jsonify({"filename": name, "url": f"/api/image/{name}"})


@app.route("/api/video/<filename>")
def serve_video(filename):
    path = Path(app.config["UPLOAD_FOLDER"]) / secure_filename(filename)
    if not path.is_file():
        return jsonify({"error": "Видео не найдено"}), 404
    return send_file(path, mimetype="video/mp4", as_attachment=False)


@app.route("/api/image/<filename>")
def serve_image(filename):
    path = Path(app.config["UPLOAD_FOLDER"]) / secure_filename(filename)
    if not path.is_file():
        return jsonify({"error": "Изображение не найдено"}), 404
    return send_file(path, as_attachment=False)


@app.route("/api/search", methods=["POST"])
def search():
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict(flat=True) if request.form else {}
    video_filename = data.get("video_filename") or data.get("video")
    query_text = (data.get("query_text") or data.get("text") or "").strip()
    image_filename = data.get("image_filename") or data.get("image")
    top_k = int(data.get("top_k", 15))
    fps_sample = float(data.get("fps_sample", 1.0))
    max_frames = int(data.get("max_frames", 300))

    if not video_filename:
        return jsonify({"error": "Укажите video_filename (после загрузки видео)"}), 400
    if not query_text and not image_filename:
        return jsonify({"error": "Укажите текст запроса и/или загрузите фото"}), 400

    video_path = Path(app.config["UPLOAD_FOLDER"]) / secure_filename(video_filename)
    if not video_path.is_file():
        return jsonify({"error": "Видео не найдено на сервере"}), 404

    query_image = None
    if image_filename:
        img_path = Path(app.config["UPLOAD_FOLDER"]) / secure_filename(image_filename)
        if img_path.is_file():
            query_image = str(img_path)

    model, processor, device = get_clip()
    try:
        moments = search_moments(
            str(video_path),
            query_image=query_image,
            query_text=query_text or None,
            top_k=top_k,
            fps_sample=fps_sample,
            max_frames=max_frames,
            model=model,
            processor=processor,
            device=device,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "video_url": f"/api/video/{video_filename}",
        "moments": moments,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
