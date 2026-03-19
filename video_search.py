# -*- coding: utf-8 -*-
"""
Поиск моментов в видео по фото или текстовому запросу (CLIP).
"""
import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
from tqdm import tqdm


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(device=None):
    device = device or get_device()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device).eval()
    return model, processor, device


def extract_frames(video_path, fps_sample=1.0, max_frames=300):
    """
    Извлекает кадры из видео.
    fps_sample: брать 1 кадр каждые N секунд (1.0 = раз в секунду)
    max_frames: максимум кадров (чтобы не перегружать память)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
    frame_interval = max(1, int(video_fps * fps_sample))
    frames = []
    timestamps = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            timestamps.append(frame_idx / video_fps)
        if len(frames) >= max_frames:
            break
        frame_idx += 1
    cap.release()
    return frames, timestamps, duration_sec


def encode_frames(model, processor, frames, device, batch_size=32):
    """Кодирует кадры в эмбеддинги CLIP."""
    embeddings = []
    for i in tqdm(range(0, len(frames), batch_size), desc="Кодирование кадров"):
        batch = frames[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            out = out / out.norm(dim=-1, keepdim=True)
        embeddings.append(out.cpu().numpy())
    return np.vstack(embeddings)


def encode_query_text(model, processor, text, device):
    """Кодирует текстовый запрос."""
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        out = out / out.norm(dim=-1, keepdim=True)
    return out.cpu().numpy().astype(np.float32)


def encode_query_image(model, processor, image, device):
    """Кодирует запрос-изображение (PIL Image)."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        out = out / out.norm(dim=-1, keepdim=True)
    return out.cpu().numpy().astype(np.float32)


def search_moments(
    video_path,
    query_image=None,
    query_text=None,
    top_k=10,
    fps_sample=1.0,
    max_frames=300,
    model=None,
    processor=None,
    device=None,
):
    """
    Ищет в видео моменты по фото и/или тексту.
    query_image: PIL Image или путь к файлу (опционально)
    query_text: строка, например "человек", "собака" (опционально)
    Возвращает список dict: [{"timestamp": sec, "score": float}, ...]
    """
    if query_image is None and query_text is None:
        raise ValueError("Укажите query_image и/или query_text")
    if model is None:
        model, processor, device = load_clip(device)
    else:
        device = device or get_device()
    frames, timestamps, duration_sec = extract_frames(
        video_path, fps_sample=fps_sample, max_frames=max_frames
    )
    if not frames:
        return []
    frame_emb = encode_frames(model, processor, frames, device)
    query_emb = None
    if query_image is not None:
        if isinstance(query_image, (str, os.PathLike)):
            img = Image.open(query_image).convert("RGB")
        else:
            img = query_image.convert("RGB") if hasattr(query_image, "convert") else query_image
        q = encode_query_image(model, processor, img, device)
        query_emb = q if query_emb is None else (query_emb + q) / 2
    if query_text is not None:
        q = encode_query_text(model, processor, query_text, device)
        query_emb = q if query_emb is None else (query_emb + q) / 2
    query_emb = query_emb.astype(np.float32)
    scores = np.dot(frame_emb, query_emb.T).flatten()
    order = np.argsort(-scores)[:top_k]
    return [
        {"timestamp": round(float(timestamps[i]), 2), "score": float(scores[i])}
        for i in order
    ]
