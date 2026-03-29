# -*- coding: utf-8 -*-
"""
Латентное представление лиц (эмбеддинги ArcFace/InsightFace) для быстрого поиска
сюжетных фрагментов по сущности «лицо». Кадры группируются во временные интервалы;
длина каждого фрагмента не превышает max_fragment_sec (по умолчанию 1 с).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Ленивая загрузка InsightFace (тяжёлая зависимость)
_face_app = None


def get_device_id() -> int:
    """CPU: -1, CUDA: 0."""
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def load_face_app(det_size: Tuple[int, int] = (640, 640)):
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis

        ctx = get_device_id()
        _face_app = FaceAnalysis(name="buffalo_l")
        _face_app.prepare(ctx_id=ctx, det_size=det_size)
    return _face_app


def _largest_face(faces):
    if not faces:
        return None
    best = None
    best_area = 0
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = f
    return best


def _normalize(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb)
    if n < 1e-8:
        return emb
    return (emb / n).astype(np.float32)


def face_embedding_from_bgr(app, bgr: np.ndarray) -> Optional[np.ndarray]:
    """Одно лицо на кадре — берём самое крупное; эмбеддинг в латентном пространстве."""
    faces = app.get(bgr)
    f = _largest_face(faces)
    if f is None:
        return None
    emb = getattr(f, "normed_embedding", None)
    if emb is None:
        emb = _normalize(np.asarray(f.embedding, dtype=np.float32))
    else:
        emb = np.asarray(emb, dtype=np.float32)
    return emb


def query_embedding_from_path(image_path: str, app=None) -> np.ndarray:
    """Латент запроса по фото (лицо на снимке)."""
    app = app or load_face_app()
    bgr = cv2.imread(image_path)
    if bgr is None:
        pil = Image.open(image_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    emb = face_embedding_from_bgr(app, bgr)
    if emb is None:
        raise ValueError("На эталонном фото не найдено лицо — загрузите чёткий портрет.")
    return emb


def extract_frames_bgr(
    video_path: str,
    fps_sample: float = 1.0,
    max_frames: int = 2000,
) -> Tuple[List[np.ndarray], List[float], float]:
    """Извлекает кадры BGR и таймкоды."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(video_fps, 1e-6)
    frame_interval = max(1, int(video_fps * fps_sample))
    frames_bgr = []
    timestamps = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames_bgr.append(frame)
            timestamps.append(frame_idx / video_fps)
        if len(frames_bgr) >= max_frames:
            break
        frame_idx += 1
    cap.release()
    return frames_bgr, timestamps, duration_sec


def group_hits_to_intervals(
    timestamps: Sequence[float],
    scores: Sequence[float],
    merge_gap_sec: float = 0.4,
    max_fragment_sec: float = 1.0,
) -> List[dict]:
    """
    Группирует кадры-попадания во временные интервалы (близкие по времени сливаются),
    затем режет каждый интервал на фрагменты длиной не более max_fragment_sec.
    """
    if not timestamps:
        return []
    ts = np.asarray(timestamps, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    order = np.argsort(ts)
    ts = ts[order]
    sc = sc[order]

    runs: List[Tuple[np.ndarray, np.ndarray]] = []
    cur_t = [ts[0]]
    cur_s = [sc[0]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] <= merge_gap_sec:
            cur_t.append(ts[i])
            cur_s.append(sc[i])
        else:
            runs.append((np.array(cur_t), np.array(cur_s)))
            cur_t = [ts[i]]
            cur_s = [sc[i]]
    runs.append((np.array(cur_t), np.array(cur_s)))

    out: List[dict] = []
    for t_arr, s_arr in runs:
        t0 = float(np.min(t_arr))
        t1 = float(np.max(t_arr))
        span = t1 - t0
        if span <= max_fragment_sec:
            out.append(
                {
                    "start": round(t0, 2),
                    "end": round(t1, 2),
                    "score": float(np.max(s_arr)),
                    "timestamp": round(t0, 2),
                }
            )
            continue
        seg = t0
        while seg < t1 + 1e-6:
            seg_end = min(seg + max_fragment_sec, t1)
            mask = (t_arr >= seg - 1e-6) & (t_arr <= seg_end + 1e-6)
            seg_score = float(np.max(s_arr[mask])) if np.any(mask) else float(np.max(s_arr))
            out.append(
                {
                    "start": round(seg, 2),
                    "end": round(seg_end, 2),
                    "score": seg_score,
                    "timestamp": round(seg, 2),
                }
            )
            seg += max_fragment_sec
    return out


def search_face_latent(
    video_path: str,
    query_image_path: str,
    fps_sample: float = 0.5,
    max_frames: int = 2000,
    similarity_threshold: float = 0.35,
    merge_gap_sec: float = 0.4,
    max_fragment_sec: float = 1.0,
    app=None,
) -> Tuple[List[dict], List[dict]]:
    """
    Ищет фрагменты видео, где лицо близко к эталону в латентном пространстве.

    Возвращает:
      fragments — интервалы [start, end] с score (длина каждого ≤ max_fragment_sec);
      moments — для совместимости с плеером: {timestamp, score} по началу фрагмента.
    """
    app = app or load_face_app()
    query_emb = query_embedding_from_path(query_image_path, app=app)

    frames_bgr, timestamps, _ = extract_frames_bgr(
        video_path, fps_sample=fps_sample, max_frames=max_frames
    )
    if not frames_bgr:
        return [], []

    hit_ts: List[float] = []
    hit_sc: List[float] = []

    for bgr, t in tqdm(list(zip(frames_bgr, timestamps)), desc="Латент лиц по кадрам"):
        emb = face_embedding_from_bgr(app, bgr)
        if emb is None:
            continue
        sim = float(np.dot(query_emb, emb))
        if sim >= similarity_threshold:
            hit_ts.append(float(t))
            hit_sc.append(sim)

    fragments = group_hits_to_intervals(
        hit_ts,
        hit_sc,
        merge_gap_sec=merge_gap_sec,
        max_fragment_sec=max_fragment_sec,
    )
    moments = [
        {"timestamp": f["timestamp"], "score": f["score"], "start": f["start"], "end": f["end"]}
        for f in fragments
    ]
    return fragments, moments
