"""Streamer layout: the creator's webcam above the gameplay or reacted content.

The user picks this framing explicitly, so unlike the automatic reaction
layout it does not need a second face in the source: gameplay usually has
none. The webcam is the one small face that stays put for most of the clip.

Output is 1080x1920, split evenly: the webcam crop on top (1080x960) and a
crop of the content beside the webcam below (1080x960). Without a webcam the
clip falls back to the full-frame fit.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import UPSCALE_SHARPEN_FILTER
from .ffmpeg import clamp_even, round_to_even

Box = Tuple[float, float, float, float]
FaceFrame = Tuple[float, Sequence[Box]]

PANEL_WIDTH = 1080
PANEL_HEIGHT = 960
PANEL_ASPECT = PANEL_WIDTH / PANEL_HEIGHT
# Captions sit on the seam between webcam and content.
STREAMER_CAPTION_Y = 0.5

# A webcam overlay is a small part of the frame. A face filling more than this
# is a full-screen camera shot, which the other framings already handle.
MAX_WEBCAM_FACE_AREA = 0.06
MIN_WEBCAM_FACE_AREA = 0.0008
# Present in at least this share of the sampled frames.
MIN_PRESENCE = 0.5


def find_webcam_face(width: int, height: int, samples: Sequence[FaceFrame]) -> Optional[Box]:
    """The small face that stays in one place for most of the clip.

    Faces in the gameplay or video move, cut, and come and go; the webcam does
    not. Among stable positions, the one seen most often wins, with ties
    going to the face nearer a frame corner, where overlays usually sit.
    """
    if len(samples) < 3:
        return None
    frame_area = float(width * height)
    clusters: List[List[Box]] = []
    for _, faces in samples:
        for face in faces:
            area = face[2] * face[3] / frame_area
            if not MIN_WEBCAM_FACE_AREA <= area <= MAX_WEBCAM_FACE_AREA:
                continue
            cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
            for cluster in clusters:
                ref = cluster[0]
                if (abs(cx - (ref[0] + ref[2] / 2)) <= max(ref[2], face[2])
                        and abs(cy - (ref[1] + ref[3] / 2)) <= max(ref[3], face[3])):
                    cluster.append(face)
                    break
            else:
                clusters.append([face])

    best: Optional[Tuple[float, float, Box]] = None
    for cluster in clusters:
        presence = len(cluster) / len(samples)
        if presence < MIN_PRESENCE:
            continue
        boxes = np.asarray(cluster)
        centres = boxes[:, :2] + boxes[:, 2:] / 2
        if (np.std(centres[:, 0]) > width * .03
                or np.std(centres[:, 1]) > height * .04
                or np.std(boxes[:, 2]) > np.median(boxes[:, 2]) * .3):
            continue
        face = tuple(float(v) for v in np.median(boxes, axis=0))
        cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
        corner = min(cx, width - cx) / width + min(cy, height - cy) / height
        key = (round(presence, 1), -corner)
        if best is None or key > best[:2]:
            best = (key[0], key[1], face)
    return best[2] if best else None


def _peak(values: np.ndarray, start: float, end: float, nearest_last: bool) -> Optional[int]:
    """Boundary index of the strongest persistent edge in [start, end).

    The returned index is the first pixel past the edge. ``nearest_last``
    breaks near-ties toward the end of the range, i.e. toward the face.
    """
    start, end = max(0, int(start)), min(len(values), int(end))
    if end - start < 3:
        return None
    region = values[start:end]
    strongest = float(region.max())
    if strongest < max(8.0, float(np.median(region)) * 2.5):
        return None
    candidates = np.flatnonzero(region >= strongest * .85)
    index = int(candidates[-1] if nearest_last else candidates[0])
    return start + index + 1


def measure_webcam_box(path: Path, width: int, height: int, duration: float, face: Box) -> Optional[Box]:
    """Find the webcam overlay's rectangle around ``face``.

    The overlay border is the edge that stays put while the gameplay moves:
    take the median edge strength over frames across the clip, then look for
    the strongest edge on each side of the face. A side with no edge before
    the frame border is the frame border. Returns None when the result does
    not look like a camera window.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        for t in np.linspace(0, max(0., duration - .2), 8):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, frame = cap.read()
            if ok and frame.shape[:2] == (height, width):
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    finally:
        cap.release()
    if len(frames) < 4:
        return None
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2
    reach_x, reach_y = fw * 5, fh * 4

    # Columns: rows around the face only, so chat or HUD below cannot win.
    row0, row1 = int(max(0, cy - fh)), int(min(height, cy + fh))
    columns = np.median([np.abs(np.diff(f[row0:row1], axis=1)).mean(axis=0) for f in frames], axis=0)
    left_range = (cx - reach_x, fx - fw * .15)
    right_range = (fx + fw * 1.15, cx + reach_x)
    left = _peak(columns, *left_range, nearest_last=True)
    right = _peak(columns, *right_range, nearest_last=False)
    left = left if left is not None else (0 if left_range[0] <= 0 else None)
    right = right if right is not None else (width if right_range[1] >= width - 1 else None)
    if left is None or right is None:
        return None

    rows = np.median([np.abs(np.diff(f[:, left:right], axis=0)).mean(axis=1) for f in frames], axis=0)
    top_range = (cy - reach_y, fy - fh * .1)
    bottom_range = (fy + fh * 1.3, cy + reach_y)
    top = _peak(rows, *top_range, nearest_last=True)
    bottom = _peak(rows, *bottom_range, nearest_last=False)
    top = top if top is not None else (0 if top_range[0] <= 0 else None)
    bottom = bottom if bottom is not None else (height if bottom_range[1] >= height - 1 else None)
    if top is None or bottom is None:
        return None

    box_w, box_h = float(right - left), float(bottom - top)
    if (box_w < fw * 1.6 or box_h < fh * 1.5
            or not .4 <= box_w / box_h <= 2.5
            or box_w * box_h > width * height * .4):
        return None
    # Stay just inside the border: frames, glows and neighbouring chat lines
    # sit on the edge itself. Frame borders need no margin.
    mx, my = max(2.0, box_w * .02), max(2.0, box_h * .02)
    x0 = left + (mx if left > 0 else 0)
    y0 = top + (my if top > 0 else 0)
    x1 = right - (mx if right < width else 0)
    y1 = bottom - (my if bottom < height else 0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def _fallback_webcam_box(width: int, height: int, face: Box) -> Box:
    """A head-and-shoulders box around the face when no overlay edge is found."""
    fx, fy, fw, fh = face
    bw = min(float(width), fw * 2.6)
    bh = min(float(height), bw / PANEL_ASPECT)
    bx = min(max(0.0, fx + fw / 2 - bw / 2), width - bw)
    by = min(max(0.0, fy + fh / 2 - bh * .45), height - bh)
    return bx, by, bw, bh


def _webcam_crop(face: Box, box: Box) -> Tuple[int, int, int, int]:
    """A 9:8 crop inside the webcam box, centred on the face."""
    bx, by, bw, bh = box
    cw = round_to_even(int(min(bw, bh * PANEL_ASPECT)))
    ch = round_to_even(int(cw / PANEL_ASPECT))
    cx = clamp_even(int(face[0] + face[2] / 2 - cw / 2), int(bx), int(bx + bw - cw))
    cy = clamp_even(int(face[1] + face[3] / 2 - ch * .45), int(by), int(by + bh - ch))
    return cw, ch, cx, cy


def _content_crop(width: int, height: int, box: Box) -> Tuple[int, int, int, int]:
    """A 9:8 crop of the content, beside the webcam column when there is room.

    Streams usually stack the webcam over chat or alerts in a side column; the
    gameplay is the rest of the frame. When the webcam floats over the middle,
    or the side left over is too narrow, use the whole frame width.
    """
    bx, _, bw, _ = box
    left_w, right_w = bx, width - (bx + bw)
    if max(left_w, right_w) >= width * .5:
        region_x, region_w = (0.0, left_w) if left_w >= right_w else (bx + bw, right_w)
    else:
        region_x, region_w = 0.0, float(width)
    cw = round_to_even(int(min(region_w, height * PANEL_ASPECT)))
    ch = round_to_even(int(min(height, cw / PANEL_ASPECT)))
    cx = clamp_even(int(region_x + (region_w - cw) / 2), 0, width - cw)
    cy = clamp_even(int((height - ch) / 2), 0, height - ch)
    return cw, ch, cx, cy


def build_streamer_filter(width: int, height: int, face: Box, box: Box) -> str:
    """filter_complex producing [vout]: webcam over content, 50/50."""
    rw, rh, rx, ry = _webcam_crop(face, box)
    cw, ch, cx, cy = _content_crop(width, height, box)
    return ";".join([
        "[0:v]split=2[cam_src][content_src]",
        f"[cam_src]crop={rw}:{rh}:{rx}:{ry},"
        f"scale={PANEL_WIDTH}:{PANEL_HEIGHT}:flags=lanczos,{UPSCALE_SHARPEN_FILTER},setsar=1[cam]",
        f"[content_src]crop={cw}:{ch}:{cx}:{cy},"
        f"scale={PANEL_WIDTH}:{PANEL_HEIGHT}:flags=lanczos,setsar=1[content]",
        "[cam][content]vstack=inputs=2,setsar=1[vout]",
    ])


def choose_streamer_layout(
    path: Path, width: int, height: int, duration: float, samples: Sequence[FaceFrame],
) -> Optional[Dict[str, Any]]:
    """The streamer plan, or None when no webcam face is found."""
    face = find_webcam_face(width, height, samples)
    if face is None:
        return None
    try:
        box = measure_webcam_box(path, width, height, duration, face)
    except Exception:
        box = None
    # No clear overlay border (a borderless camera, or busy content right
    # next to it): frame the head and shoulders instead.
    edge_found = box is not None
    if box is None:
        box = _fallback_webcam_box(width, height, face)
    return {
        "kind": "streamer",
        "video_filter": build_streamer_filter(width, height, face, box),
        "caption_y": STREAMER_CAPTION_Y,
        "face": face,
        "webcam": box,
        "edge_found": edge_found,
    }
