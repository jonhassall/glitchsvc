import argparse
import os
import random
import subprocess
import uuid
from collections import Counter

import cv2
import numpy as np

# Reduce noisy non-fatal backend warnings (for example unsupported NNPACK).
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# === CONFIG ===
OUTPUT_ROOT_DIR = "glitched_outputs"
TEMP_DIR = "tmp_svc"
BASE_LAYER_FILE = os.path.join(TEMP_DIR, "base_layer.264")
ENH_LAYER_FILE = os.path.join(TEMP_DIR, "enh_layer.264")
GLITCHED_FILE = os.path.join(TEMP_DIR, "glitched.264")

NUM_OUTPUTS = 14
GLITCH_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 0.99]

GLITCH_TYPES = {
    "random": "Random byte corruption",
    "zero": "Byte zeroing (blocky artifacts)",
    "block": "Block corruption (large artifacts)",
    "constant": "Constant damage throughout NAL",
    "interval": "Damage at regular intervals",
    "keyframe": "Obliterate keyframes (I-frames)",
    "keyframe_destroy": "Completely destroy keyframes (total annihilation)",
}


def log_stage(stage, total_stages, message):
    pct = int((stage / total_stages) * 100)
    print(f"[stage {stage}/{total_stages} | {pct}%] {message}")


def update_progress(label, current, total, last_reported=-1, step=10):
    if total <= 0:
        return last_reported
    pct = int((current / total) * 100)
    if pct >= 100 and last_reported < 100:
        print(f"[{label}] 100% ({current}/{total})")
        return 100
    if pct // step > last_reported // step:
        print(f"[{label}] {pct}% ({current}/{total})")
        return pct
    return last_reported


def run_cmd(cmd, quiet=False, capture_output=False):
    kwargs = {"check": True}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    if capture_output:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    return subprocess.run(cmd, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate glitched video variants from an input file")
    parser.add_argument("input_file", help="Path to the input video file")
    parser.add_argument("--yolo", action="store_true", help="Enable YOLO overlays on generated outputs")
    parser.add_argument(
        "--yolo-mode",
        choices=["segmentation", "box"],
        default="segmentation",
        help="Overlay mode when YOLO is enabled",
    )
    parser.add_argument("--yolo-model", default="yolo11n-seg.pt", help="YOLO model path/name")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--whisper", action="store_true", help="Enable Whisper word captions on generated outputs")
    parser.add_argument("--whisper-model", default="base", help="Whisper model name (tiny/base/small/medium/large)")
    parser.add_argument("--emotion", action="store_true", help="Enable emotion analysis overlay on detected faces")
    parser.add_argument("--emotion-hz", type=float, default=2.0, help="Emotion analysis frequency in Hz (default: 2)")
    return parser.parse_args()


def get_input_framerate(input_file):
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_file,
        ],
        capture_output=True,
    )
    framerate = result.stdout.strip()
    if not framerate:
        raise RuntimeError("Could not read input framerate from ffprobe output")
    return framerate


def find_nal_indices(h264_bytes):
    """Return (offset, start_code_len) for every NAL start code.

    Uses bytes.find() to avoid creating a temporary slice object at every
    byte position, which was the dominant source of GC pressure in large
    files.
    """
    indices = []
    i = 0
    n = len(h264_bytes)
    while i < n - 3:
        j = h264_bytes.find(b"\x00\x00\x01", i)
        if j == -1:
            break
        # A leading zero makes this a 4-byte start code (\x00\x00\x00\x01).
        if j >= 1 and h264_bytes[j - 1] == 0:
            indices.append((j - 1, 4))
        else:
            indices.append((j, 3))
        i = j + 3
    return indices


def corrupt_nal(data, nal_data_start, nal_end, glitch_type):
    if glitch_type == "random":
        corrupt_pos = random.randint(nal_data_start + 10, nal_end - 1)
        data[corrupt_pos] = random.randint(0, 255)

    elif glitch_type == "zero":
        corrupt_pos = random.randint(nal_data_start + 10, nal_end - 1)
        data[corrupt_pos] = 0

    elif glitch_type == "block":
        block_size = random.randint(5, 20)
        start_pos = random.randint(nal_data_start + 10, max(nal_data_start + 11, nal_end - block_size))
        for offset in range(block_size):
            if start_pos + offset < nal_end:
                data[start_pos + offset] = random.randint(0, 255)

    elif glitch_type == "constant":
        nal_length = nal_end - nal_data_start - 10
        num_corruptions = max(5, nal_length // 20)
        for _ in range(num_corruptions):
            corrupt_pos = random.randint(nal_data_start + 10, nal_end - 1)
            data[corrupt_pos] = random.randint(0, 255)

    elif glitch_type == "interval":
        interval = random.randint(15, 40)
        pos = nal_data_start + 10
        while pos < nal_end - 1:
            data[pos] = random.randint(0, 255)
            pos += interval

    elif glitch_type == "keyframe":
        nal_length = nal_end - nal_data_start - 10
        num_corruptions = max(20, nal_length // 5)
        for _ in range(num_corruptions):
            corrupt_pos = random.randint(nal_data_start + 10, nal_end - 1)
            data[corrupt_pos] = random.randint(0, 255)

    elif glitch_type == "keyframe_destroy":
        nal_length = nal_end - nal_data_start - 10
        num_corruptions = max(50, int(nal_length * 0.7))
        corrupted_positions = set()
        for _ in range(num_corruptions):
            corrupt_pos = random.randint(nal_data_start + 10, nal_end - 1)
            if corrupt_pos not in corrupted_positions:
                data[corrupt_pos] = random.randint(0, 255)
                corrupted_positions.add(corrupt_pos)


class YoloAnnotator:
    def __init__(self, enabled, mode, model_path, conf):
        self.enabled = enabled
        self.mode = mode
        self.conf = conf
        self.font = self._load_font(size=16)
        self.alpha = int(255 * 0.75)
        self.border_color = (255, 70, 70, self.alpha)
        self.text_color = (255, 255, 255, self.alpha)
        self.fallback_used = False
        self.device = "cpu"
        self.model = None
        self.cached_by_frame = []

        if self.enabled:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Loading YOLO model '{model_path}' on device '{self.device}'...")
            self.model = YOLO(model_path)

    @staticmethod
    def _load_font(size=16):
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
        for path in font_paths:
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def _infer(self, frame):
        try:
            return self.model(frame, conf=self.conf, verbose=False, device=self.device)[0]
        except Exception as exc:
            if self.device.startswith("cuda") and not self.fallback_used:
                print(f"CUDA inference failed ({exc}). Falling back to CPU and retrying once...")
                self.device = "cpu"
                self.fallback_used = True
                return self.model(frame, conf=self.conf, verbose=False, device=self.device)[0]
            raise

    def _get_class_name(self, result, idx):
        name = "object"
        if hasattr(result, "names") and result.names is not None:
            if result.boxes is not None and result.boxes.cls is not None and idx < len(result.boxes.cls):
                cls_id = int(result.boxes.cls[idx].item())
                if isinstance(result.names, dict):
                    name = str(result.names.get(cls_id, name))
                elif isinstance(result.names, (list, tuple)) and 0 <= cls_id < len(result.names):
                    name = str(result.names[cls_id])
        return name

    def _get_label(self, result, idx, conf):
        name = self._get_class_name(result, idx)
        return f"{name} {int(conf * 100)}%"

    def _extract_frame_detections(self, result, width, height):
        frame_detections = []
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return frame_detections

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes_xyxy), dtype=float)
        masks_xy = result.masks.xy if (result.masks is not None and hasattr(result.masks, "xy")) else None

        for idx, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = max(0.0, min(float(width - 1), x1))
            y1 = max(0.0, min(float(height - 1), y1))
            x2 = max(0.0, min(float(width - 1), x2))
            y2 = max(0.0, min(float(height - 1), y2))

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            conf = float(confs[idx]) if idx < len(confs) else 1.0
            class_name = self._get_class_name(result, idx)
            label = f"{class_name} {int(conf * 100)}%"

            det = {
                "box": [x1 / width, y1 / height, x2 / width, y2 / height],
                "label": label,
                "class_name": class_name,
                "mask": None,
            }

            if masks_xy is not None and idx < len(masks_xy):
                pts = masks_xy[idx]
                if pts is not None and len(pts) >= 2:
                    normalized_pts = []
                    for p in pts:
                        px = max(0.0, min(float(width - 1), float(p[0])))
                        py = max(0.0, min(float(height - 1), float(p[1])))
                        normalized_pts.append([px / width, py / height])
                    if len(normalized_pts) >= 2:
                        det["mask"] = normalized_pts

            frame_detections.append(det)

        return frame_detections

    def build_cache(self, input_video_path):
        if not self.enabled:
            return

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video for YOLO analysis: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError(f"Invalid input dimensions for YOLO analysis: {input_video_path}")

        print("Running one-time YOLO analysis cache on input video...")
        self.cached_by_frame = []
        analyzed_frames = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        last_progress = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = self._infer(frame)
            self.cached_by_frame.append(self._extract_frame_detections(result, width, height))
            analyzed_frames += 1
            last_progress = update_progress("YOLO cache", analyzed_frames, total_frames, last_progress)

        cap.release()
        print(f"YOLO cache ready: {analyzed_frames} frame(s) analyzed once")

    def draw_from_cache(self, frame, frame_index):
        if not self.enabled:
            return frame
        if frame_index >= len(self.cached_by_frame):
            return frame

        detections = self.cached_by_frame[frame_index]
        if not detections:
            return frame

        h, w = frame.shape[:2]
        base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for det in detections:
            x1 = int(det["box"][0] * w)
            y1 = int(det["box"][1] * h)
            x2 = int(det["box"][2] * w)
            y2 = int(det["box"][3] * h)

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            drew_seg = False
            if self.mode == "segmentation" and det["mask"] is not None:
                pts = []
                for p in det["mask"]:
                    px = int(max(0, min(w - 1, p[0] * w)))
                    py = int(max(0, min(h - 1, p[1] * h)))
                    pts.append((px, py))
                if len(pts) >= 2:
                    draw.line(pts + [pts[0]], fill=self.border_color, width=2)
                    drew_seg = True

            if self.mode == "box" or not drew_seg:
                draw.rectangle([(x1, y1), (x2, y2)], outline=self.border_color, width=2)

            label = det["label"]
            tb = draw.textbbox((0, 0), label, font=self.font)
            tw = max(1, tb[2] - tb[0])
            th = max(1, tb[3] - tb[1])
            tx = max(0, min(w - tw - 1, x2 - tw - 2))
            ty = max(0, min(h - th - 1, y2 - th - 2))
            draw.text((tx, ty), label, font=self.font, fill=self.text_color)

        class_counts = Counter(det.get("class_name", "object") for det in detections)
        if class_counts:
            self._draw_object_hud(draw, class_counts, w, h)

        composed = Image.alpha_composite(base, overlay)
        return cv2.cvtColor(np.array(composed), cv2.COLOR_RGBA2BGR)

    def _draw_object_hud(self, draw, class_counts, w, h):
        lines = [f"{cls}: {cnt}" for cls, cnt in sorted(class_counts.items())]
        padding = 6
        line_height = 0
        max_line_w = 0
        for line in lines:
            tb = draw.textbbox((0, 0), line, font=self.font)
            lw = tb[2] - tb[0]
            lh = tb[3] - tb[1]
            max_line_w = max(max_line_w, lw)
            line_height = max(line_height, lh)
        total_w = max_line_w + padding * 2
        total_h = line_height * len(lines) + padding * 2
        draw.rectangle([(0, 0), (total_w, total_h)], fill=(0, 0, 0, 180))
        for i, line in enumerate(lines):
            draw.text((padding, padding + i * line_height), line, font=self.font, fill=self.text_color)


class WhisperCaptioner:
    def __init__(self, enabled, model_name="base"):
        self.enabled = enabled
        self.model_name = model_name
        self.word_segments = []  # list of {start, end, word}
        self.font = YoloAnnotator._load_font(size=24)
        self.bg_color = (0, 0, 0, 160)
        self.text_color = (255, 255, 255, 255)

    def build_cache(self, input_video_path):
        if not self.enabled:
            return
        import whisper

        audio_path = os.path.join(TEMP_DIR, f"audio_{uuid.uuid4().hex}.wav")
        try:
            print("Extracting audio for Whisper transcription...")
            run_cmd(
                ["ffmpeg", "-y", "-i", input_video_path, "-vn", "-ar", "16000", "-ac", "1", audio_path],
                quiet=True,
            )
            print(f"Loading Whisper model '{self.model_name}'...")
            model = whisper.load_model(self.model_name)
            print("Running Whisper transcription with word timestamps...")
            result = model.transcribe(audio_path, word_timestamps=True)
            self.word_segments = []
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    self.word_segments.append(
                        {
                            "start": word_info["start"],
                            "end": word_info["end"],
                            "word": word_info["word"].strip(),
                        }
                    )
            print(f"Whisper: {len(self.word_segments)} word(s) transcribed")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def _get_active_words(self, frame_index, fps):
        if not self.word_segments:
            return ""
        t = frame_index / fps if fps > 0 else 0
        active = [w["word"] for w in self.word_segments if w["start"] <= t <= w["end"]]
        past = [w["word"] for w in self.word_segments if w["end"] < t]
        context = past[-4:] + active
        return " ".join(context).strip()

    def draw_caption(self, frame, frame_index, fps):
        if not self.enabled:
            return frame
        text = self._get_active_words(frame_index, fps)
        if not text:
            return frame

        h, w = frame.shape[:2]
        base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        max_text_w = int(w * 0.9)
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            test = " ".join(current_line + [word])
            tb = draw.textbbox((0, 0), test, font=self.font)
            if tb[2] - tb[0] > max_text_w and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            lines.append(" ".join(current_line))
        lines = lines[-2:]

        sample_tb = draw.textbbox((0, 0), "Ag", font=self.font)
        line_h = sample_tb[3] - sample_tb[1] + 4
        total_h = line_h * len(lines) + 8
        y_start = h - total_h - 20

        max_line_w = max((draw.textbbox((0, 0), ln, font=self.font)[2] for ln in lines), default=1)
        x_pad = 10
        bg_x1 = (w - max_line_w) // 2 - x_pad
        bg_x2 = (w + max_line_w) // 2 + x_pad
        draw.rectangle([(bg_x1, y_start - 4), (bg_x2, y_start + total_h)], fill=self.bg_color)

        for i, line in enumerate(lines):
            tb = draw.textbbox((0, 0), line, font=self.font)
            lw = tb[2] - tb[0]
            lx = (w - lw) // 2
            ly = y_start + i * line_h
            draw.text((lx, ly), line, font=self.font, fill=self.text_color)

        composed = Image.alpha_composite(base, overlay)
        return cv2.cvtColor(np.array(composed), cv2.COLOR_RGBA2BGR)


class EmotionAnalyzer:
    EMOTION_COLORS = {
        "happy": (0, 255, 100, 220),
        "sad": (100, 100, 255, 220),
        "angry": (255, 60, 60, 220),
        "fear": (200, 100, 255, 220),
        "surprise": (0, 230, 230, 220),
        "disgust": (60, 200, 100, 220),
        "neutral": (200, 200, 200, 220),
    }
    DEFAULT_COLOR = (255, 255, 255, 220)

    def __init__(self, enabled, hz=2.0):
        self.enabled = enabled
        self.hz = hz
        self.cached_by_frame = []
        self.font = YoloAnnotator._load_font(size=14)

    def _analyze_frame(self, frame):
        try:
            from deepface import DeepFace

            results = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if not isinstance(results, list):
                results = [results]
            faces = []
            for r in results:
                region = r.get("region", {})
                x = region.get("x", 0)
                y = region.get("y", 0)
                rw = region.get("w", 0)
                rh = region.get("h", 0)
                if rw <= 0 or rh <= 0:
                    continue
                emotion = r.get("dominant_emotion", "neutral")
                conf = r.get("emotion", {}).get(emotion, 0.0)
                faces.append({"box": [x, y, x + rw, y + rh], "emotion": emotion, "conf": float(conf)})
            return faces
        except Exception:
            return []

    def build_cache(self, input_video_path):
        if not self.enabled:
            return

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for emotion analysis: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_interval = max(1, int(round(fps / self.hz)))

        print(f"Running emotion analysis at {self.hz}Hz (every {frame_interval} frame(s))...")
        self.cached_by_frame = []
        last_result = []
        frame_index = 0
        last_progress = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                last_result = self._analyze_frame(frame)
            self.cached_by_frame.append(last_result)
            frame_index += 1
            last_progress = update_progress("Emotion", frame_index, total_frames, last_progress)

        cap.release()
        print(f"Emotion analysis cache ready: {frame_index} frame(s) processed")

    def draw_from_cache(self, frame, frame_index):
        if not self.enabled or frame_index >= len(self.cached_by_frame):
            return frame

        faces = self.cached_by_frame[frame_index]
        if not faces:
            return frame

        h, w = frame.shape[:2]
        base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face["box"]]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            emotion = face["emotion"]
            conf = face["conf"]
            label = f"{emotion} {int(conf)}%"
            color = self.EMOTION_COLORS.get(emotion, self.DEFAULT_COLOR)

            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            tb = draw.textbbox((0, 0), label, font=self.font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            tx = max(0, min(w - tw - 1, x1))
            ty = max(0, y1 - th - 4)
            draw.rectangle([(tx - 1, ty - 1), (tx + tw + 1, ty + th + 1)], fill=(0, 0, 0, 160))
            draw.text((tx, ty), label, font=self.font, fill=color)

        composed = Image.alpha_composite(base, overlay)
        return cv2.cvtColor(np.array(composed), cv2.COLOR_RGBA2BGR)


def annotate_video_from_cache(input_video_path, output_video_path, annotator, captioner=None, emotion_analyzer=None):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for annotation: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for: {input_video_path}")

    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {output_video_path}")

    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last_progress = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated = annotator.draw_from_cache(frame, frame_index)
        if emotion_analyzer is not None:
            annotated = emotion_analyzer.draw_from_cache(annotated, frame_index)
        if captioner is not None:
            annotated = captioner.draw_caption(annotated, frame_index, fps)
        writer.write(annotated)
        frame_index += 1
        last_progress = update_progress("Annotate", frame_index, total_frames, last_progress)

    cap.release()
    writer.release()


def remux_audio(video_no_audio_path, audio_source_path, final_output_path):
    temp_output = os.path.join(TEMP_DIR, f"remuxed_{uuid.uuid4().hex}.mp4")
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_no_audio_path,
            "-i",
            audio_source_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            temp_output,
        ],
        quiet=True,
    )
    os.replace(temp_output, final_output_path)


def main():
    args = parse_args()
    input_file = args.input_file

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(OUTPUT_ROOT_DIR, input_basename)

    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    total_stages = 7
    log_stage(1, total_stages, "Reading input metadata")
    input_framerate = get_input_framerate(input_file)
    print(f"Input framerate: {input_framerate} fps")

    log_stage(2, total_stages, "Encoding base layer")
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-vf",
            "scale=iw/4:ih/4",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "28",
            "-f",
            "h264",
            BASE_LAYER_FILE,
        ],
        quiet=True,
    )

    log_stage(3, total_stages, "Encoding enhancement layer")
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "24",
            "-f",
            "h264",
            ENH_LAYER_FILE,
        ],
        quiet=True,
    )

    log_stage(4, total_stages, "Finding NAL units")
    with open(ENH_LAYER_FILE, "rb") as f:
        _enh_bytes = f.read()
    indices = find_nal_indices(_enh_bytes)
    del _enh_bytes  # free the large buffer; each iteration re-reads from disk

    annotator = YoloAnnotator(
        enabled=args.yolo,
        mode=args.yolo_mode,
        model_path=args.yolo_model,
        conf=args.yolo_conf,
    )
    if annotator.enabled:
        log_stage(5, total_stages, "Building one-time YOLO cache")
        annotator.build_cache(input_file)
    else:
        log_stage(5, total_stages, "Skipping YOLO cache (disabled)")

    captioner = WhisperCaptioner(enabled=args.whisper, model_name=args.whisper_model)
    if captioner.enabled:
        log_stage(6, total_stages, "Building Whisper transcription cache")
        captioner.build_cache(input_file)
    else:
        log_stage(6, total_stages, "Skipping Whisper (disabled)")

    emotion_analyzer = EmotionAnalyzer(enabled=args.emotion, hz=args.emotion_hz)
    if emotion_analyzer.enabled:
        log_stage(7, total_stages, f"Building emotion analysis cache ({args.emotion_hz}Hz)")
        emotion_analyzer.build_cache(input_file)
    else:
        log_stage(7, total_stages, "Skipping emotion analysis (disabled)")

    total_output_count = len(GLITCH_TYPES) * NUM_OUTPUTS
    completed_output_count = 0

    for glitch_type_index, (glitch_type, glitch_desc) in enumerate(GLITCH_TYPES.items(), 1):
        print(f"\n{'=' * 60}")
        print(f"GLITCH TYPE: {glitch_desc} ({glitch_type_index}/{len(GLITCH_TYPES)})")
        print(f"{'=' * 60}")

        for video_num, glitch_prob in enumerate(GLITCH_LEVELS, 1):
            current_output = completed_output_count + 1
            overall_pct = int((current_output / total_output_count) * 100)
            print(
                f"\n=== Generating {glitch_type} video {video_num}/{NUM_OUTPUTS} "
                f"(corruption: {glitch_prob * 100:.0f}%) | overall {overall_pct}% "
                f"({current_output}/{total_output_count}) ==="
            )

            with open(ENH_LAYER_FILE, "rb") as f:
                data = bytearray(f.read())
            corrupted_count = 0
            last_nal_progress = -1
            total_nals = len(indices)

            for i, (idx, start_code_len) in enumerate(indices):
                nal_byte_pos = idx + start_code_len
                if nal_byte_pos < len(data):
                    nal_type = data[nal_byte_pos] & 0x1F

                    if glitch_type in ["keyframe", "keyframe_destroy"]:
                        should_corrupt = nal_type == 5 and random.random() < glitch_prob
                    else:
                        should_corrupt = nal_type in [1, 5] and random.random() < glitch_prob

                    if should_corrupt:
                        nal_end = indices[i + 1][0] if i + 1 < len(indices) else len(data)
                        nal_data_start = nal_byte_pos + 1
                        if nal_data_start + 10 < nal_end:
                            corrupt_nal(data, nal_data_start, nal_end, glitch_type)
                            corrupted_count += 1

                last_nal_progress = update_progress("NAL scan", i + 1, total_nals, last_nal_progress)

            print(f"Corrupted {corrupted_count} NAL units")

            with open(GLITCHED_FILE, "wb") as f:
                f.write(data)

            output_file = os.path.join(
                output_dir,
                f"glitched_{glitch_type}_{video_num:02d}_{int(glitch_prob * 100):02d}pct.mp4",
            )
            print(f"Remuxing to {output_file}...")

            run_cmd(
                [
                    "ffmpeg",
                    "-y",
                    "-r",
                    input_framerate,
                    "-i",
                    GLITCHED_FILE,
                    "-i",
                    input_file,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a?",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "copy",
                    "-r",
                    input_framerate,
                    "-vsync",
                    "cfr",
                    "-fflags",
                    "+genpts",
                    output_file,
                ],
                quiet=True,
            )

            if annotator.enabled or captioner.enabled or emotion_analyzer.enabled:
                overlay_desc = ", ".join(
                    filter(
                        None,
                        [
                            f"YOLO ({annotator.mode})" if annotator.enabled else None,
                            "Whisper captions" if captioner.enabled else None,
                            f"emotion @ {emotion_analyzer.hz}Hz" if emotion_analyzer.enabled else None,
                        ],
                    )
                )
                print(f"Applying overlays [{overlay_desc}] to {output_file}...")
                annotated_no_audio = os.path.join(TEMP_DIR, f"annotated_{uuid.uuid4().hex}.mp4")
                try:
                    annotate_video_from_cache(
                        output_file,
                        annotated_no_audio,
                        annotator,
                        captioner=captioner if captioner.enabled else None,
                        emotion_analyzer=emotion_analyzer if emotion_analyzer.enabled else None,
                    )
                    remux_audio(annotated_no_audio, output_file, output_file)
                finally:
                    if os.path.exists(annotated_no_audio):
                        os.remove(annotated_no_audio)

            completed_output_count += 1
            done_pct = int((completed_output_count / total_output_count) * 100)
            print(f"[overall] {done_pct}% complete ({completed_output_count}/{total_output_count} outputs)")

    print(f"\n{'=' * 60}")
    print(
        f"Done! Generated {NUM_OUTPUTS * len(GLITCH_TYPES)} glitched videos "
        f"({len(GLITCH_TYPES)} types x {NUM_OUTPUTS} levels)"
    )
    print(f"Output directory: '{output_dir}/'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
