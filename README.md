# Glitch SVC with Docker Compose

Generate glitched video variants by corrupting H.264 NAL units.

## Quick start

1. Put your source video in the project folder.
2. Build the container once:

```bash
docker compose build
```

3. Run and specify the video to process:

```bash
docker compose run --rm glitchsvc your_video.mp4
```

YOLO overlays are optional and off by default.

4. Optional YOLO segmentation overlays (default YOLO mode):

```bash
docker compose run --rm glitchsvc your_video.mp4 --yolo
```

5. Optional YOLO rectangle overlays instead of segmentation borders:

```bash
docker compose run --rm glitchsvc your_video.mp4 --yolo --yolo-mode box
```

6. Find generated files in:

```text
glitched_outputs/your_video/
```

## Run again

Use any other video file in the same way:

```bash
docker compose run --rm glitchsvc another_video.mov
```

Use a different YOLO model or confidence threshold:

```bash
docker compose run --rm glitchsvc your_video.mp4 --yolo --yolo-model yolo11s-seg.pt --yolo-conf 0.35
```

7. Optional emotion analysis overlay on detected faces:

```bash
docker compose run --rm glitchsvc your_video.mp4 --emotion
```

8. Adjust emotion analysis frequency (default is 2 Hz):

```bash
docker compose run --rm glitchsvc your_video.mp4 --emotion --emotion-hz 1.0
```

## Everything on

Run with all overlays and analysis enabled at once:

```bash
docker compose run --rm glitchsvc your_video.mp4 --yolo --yolo-mode segmentation --yolo-model yolo11n-seg.pt --yolo-conf 0.25 --whisper --whisper-model base --emotion --emotion-hz 2.0
```

## Clean up containers

```bash
docker compose down
```

## Notes

- The compose service mounts this folder into the container, so outputs are written directly to your local project folder.
- Output is grouped by input filename (without extension), for example: `glitched_outputs/input/`.
- The script generates multiple outputs across several glitch types and corruption levels.
- YOLO uses GPU automatically when available and falls back to CPU otherwise.
- YOLO analysis is performed once on the input video, then reused for every generated output.
- `--yolo-mode` defaults to `segmentation`; use `--yolo-mode box` for rectangle-only overlays.
- Emotion analysis is performed once on the input video and reused for every generated output.
- `--emotion-hz` defaults to `2.0`; lower values process fewer frames per second and run faster.
