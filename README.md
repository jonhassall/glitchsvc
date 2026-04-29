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

4. Find generated files in:

```text
glitched_outputs/your_video/
```

## Run again

Use any other video file in the same way:

```bash
docker compose run --rm glitchsvc another_video.mov
```

## Clean up containers

```bash
docker compose down
```

## Notes

- The compose service mounts this folder into the container, so outputs are written directly to your local project folder.
- Output is grouped by input filename (without extension), for example: `glitched_outputs/input/`.
- The script generates multiple outputs across several glitch types and corruption levels.
