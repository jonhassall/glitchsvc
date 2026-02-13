# Glitch SVC Video Generator

Creates glitched video art by intentionally corrupting H.264 video streams.

## What it does

1. Encodes video into base and enhancement layers
2. Corrupts the enhancement layer NAL units to create glitch effects
3. Generates 30 videos with 3 damage types at 10 corruption levels each

## Glitch Types

- **Random**: Random byte corruption (chaotic artifacts)
- **Zero**: Byte zeroing (blocky/black artifacts)
- **Block**: Block corruption (large consecutive damage)

## Requirements

- Python 3.x
- FFmpeg (must be in PATH)
- NumPy

```bash
pip install -r requirements.txt
```

## Usage

Place your input video as `input.mp4` in the project directory, then run:

```bash
python glitch_svc.py
```

Output videos are saved to `glitched_outputs/` with filenames like:
- `glitched_random_01_05pct.mp4` (5% corruption)
- `glitched_zero_05_25pct.mp4` (25% corruption)
- `glitched_block_10_80pct.mp4` (80% corruption)

## Notes

- The script preserves audio from the original video
- Critical headers (SPS/PPS) are not corrupted to prevent playback issues
- Only slice NAL units are corrupted for controlled glitch effects
