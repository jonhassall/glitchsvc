# Glitch SVC Video Generator

Creates glitched video art by intentionally corrupting H.264 video streams.

## What it does

1. Encodes video into base and enhancement layers
2. Corrupts the enhancement layer NAL units to create glitch effects
3. Generates 98 videos with 7 damage types at 14 corruption levels each (5%-99%)

## Glitch Types

- **Random**: Random byte corruption (chaotic artifacts)
- **Zero**: Byte zeroing (blocky/black artifacts)
- **Block**: Block corruption (large consecutive damage)
- **Constant**: Constant damage throughout NAL units (persistent artifacts)
- **Interval**: Damage at regular intervals (rhythmic glitches)
- **Keyframe**: Obliterate keyframes/I-frames (severe reference corruption)
- **Keyframe_destroy**: Complete keyframe annihilation (~70% data destruction)

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
- `glitched_constant_07_40pct.mp4` (40% corruption)
- `glitched_interval_03_15pct.mp4` (15% corruption)
- `glitched_keyframe_09_65pct.mp4` (65% corruption)
- `glitched_keyframe_destroy_11_90pct.mp4` (90% corruption)
- `glitched_random_13_98pct.mp4` (98% corruption)
- `glitched_zero_14_99pct.mp4` (99% corruption)

## Notes

- The script preserves audio from the original video
- Critical headers (SPS/PPS) are not corrupted to prevent playback issues
- Only slice NAL units are corrupted for controlled glitch effects
