import os
import subprocess
import random
import numpy as np

# === CONFIG ===
INPUT_FILE = "input.mp4"
OUTPUT_DIR = "glitched_outputs"
TEMP_DIR = "tmp_svc"
BASE_LAYER_FILE = os.path.join(TEMP_DIR, "base_layer.264")
ENH_LAYER_FILE = os.path.join(TEMP_DIR, "enh_layer.264")
GLITCHED_FILE = os.path.join(TEMP_DIR, "glitched.264")

NUM_OUTPUTS = 10  # Number of videos per glitch type
GLITCH_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80]  # Corruption probabilities

# Define 3 glitch types
GLITCH_TYPES = {
    "random": "Random byte corruption",
    "zero": "Byte zeroing (blocky artifacts)",
    "block": "Block corruption (large artifacts)",
    "constant": "Constant damage throughout NAL",
    "interval": "Damage at regular intervals",
    "keyframe": "Obliterate keyframes (I-frames)"
}

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === STEP 1: Encode base layer (low-res) ===
print("Encoding base layer (low-res)...")
subprocess.run([
    "ffmpeg", "-y", "-i", INPUT_FILE,
    "-vf", "scale=iw/4:ih/4",
    "-c:v", "libx264",
    "-preset", "slow",
    "-crf", "28",
    "-f", "h264",
    BASE_LAYER_FILE
])

# === STEP 2: Encode enhancement layer (full-res) ===
print("Encoding enhancement layer (full-res)...")
subprocess.run([
    "ffmpeg", "-y", "-i", INPUT_FILE,
    "-c:v", "libx264",
    "-preset", "slow",
    "-crf", "24",
    "-f", "h264",
    ENH_LAYER_FILE
])

# Read enhancement layer once
with open(ENH_LAYER_FILE, "rb") as f:
    original_data = bytearray(f.read())

# Find NAL start codes (0x000001 or 0x00000001)
print("Finding NAL units...")
indices = []
i = 0
while i < len(original_data)-4:
    if original_data[i:i+3] == b'\x00\x00\x01':
        indices.append((i, 3))  # 3-byte start code
    elif original_data[i:i+4] == b'\x00\x00\x00\x01':
        indices.append((i, 4))  # 4-byte start code
    i += 1

# Function to apply corruption based on type
def corrupt_nal(data, nal_data_start, nal_end, glitch_type):
    """Apply different types of corruption to NAL unit data"""
    if glitch_type == "random":
        # Random byte corruption
        corrupt_pos = random.randint(nal_data_start + 10, nal_end - 1)
        data[corrupt_pos] = random.randint(0, 255)
    
    elif glitch_type == "zero":
        # Zero out bytes (creates blocky artifacts)
        corrupt_pos = random.randint(nal_data_start + 10, nal_end - 1)
        data[corrupt_pos] = 0
    
    elif glitch_type == "block":
        # Corrupt a block of consecutive bytes (larger artifacts)
        block_size = random.randint(5, 20)
        start_pos = random.randint(nal_data_start + 10, max(nal_data_start + 11, nal_end - block_size))
        for offset in range(block_size):
            if start_pos + offset < nal_end:
                data[start_pos + offset] = random.randint(0, 255)
    
    elif glitch_type == "constant":
        # Constant damage throughout the NAL unit
        nal_length = nal_end - nal_data_start - 10
        num_corruptions = max(5, nal_length // 20)  # Corrupt ~5% of bytes
        for _ in range(num_corruptions):
            corrupt_pos = random.randint(nal_data_start + 10, nal_end - 1)
            data[corrupt_pos] = random.randint(0, 255)
    
    elif glitch_type == "interval":
        # Damage at regular intervals throughout NAL
        interval = random.randint(15, 40)  # Corrupt every N bytes
        start_pos = nal_data_start + 10
        pos = start_pos
        while pos < nal_end - 1:
            data[pos] = random.randint(0, 255)
            pos += interval
    
    elif glitch_type == "keyframe":
        # Obliterate keyframe data - heavy corruption
        nal_length = nal_end - nal_data_start - 10
        num_corruptions = max(20, nal_length // 5)  # Corrupt ~20% of bytes
        for _ in range(num_corruptions):
            corrupt_pos = random.randint(nal_data_start + 10, nal_end - 1)
            data[corrupt_pos] = random.randint(0, 255)

# Generate multiple videos with varying corruption levels
for glitch_type, glitch_desc in GLITCH_TYPES.items():
    print(f"\n{'='*60}")
    print(f"GLITCH TYPE: {glitch_desc}")
    print(f"{'='*60}")
    
    for video_num, glitch_prob in enumerate(GLITCH_LEVELS, 1):
        print(f"\n=== Generating {glitch_type} video {video_num}/{NUM_OUTPUTS} (corruption: {glitch_prob*100:.0f}%) ===")
        
        # Make a fresh copy of the data
        data = bytearray(original_data)
        
        # === STEP 3: Corrupt enhancement layer NALs ===
        print(f"Corrupting enhancement layer with {glitch_prob*100:.0f}% probability...")
        corrupted_count = 0
        
        # Corrupt enhancement NALs - only corrupt slice data, not headers
        for i, (idx, start_code_len) in enumerate(indices):
            # Get NAL unit type (lower 5 bits of first byte after start code)
            nal_byte_pos = idx + start_code_len
            if nal_byte_pos < len(data):
                nal_type = data[nal_byte_pos] & 0x1F
                
                # Determine which NAL types to target based on glitch type
                if glitch_type == "keyframe":
                    # Target only keyframes (IDR frames, type 5)
                    should_corrupt = nal_type == 5 and random.random() < glitch_prob
                else:
                    # Target regular slices (type 1) and keyframes (type 5), skip SPS(7), PPS(8), SEI(6)
                    should_corrupt = nal_type in [1, 5] and random.random() < glitch_prob
                
                if should_corrupt:
                    nal_end = indices[i+1][0] if i+1 < len(indices) else len(data)
                    nal_data_start = nal_byte_pos + 1
                    
                    # Skip NAL units that are too small to corrupt safely
                    if nal_data_start + 10 < nal_end:
                        # Apply corruption based on glitch type
                        corrupt_nal(data, nal_data_start, nal_end, glitch_type)
                        corrupted_count += 1
        
        print(f"Corrupted {corrupted_count} NAL units")
        
        with open(GLITCHED_FILE, "wb") as f:
            f.write(data)
        
        # === STEP 4: Remux glitched stream into MP4 ===
        output_file = os.path.join(OUTPUT_DIR, f"glitched_{glitch_type}_{video_num:02d}_{int(glitch_prob*100):02d}pct.mp4")
        print(f"Remuxing to {output_file}...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", GLITCHED_FILE,
            "-i", INPUT_FILE,
            "-map", "0:v:0",  # video from glitched file
            "-map", "1:a?",   # audio from original (if exists)
            "-c:v", "copy",
            "-c:a", "copy",
            "-fflags", "+genpts",  # generate presentation timestamps
            output_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"\n{'='*60}")
print(f"✓ Done! Generated {NUM_OUTPUTS * len(GLITCH_TYPES)} glitched videos ({len(GLITCH_TYPES)} types × {NUM_OUTPUTS} levels)")
print(f"Output directory: '{OUTPUT_DIR}/'")
print(f"{'='*60}")
