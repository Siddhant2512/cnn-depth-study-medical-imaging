import pandas as pd
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
DATA_DIR = "/data/pcam/PatchCamelyon/data"
TRAIN_FILES = [
    f"{DATA_DIR}/train-00000-of-00003.parquet",
    f"{DATA_DIR}/train-00001-of-00003.parquet",
    f"{DATA_DIR}/train-00002-of-00003.parquet",
]
VAL_FILE = f"{DATA_DIR}/validation-00000-of-00001.parquet"
TEST_FILE = f"{DATA_DIR}/test-00000-of-00001.parquet"

def decode_pcam_image(image_dict):
    """Decode PNG bytes to numpy array"""
    b = image_dict["bytes"]
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return np.array(img, dtype=np.uint8)

# ========================================
# 1. Quick Dataset Statistics
# ========================================
print("="*60)
print("DATASET STATISTICS")
print("="*60)

for split, files in [("Train", TRAIN_FILES), ("Val", [VAL_FILE]), ("Test", [TEST_FILE])]:
    total_samples = 0
    total_positive = 0
    
    for file in files:
        df = pd.read_parquet(file)
        total_samples += len(df)
        total_positive += df['label'].sum()
    
    print(f"\n{split} Split:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Positive (tumor): {total_positive:,} ({100*total_positive/total_samples:.2f}%)")
    print(f"  Negative (normal): {total_samples-total_positive:,} ({100*(total_samples-total_positive)/total_samples:.2f}%)")

# ========================================
# 2. Visualize Sample Images
# ========================================
print("\n" + "="*60)
print("VISUALIZING SAMPLES")
print("="*60)

df_sample = pd.read_parquet(TRAIN_FILES[0])

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('PatchCamelyon Samples (Top: Normal, Bottom: Tumor)', fontsize=14, fontweight='bold')

# Get 5 normal and 5 tumor samples
normal_samples = df_sample[df_sample['label'] == 0].head(5)
tumor_samples = df_sample[df_sample['label'] == 1].head(5)

for i, (idx, row) in enumerate(normal_samples.iterrows()):
    img = decode_pcam_image(row['image'])
    axes[0, i].imshow(img)
    axes[0, i].set_title(f'Normal #{i+1}')
    axes[0, i].axis('off')

for i, (idx, row) in enumerate(tumor_samples.iterrows()):
    img = decode_pcam_image(row['image'])
    axes[1, i].imshow(img)
    axes[1, i].set_title(f'Tumor #{i+1}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('pcam_samples.png', dpi=150, bbox_inches='tight')
print("✓ Saved: pcam_samples.png")
plt.show()

# ========================================
# 3. Compute Normalization Statistics (CRITICAL!)
# ========================================
print("\n" + "="*60)
print("COMPUTING NORMALIZATION STATISTICS")
print("="*60)
print("This will take ~2-3 minutes...")

# Sample 10,000 images for statistics (enough for stable estimates)
np.random.seed(42)
sample_size = 10000

all_pixels = []

for file in TRAIN_FILES:
    df = pd.read_parquet(file)
    n_samples = min(sample_size // 3, len(df))
    indices = np.random.choice(len(df), n_samples, replace=False)
    
    for idx in tqdm(indices, desc=f"Processing {file.split('/')[-1]}"):
        img = decode_pcam_image(df.iloc[idx]['image'])
        all_pixels.append(img.reshape(-1, 3))  # Flatten spatial dims, keep channels

# Stack all pixels: shape (N_pixels, 3)
all_pixels = np.vstack(all_pixels).astype(np.float32) / 255.0  # Normalize to [0, 1]

# Compute mean and std per channel
mean_rgb = all_pixels.mean(axis=0)
std_rgb = all_pixels.std(axis=0)

print("\n" + "="*60)
print("NORMALIZATION STATISTICS (for transforms)")
print("="*60)
print(f"Mean (RGB): [{mean_rgb[0]:.4f}, {mean_rgb[1]:.4f}, {mean_rgb[2]:.4f}]")
print(f"Std  (RGB): [{std_rgb[0]:.4f}, {std_rgb[1]:.4f}, {std_rgb[2]:.4f}]")
print("\nUse these in your transforms:")
print(f"transforms.Normalize(mean={mean_rgb.tolist()}, std={std_rgb.tolist()})")

# Save to file for later use
stats = {
    'mean': mean_rgb.tolist(),
    'std': std_rgb.tolist()
}

import json
with open('pcam_normalization_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("\n✓ Saved: pcam_normalization_stats.json")

# ========================================
# 4. Pixel Intensity Distribution
# ========================================
print("\n" + "="*60)
print("ANALYZING PIXEL DISTRIBUTIONS")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = ['red', 'green', 'blue']
channel_names = ['Red', 'Green', 'Blue']

for i, (color, name) in enumerate(zip(colors, channel_names)):
    axes[i].hist(all_pixels[:, i], bins=50, color=color, alpha=0.7, edgecolor='black')
    axes[i].axvline(mean_rgb[i], color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_rgb[i]:.3f}')
    axes[i].set_xlabel('Normalized Pixel Value [0, 1]')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{name} Channel Distribution')
    axes[i].legend()
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pcam_pixel_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: pcam_pixel_distribution.png")
plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Check pcam_samples.png to understand the data visually")
print("2. Use the normalization stats in your training transforms")
print("3. Note the class balance for potential weighted loss functions")








