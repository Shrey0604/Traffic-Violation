import os
import shutil

# ── Remapping tables ──────────────────────────────────────────────────────────
# d1: ['1-2-helmet'=0, '3-4-helmet'=1, 'Bald'=2, 'Cap'=3, 'Face and Hair'=4, 'Full-face-helmet'=5]
D1_REMAP = {0: 2, 1: 2, 2: 3, 3: 3, 4: 3, 5: 2}

# d2: ['bike'=0, 'motorcycle'=1]
D2_REMAP = {0: 0, 1: 0}

# d3: ['number plate'=0]
D3_REMAP = {0: 4}

DATASETS = [
    ("d1", D1_REMAP),
    ("d2", D2_REMAP),
    ("d3", D3_REMAP),
]

# ── Create merged folder structure ────────────────────────────────────────────
for split in ["train", "valid", "test"]:
    os.makedirs(f"merged/{split}/images", exist_ok=True)
    os.makedirs(f"merged/{split}/labels", exist_ok=True)

# ── Process each dataset ──────────────────────────────────────────────────────
def remap_and_copy_labels(src_label_dir, dst_label_dir, remap, prefix):
    if not os.path.exists(src_label_dir):
        print(f"  Skipping missing: {src_label_dir}")
        return
    for fname in os.listdir(src_label_dir):
        if not fname.endswith(".txt"):
            continue
        src_path = os.path.join(src_label_dir, fname)
        dst_path = os.path.join(dst_label_dir, prefix + fname)
        new_lines = []
        with open(src_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_id = int(parts[0])
                if old_id in remap:
                    parts[0] = str(remap[old_id])
                    new_lines.append(" ".join(parts) + "\n")
        if new_lines:
            with open(dst_path, "w") as f:
                f.writelines(new_lines)

def copy_images(src_img_dir, dst_img_dir, prefix):
    if not os.path.exists(src_img_dir):
        print(f"  Skipping missing: {src_img_dir}")
        return
    for fname in os.listdir(src_img_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy(
                os.path.join(src_img_dir, fname),
                os.path.join(dst_img_dir, prefix + fname)
            )

for ds_name, remap in DATASETS:
    print(f"Processing {ds_name}...")
    for split in ["train", "valid", "test"]:
        prefix = f"{ds_name}_"
        copy_images(
            f"{ds_name}/{split}/images",
            f"merged/{split}/images",
            prefix
        )
        remap_and_copy_labels(
            f"{ds_name}/{split}/labels",
            f"merged/{split}/labels",
            remap,
            prefix
        )
    print(f"  Done.")

print("\nMerge complete!")

# ── Count results ─────────────────────────────────────────────────────────────
for split in ["train", "valid", "test"]:
    n = len(os.listdir(f"merged/{split}/images"))
    print(f"  merged/{split}/images: {n} images")