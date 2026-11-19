"""
EcoSort AI - Dataset Preprocessing Pipeline
Reads parquet files from Dataset/, verifies splits, checks corrupted/duplicate images,
resizes to 224x224 RGB, saves organized directory structure, and calculates class statistics.
"""

import os
import io
import json
import hashlib
import pyarrow.parquet as pq
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

DATASET_DIR = "Dataset"
PROCESSED_DIR = "processed_dataset"
TARGET_SIZE = (224, 224)

# Canonical 6 classes
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

SPLIT_FILES = {
    "train": [
        os.path.join(DATASET_DIR, "train-00000-of-00003.parquet"),
        os.path.join(DATASET_DIR, "train-00001-of-00003.parquet"),
        os.path.join(DATASET_DIR, "train-00002-of-00003.parquet"),
    ],
    "val": [
        os.path.join(DATASET_DIR, "validation-00000-of-00001.parquet"),
    ],
    "test": [
        os.path.join(DATASET_DIR, "test-00000-of-00001.parquet"),
    ],
}


def preprocess_dataset(target_size=TARGET_SIZE, save_images=True):
    print("=" * 60)
    print("EcoSort AI - Dataset Preprocessing Pipeline Starting")
    print("=" * 60)

    # Prepare directories
    for split in ["train", "val", "test"]:
        for cls_name in CLASSES:
            split_cls_dir = os.path.join(PROCESSED_DIR, split, cls_name)
            os.makedirs(split_cls_dir, exist_ok=True)

    seen_hashes = set()
    duplicate_count = 0
    corrupted_count = 0
    split_stats = defaultdict(lambda: defaultdict(int))
    total_processed = 0

    summary = {
        "target_size": list(target_size),
        "classes": CLASSES,
        "label_mapping": LABEL_MAP,
        "splits": {},
        "duplicates_detected": 0,
        "corrupted_detected": 0,
        "total_extracted": 0,
    }

    for split, file_list in SPLIT_FILES.items():
        print(f"\n[Processing Split: {split.upper()}] - {len(file_list)} Parquet file(s)...")
        split_img_idx = 0

        for file_path in file_list:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Parquet file not found: {file_path}")

            print(f"  Reading {os.path.basename(file_path)}...")
            parquet_file = pq.ParquetFile(file_path)
            num_row_groups = parquet_file.num_row_groups

            for rg_idx in range(num_row_groups):
                table = parquet_file.read_row_group(rg_idx, columns=["image", "label_name"])
                pydict = table.to_pydict()
                images_col = pydict["image"]
                labels_col = pydict["label_name"]

                for i in range(len(labels_col)):
                    total_processed += 1
                    raw_label = labels_col[i]
                    if raw_label is None:
                        continue
                    label_name = str(raw_label).strip().lower()

                    if label_name not in LABEL_MAP:
                        print(f"Warning: Unexpected label '{label_name}' encountered! Skipping.")
                        continue

                    img_bytes = images_col[i]["bytes"]
                    if not img_bytes:
                        corrupted_count += 1
                        continue

                    # Check for duplicates using SHA256 hash
                    img_hash = hashlib.sha256(img_bytes).hexdigest()
                    is_duplicate = img_hash in seen_hashes
                    if is_duplicate:
                        duplicate_count += 1
                    else:
                        seen_hashes.add(img_hash)

                    # Verify and convert image
                    try:
                        im = Image.open(io.BytesIO(img_bytes))
                        im.verify()
                        # Re-open after verify() as per PIL requirements
                        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    except Exception:
                        corrupted_count += 1
                        continue

                    # Resize to target size
                    im_resized = im.resize(target_size, Image.Resampling.BILINEAR)

                    # Save to processed folder
                    if save_images:
                        filename = f"{split}_{label_name}_{split_img_idx:06d}.jpg"
                        dest_path = os.path.join(PROCESSED_DIR, split, label_name, filename)
                        im_resized.save(dest_path, "JPEG", quality=88)

                    split_img_idx += 1
                    split_stats[split][label_name] += 1

                print(f"    Completed Row Group {rg_idx + 1}/{num_row_groups} (extracted {split_img_idx} so far)")

        summary["splits"][split] = dict(split_stats[split])
        print(f"Finished {split}: {split_img_idx} images extracted across {len(split_stats[split])} classes.")

    summary["duplicates_detected"] = duplicate_count
    summary["corrupted_detected"] = corrupted_count
    summary["total_extracted"] = sum(sum(v.values()) for v in split_stats.values())

    # Calculate class weights for training to handle imbalance
    train_counts = split_stats["train"]
    total_train = sum(train_counts.values())
    num_classes = len(CLASSES)
    class_weights = {}
    for cls_name in CLASSES:
        count = train_counts.get(cls_name, 1)
        # Standard balanced weight: total_samples / (num_classes * class_count)
        weight = total_train / (num_classes * count)
        class_weights[cls_name] = round(weight, 5)

    summary["class_weights"] = class_weights

    # Write summary JSON
    summary_path = os.path.join(PROCESSED_DIR, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nDataset summary written to: {summary_path}")

    # Write class weights JSON
    weights_path = os.path.join(PROCESSED_DIR, "class_weights.json")
    with open(weights_path, "w") as f:
        json.dump(class_weights, f, indent=4)
    print(f"Class weights written to: {weights_path}")

    # Write markdown report
    report_path = os.path.join(PROCESSED_DIR, "dataset_report.md")
    generate_markdown_report(summary, report_path)
    print(f"Dataset report written to: {report_path}")

    print("=" * 60)
    print("Preprocessing successfully completed!")
    print(f"Total Images Processed: {total_processed}")
    print(f"Total Successfully Extracted: {summary['total_extracted']}")
    print(f"Corrupted Images: {corrupted_count}")
    print(f"Duplicate Images Flagged: {duplicate_count}")
    print("=" * 60)
    return summary


def generate_markdown_report(summary, report_path):
    train_total = sum(summary["splits"].get("train", {}).values())
    val_total = sum(summary["splits"].get("val", {}).values())
    test_total = sum(summary["splits"].get("test", {}).values())
    total_imgs = summary["total_extracted"]

    lines = [
        "# EcoSort AI - Dataset Preprocessing & Distribution Report",
        "",
        "## 1. Overview",
        f"- **Dataset Source:** HuggingFace `TrashBusters/combined`",
        f"- **Total Images Extracted:** {total_imgs:,}",
        f"- **Target Resolution:** {summary['target_size'][0]} x {summary['target_size'][1]} RGB",
        f"- **Corrupted Images Detected:** {summary['corrupted_detected']}",
        f"- **Duplicate Images Flagged:** {summary['duplicates_detected']}",
        "",
        "## 2. Dataset Split Breakdown",
        f"- **Training Split:** {train_total:,} images ({train_total/total_imgs*100:.1f}%)",
        f"- **Validation Split:** {val_total:,} images ({val_total/total_imgs*100:.1f}%)",
        f"- **Test Split:** {test_total:,} images ({test_total/total_imgs*100:.1f}%)",
        "",
        "## 3. Class Distribution by Split",
        "",
        "| Class ID | Waste Category | Training | Validation | Test | Total | Class Weight |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for idx, cls_name in enumerate(CLASSES):
        tr = summary["splits"].get("train", {}).get(cls_name, 0)
        va = summary["splits"].get("val", {}).get(cls_name, 0)
        te = summary["splits"].get("test", {}).get(cls_name, 0)
        tot = tr + va + te
        weight = summary.get("class_weights", {}).get(cls_name, 1.0)
        lines.append(f"| {idx} | {cls_name.capitalize()} | {tr:,} | {va:,} | {te:,} | {tot:,} | {weight:.4f} |")

    lines.extend([
        "",
        "## 4. Class Imbalance Mitigation",
        "Class weights are calculated via inverse class frequencies: `w_c = N / (K * N_c)`.",
        "These weights are integrated directly into PyTorch's `CrossEntropyLoss(weight=class_weights_tensor)` during training.",
        "",
        "## 5. Waste Segregation Mapping",
        "- **Cardboard**: Blue Dustbin (Dry Waste)",
        "- **Glass**: Blue Dustbin (Dry Waste)",
        "- **Metal**: Blue Dustbin (Dry Waste)",
        "- **Paper**: Blue Dustbin (Dry Waste)",
        "- **Plastic**: Blue Dustbin (Dry Waste)",
        "- **Trash**: User Verification Required (Organic -> Green Dustbin, Dry Residual -> Blue/Residual, Hazardous -> Specialized route)",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    preprocess_dataset()
