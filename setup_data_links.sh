#!/bin/bash
# Setup symlinks để VTN3GCN code đọc đúng MM-WLAuslan structure.
# Chạy 1 lần TRƯỚC khi extract pose hoặc train.
set -e

BASE="/mnt/sda1/VSLR_Storage/MM-WLAuslan"

# 1. Symlink CSV labels → base_url/labels/
echo "=== Setup label symlinks ==="
mkdir -p "$BASE/labels"
ln -sf "/mnt/sda1/Duong/back-up-duy/MM-WLAuslan/Label/labelThreeViewSTU/train_labels.csv" "$BASE/labels/train_labels.csv"
ln -sf "/mnt/sda1/Duong/back-up-duy/MM-WLAuslan/Label/labelThreeViewSTU/val_labels.csv"   "$BASE/labels/val_labels.csv"
ln -sf "/mnt/sda1/Duong/back-up-duy/MM-WLAuslan/Label/labelThreeViewSTU/test_labels.csv"  "$BASE/labels/test_labels.csv"

# 2. Symlink TẤT CẢ video từ cropped/, crop_valid/, testSTU/ vào videos/ flat
# Code đọc {base_url}/videos/{filename} - cần all videos ở 1 chỗ
echo "=== Setup video symlinks (may take 1-2 min) ==="
mkdir -p "$BASE/videos_unified"
for src in cropped crop_valid testSTU testITW testSYN testTED; do
    if [ -d "$BASE/videos/$src" ]; then
        echo "Linking $src..."
        for f in "$BASE/videos/$src"/*.mp4; do
            ln -sf "$f" "$BASE/videos_unified/$(basename $f)"
        done
    fi
done

# 3. Rename videos -> videos_raw, videos_unified -> videos
if [ -d "$BASE/videos" ] && [ ! -L "$BASE/videos" ]; then
    mv "$BASE/videos" "$BASE/videos_raw"
fi
mv "$BASE/videos_unified" "$BASE/videos"

echo "=== DONE ==="
echo "Verify:"
ls -la "$BASE/labels/"
echo "Total videos in unified folder:"
ls "$BASE/videos/" | wc -l
echo ""
echo "First train sample exists:"
first=$(awk -F',' 'NR==2 {print $1}' "$BASE/labels/train_labels.csv")
ls -la "$BASE/videos/$first"
