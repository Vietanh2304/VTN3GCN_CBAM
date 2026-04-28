#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash run_extract_mmwlauslan.sh <split>"
    echo "split: train, val, test_stu, test_itw, test_syn, test_ted"
    exit 1
fi

SPLIT=$1

# Map splits to corresponding paths
case $SPLIT in
    train)
        CSV_PATH="/mnt/sda1/Duong/back-up-duy/MM-WLAuslan/Label/labelThreeViewSTU/train_labels.csv"
        VIDEO_DIR="/mnt/sda1/VSLR_Storage/MM-WLAuslan/videos/cropped/"
        ;;
    val)
        CSV_PATH="/mnt/sda1/Duong/back-up-duy/MM-WLAuslan/Label/labelThreeViewSTU/val_labels.csv"
        VIDEO_DIR="/mnt/sda1/VSLR_Storage/MM-WLAuslan/videos/crop_valid/"
        ;;
    test_stu)
        CSV_PATH="/mnt/sda1/Duong/back-up-duy/MM-WLAuslan/Label/labelThreeViewSTU/test_labels.csv"
        VIDEO_DIR="/mnt/sda1/VSLR_Storage/MM-WLAuslan/videos/testSTU/"
        ;;
    test_itw)
        CSV_PATH="/mnt/sda1/Duong/back-up-duy/MM-WLAuslan/Label/labelThreeViewITW/test_labels.csv"
        VIDEO_DIR="/mnt/sda1/VSLR_Storage/MM-WLAuslan/videos/testITW/"
        ;;
    test_syn)
        CSV_PATH="/mnt/sda1/Duong/back-up-duy/MM-WLAuslan/Label/labelThreeViewSYN/test_labels.csv"
        VIDEO_DIR="/mnt/sda1/VSLR_Storage/MM-WLAuslan/videos/testSYN/"
        ;;
    test_ted)
        CSV_PATH="/mnt/sda1/Duong/back-up-duy/MM-WLAuslan/Label/labelThreeViewTED/test_labels.csv"
        VIDEO_DIR="/mnt/sda1/VSLR_Storage/MM-WLAuslan/videos/testTED/"
        ;;
    *)
        echo "Invalid split. Use: train, val, test_stu, test_itw, test_syn, test_ted"
        exit 1
        ;;
esac

echo "Running extraction for SPLIT=$SPLIT"
echo "CSV_PATH=$CSV_PATH"
echo "VIDEO_DIR=$VIDEO_DIR"

OUTPUT_POSES="/mnt/sda1/VSLR_Storage/MM-WLAuslan/poses"
OUTPUT_WHOLEBODY="/mnt/sda1/VSLR_Storage/MM-WLAuslan/wholebody"
OUTPUT_POSEFLOW="/mnt/sda1/VSLR_Storage/MM-WLAuslan/poseflow"

# STEP 1: gen_pose_mmwlauslan.py (2 GPUs)
echo "=== STEP 1: Extracting Poses (Halpe-26) ==="
CUDA_VISIBLE_DEVICES=0 python tools/gen_pose_mmwlauslan.py \
    --csv_path "$CSV_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_POSES" \
    --shard 0 --num_shards 2 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python tools/gen_pose_mmwlauslan.py \
    --csv_path "$CSV_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_POSES" \
    --shard 1 --num_shards 2 &
PID2=$!

wait $PID1
wait $PID2

# STEP 2: gen_wholebody_mmwlauslan.py (2 GPUs)
echo "=== STEP 2: Extracting Wholebody ==="
CUDA_VISIBLE_DEVICES=0 python tools/gen_wholebody_mmwlauslan.py \
    --csv_path "$CSV_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_WHOLEBODY" \
    --shard 0 --num_shards 2 &
PID3=$!

CUDA_VISIBLE_DEVICES=1 python tools/gen_wholebody_mmwlauslan.py \
    --csv_path "$CSV_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_WHOLEBODY" \
    --shard 1 --num_shards 2 &
PID4=$!

wait $PID3
wait $PID4

# STEP 3: extract_poseflow_mmwlauslan.py (CPU)
echo "=== STEP 3: Extracting Poseflow ==="
python tools/extract_poseflow_mmwlauslan.py \
    --csv_path "$CSV_PATH" \
    --poses_dir "$OUTPUT_POSES" \
    --wholebody_dir "$OUTPUT_WHOLEBODY" \
    --output_dir "$OUTPUT_POSEFLOW" \
    --shard 0 --num_shards 1

# STEP 4: extract_hand_kp.py (CPU only, fast)
echo "=== STEP 4: Extracting Hand Keypoints ==="
python tools/extract_hand_kp.py

echo "=== PIPELINE COMPLETED SUCCESSFULLY FOR $SPLIT ==="
