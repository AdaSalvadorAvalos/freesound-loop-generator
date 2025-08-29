#!/bin/bash
################################################################################
# RAVE Preprocessing, Training, and Export Script
#
# This script automates the workflow for training a RAVE (Realtime Audio 
# Variational AutoEncoder) model on a custom audio dataset. It includes:
#   1. Preprocessing audio files for RAVE.
#   2. Training the model with a two-phase procedure:
#       - Phase 1 (VAE, 1M steps): Learns a compact latent representation.
#       - Phase 2 (Adversarial fine-tuning, 1M steps): Freezes the encoder
#         and trains the decoder adversarially to improve synthesis realism.
#   3. Exporting the trained model as a TorchScript file with streaming 
#      convolution support for real-time inference.
#
# User-configurable variables:
#   INPUT_PATH       - Path to raw WAV audio files.
#   PREPROCESS_PATH  - Directory for preprocessed audio data.
#   CHANNELS         - Number of audio channels.
#   CONFIG           - RAVE architecture configuration (default: v2).
#   RUN_NAME         - Name of the training run.
#   DB_PATH          - Database path (usually same as PREPROCESS_PATH).
#   OUT_PATH         - Directory to store training outputs.
#   TRAIN_CONFIGS    - Additional training configurations (e.g., noise, causal).
#   SAVE_EVERY       - Interval (in steps) to save model checkpoints.
#   MAX_STEPS        - Total training steps (default: 2,000,000).
#   CHECKPOINT_PATH  - Path to a specific checkpoint for export (optional).
#   EXPORT_PATH      - Path to save the final TorchScript model.
#   STREAMING        - Enable streaming mode for real-time inference (True/False).
#
# Usage:
#   1. Set the paths and configurations at the top of this script.
#   2. Make the script executable: chmod +x train.sh
#   3. Run the script: ./train.sh
################################################################################


# User-configurable paths
INPUT_PATH="path/to/waves"
PREPROCESS_PATH="path/to/preprocessed"
CHANNELS=1

CONFIG="v2"
RUN_NAME="foobar"
DB_PATH="$PREPROCESS_PATH"
OUT_PATH="runs/$RUN_NAME"
TRAIN_CONFIGS="noise causal"
SAVE_EVERY=500000
MAX_STEPS=2000000  # 2 million steps: first 1M Phase 1, next 1M Phase 2

CHECKPOINT_PATH="path/to/checkpoints/fname.ckpt"
EXPORT_PATH="path/to/save/model"
STREAMING="True"

# Step 1: Preprocess audio
echo "Preprocessing audio..."
rave preprocess --input_path "$INPUT_PATH" --output_path "$PREPROCESS_PATH" --channels $CHANNELS

# Step 2: Train the model (single command handles both phases)
echo "Training the model (2M steps: VAE + adversarial fine-tuning)..."
rave train \
    --config $CONFIG \
    --name $RUN_NAME \
    --db_path "$DB_PATH" \
    --out_path "$OUT_PATH" \
    $(for cfg in $TRAIN_CONFIGS; do echo --config $cfg; done) \
    --channels $CHANNELS \
    --save_every $SAVE_EVERY \
    --max_steps $MAX_STEPS

# Step 3: Export the model
echo "Exporting the model..."
rave export \
    --run "$CHECKPOINT_PATH" \
    --name $RUN_NAME \
    --output "$EXPORT_PATH" \
    --streaming $STREAMING

echo "All steps completed successfully!"
