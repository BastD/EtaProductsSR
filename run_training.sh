#!/bin/bash

cat run_training.sh

# Training script with SUBSAMPLING for fast iteration
# Use this for debugging and rapid experiments

set -e

DATA_PATH="./data_P_1_4_SUM_2_2_EXP_1_6_MAXK_12_RELCOEF_20_PRO_MOD_SW_2M/eta_product_dataset.pkl"
OUTPUT_DIR="./checkpoints_P_1_4_SUM_2_2_EXP_1_6_MAXK_12_RELCOEF_20_PRO_MOD_SW_2M"

# Model hyperparameters
D_MODEL=256
NHEAD=8
NUM_ENCODER_LAYERS=4
NUM_DECODER_LAYERS=4
DIM_FEEDFORWARD=1024
DROPOUT=0.2
MAX_LEN=100

# Training hyperparameters
EPOCHS=1000
BATCH_SIZE=64
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
GRAD_CLIP=5.0

# System
SEED=42

echo "================================================"
echo "TRAINING" 
echo "================================================"

python train_tf_sr.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --num_encoder_layers $NUM_ENCODER_LAYERS \
    --num_decoder_layers $NUM_DECODER_LAYERS \
    --dim_feedforward $DIM_FEEDFORWARD \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --grad_clip $GRAD_CLIP \
    --label_smoothing 0.0 \
    --early_stopping 1000 \
    --seed $SEED \
    --max_formula_len $MAX_LEN \
    --use_periodic


