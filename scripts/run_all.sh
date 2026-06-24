#!/bin/bash
# Run the complete pipeline

set -e

echo "=== RecSys-HR 2026 WorkRB Challenge ==="
echo "=== Complete Pipeline Runner ==="

# Step 1: Clone challenge repo if not present
if [ ! -d "workrb-recsys-hr-challenge-2026" ]; then
    echo "Cloning challenge repository..."
    git clone https://github.com/techwolf-ai/workrb-recsys-hr-challenge-2026.git
fi

cd workrb-recsys-hr-challenge-2026

# Step 2: Install dependencies
echo "Installing dependencies..."
uv sync

# Step 3: Copy our model files
echo "Setting up models..."
cp ../src/model_biencoder.py participant/model_biencoder.py
cp ../src/model_cross_encoder.py participant/model_cross_encoder.py

# Step 4: Train bi-encoder
echo "Training bi-encoder..."
uv run python participant/train.py \
    --set model.init.leaderboard_name="ImprovedBiEncoder-v1" \
    --set model.init.model_name="sentence-transformers/all-mpnet-base-v2" \
    --set model.init.max_length=192 \
    --set model.init.temperature=0.03 \
    --set data.batch_size=128 \
    --set optim.learning_rate=2e-5 \
    --set epochs=3

# Step 5: Generate submission
echo "Generating submission..."
uv run python ../src/generate_submission.py

echo "=== Done! ==="
echo "Upload submission/submission.zip to CodaBench"
