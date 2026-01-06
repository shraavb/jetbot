# VLA Experiments

This directory contains experiment configurations, results, and documentation for the JetBot Vision-Language-Action (VLA) model.

## Directory Structure

```
experiments/
├── README.md                    # This file
├── configs/                     # Experiment configurations
│   └── smolvla_jetbot_v1.yaml   # Current best model config
├── logs/                        # Training logs and tensorboard
├── results/                     # Evaluation results (JSON)
├── visualizations/              # Plots, trajectories, analysis
└── reports/                     # Experiment reports and summaries
```

## Current Model: SmolVLA-JetBot-v1

### Training Data Statistics

| Dataset | Samples | Source | Description |
|---------|---------|--------|-------------|
| dataset_vla | 10,000 | Isaac Sim | Primary training data with domain randomization |
| dataset_vla_synthetic | 1,000 | Isaac Sim | Initial synthetic data |
| dataset_vla_synthetic_large | 2,000 | Isaac Sim | Extended synthetic data |

**Instruction Distribution:**
- Forward motion: ~25%
- Left turns: ~18%
- Right turns: ~18%
- Obstacle avoidance: ~18%
- Object approach: ~12%
- Stop/wait: ~6%
- Other: ~3%

### Model Architecture

```
Base Model: HuggingFaceTB/SmolVLM-500M-Instruct
├── Vision Encoder: SigLIP-400M
├── Language Model: SmolLM-360M
└── Hidden Size: 960

Action Head (Fine-tuned):
├── Linear(960 → 128)
├── ReLU + Dropout(0.1)
├── Linear(128 → 2)
└── Tanh (output in [-1, 1])

Total Parameters: ~500M (frozen) + 123K (trainable action head)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-5 |
| Batch Size | 2 |
| Epochs | 20 |
| Warmup | 10% of steps |
| Scheduler | OneCycleLR |
| Loss | MSE |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Gradient Clipping | 1.0 |

### Training Results

- **Best Validation Loss**: See `logs/training_log.json`
- **Training Time**: ~X hours on RTX 4090
- **Checkpoint**: `models/smolvla_jetbot/best/`

### Evaluation Results

See `results/vla_eval_20260105_v1.json` for detailed evaluation data.

**Summary (50 steps per instruction):**

| Instruction | Avg Left | Avg Right | R-L Diff | Distance | Assessment |
|-------------|----------|-----------|----------|----------|------------|
| go forward | 0.51 | 0.54 | +0.03 | 0.141m | ✅ Correct |
| turn left | 0.39 | 0.72 | +0.33 | 0.141m | ✅ Correct |
| turn right | 0.34 | 0.68 | +0.34 | 0.129m | ❌ Incorrect |
| go towards red ball | 0.31 | 0.10 | -0.21 | 0.055m | 🔄 Partial |

**Accuracy**: 2/4 correct (50%)

**Key Observations:**
- "turn right" produces the **same motor pattern** as "turn left" (R > L)
- For correct right turn, we need L > R (negative R-L diff)
- Visual grounding shows some response but not clearly goal-directed

### Known Limitations

1. **Turn Right Confusion**: Model doesn't differentiate "turn right" from "turn left" well
2. **Visual Grounding**: Limited response to visual targets (red ball, etc.)
3. **Speed Control**: No differentiation between "slow" and "fast" commands
4. **Obstacle Avoidance**: Not yet tested extensively

### Failure Cases

1. "turn right" produces left-turning behavior (R > L instead of L > R)
2. Complex instructions like "go around the obstacle" not well understood
3. Model may output similar actions regardless of visual scene changes

### Sim-to-Real Transfer Notes

**Expected Gaps:**
- Lighting conditions differ significantly
- Real camera has different intrinsics/distortion
- Motor response curves differ from simulation
- Floor textures and obstacles look different

**Mitigation Strategies:**
1. Domain randomization during data collection
2. Real-world fine-tuning with small dataset
3. Action scaling/calibration on real robot
4. Visual domain adaptation (style transfer)

## Running Experiments

### Data Collection
```bash
python jetbot/isaac_sim/runpod_setup_simple.py \
    --collect-data \
    --episodes 200 \
    --steps 50 \
    --output /workspace/dataset_new
```

### Training
```bash
python -m server.vla_server.fine_tuning.train_smolvla \
    --data-dir dataset_vla \
    --output-dir models/smolvla_jetbot \
    --epochs 20 \
    --batch-size 2 \
    --lr 5e-5
```

### Evaluation
```bash
# Start VLA server
python -m server.vla_server.server --model-type smolvla --fine-tuned \
    --model /path/to/checkpoint

# Run evaluation
python jetbot/isaac_sim/vla_experiment_test.py \
    --steps 50 \
    --instructions "go forward" "turn left" "turn right" \
    --output results/eval_v1.json
```

## TODO

### High Priority
- [ ] **Verify training data labeling**: Check that "turn right" samples in `dataset_turn_right_10k` actually have L > R motor values (left motor > right motor). The v2 model still produces R > L for "turn right", suggesting the training data may be incorrectly labeled.

### Medium Priority
- [ ] Create data augmentation script to mirror "turn left" trajectories → "turn right" with swapped motor values
- [ ] Investigate why visual grounding works (red ball → right turn) but language doesn't ("turn right" → left turn)
- [ ] Add contrastive training: pair "turn left" and "turn right" examples in same batch

### Low Priority
- [ ] Test more instruction variations (e.g., "rotate right", "go right", "veer right")
- [ ] Add speed control instructions ("go forward slowly", "turn left quickly")
- [ ] Test obstacle avoidance scenarios

## Changelog

### v2 (2026-01-06)
- Trained on `dataset_turn_right_10k` (10k samples with ~23% right-turn instructions)
- Training: 5 epochs, batch size 16, best val_loss=0.0394
- **Results**: 75% accuracy (3/4 correct)
  - ✅ "go forward": IMPROVED (L=0.85, R=0.94)
  - ✅ "turn left": Similar (L=0.57, R=0.92)
  - ❌ "turn right": STILL BROKEN (L=0.64, R=0.95) - turns left instead
  - ✅ "go towards red ball": IMPROVED (L=0.86, R=0.45) - correct right turn
- **Key insight**: Visual grounding works but language differentiation doesn't

### v1 (2026-01-05)
- Initial SmolVLA fine-tuning on 10k samples
- Action head with 960→128→2 architecture
- Basic instruction following for forward/left
- Known issues with turn right differentiation
