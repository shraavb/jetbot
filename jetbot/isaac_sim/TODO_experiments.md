# VLA Language Grounding Experiments - TODO

## Prerequisites
- [x] Isaac Sim installed and configured (on RunPod)
- [x] VLA server running (SmolVLA)
- [x] Python environment with dependencies

## Completed Experiments

### v1 Experiment (2026-01-05)
- [x] Trained SmolVLA on 10k samples
- [x] Evaluated on 4 instructions (go forward, turn left, turn right, red ball)
- [x] Results: 50% accuracy (2/4) - turn right broken
- [x] Documented in `experiments/results/vla_eval_20260105_v1.json`

### v2 Experiment (2026-01-06)
- [x] Collected new dataset: `dataset_turn_right_10k` (10k samples, ~23% right-turn instructions)
- [x] Trained SmolVLA v2: 5 epochs, batch-size 16, val_loss=0.0394
- [x] Evaluated on same 4 instructions
- [x] Results: 75% accuracy (3/4) - turn right STILL broken
- [x] Documented in `experiments/results/vla_eval_20260106_v2.json`

### Key Finding
- "turn right" produces R > L (left turn) in both v1 and v2
- Visual grounding works: "red ball" correctly produces L > R (right turn)
- **Next step**: Verify training data has correct motor values for "turn right" samples

## Scene Configurations Available

| Config | Description | Objects |
|--------|-------------|---------|
| `color_navigation` | Color-based navigation | Red/Blue/Green balls, Yellow box |
| `obstacle_course` | Avoidance testing | Chairs, Red box |
| `room_navigation` | Room with furniture | Door, Chair, Person, Red ball |
| `multi_target` | Multiple colored targets | Balls, boxes, cone, person, chair, door |
| `robot_avoidance` | JetBot collision avoidance | 3 JetBots, Red ball target |
| `multi_robot` | Comprehensive testing | 3 JetBots + colored targets + furniture |

## Commands to Run

### 1. Generate Mock Training Data (No Isaac Sim Required)
```bash
# Quick test - generates synthetic images with language labels
python -m jetbot.isaac_sim.collect_language_data \
    --mock \
    --episodes 100 \
    --output ./dataset_language_mock
```

### 2. Collect Training Data with Isaac Sim
```bash
# Basic data collection with multi-target scene
python -m jetbot.isaac_sim.collect_language_data \
    --config multi_target \
    --episodes 100 \
    --steps 50 \
    --output ./dataset_language \
    --headless

# Robot avoidance focused data
python -m jetbot.isaac_sim.collect_language_data \
    --config robot_avoidance \
    --episodes 100 \
    --steps 50 \
    --output ./dataset_robot_avoidance \
    --headless

# Multi-robot comprehensive data
python -m jetbot.isaac_sim.collect_language_data \
    --config multi_robot \
    --episodes 200 \
    --steps 50 \
    --output ./dataset_multi_robot \
    --headless
```

### 3. Run VLA Experiments (Requires VLA Server)

First, start VLA server:
```bash
python -m server.vla_server.server --model-type smolvla --port 5555
```

Then run experiments:
```bash
# Test all standard instruction categories
python -m jetbot.isaac_sim.vla_experiment \
    --config multi_robot \
    --standard-tests \
    --episodes 3 \
    --headless

# Test specific categories
python -m jetbot.isaac_sim.vla_experiment \
    --config robot_avoidance \
    --categories robot_avoidance color_target \
    --episodes 5 \
    --headless

# Test custom instructions
python -m jetbot.isaac_sim.vla_experiment \
    --config multi_robot \
    --instructions \
        "go towards the red ball" \
        "avoid colliding into the jetbot" \
        "swerve around the jetbot in front of you" \
        "follow the person in front of you" \
        "go around the chair" \
        "go towards the door" \
    --episodes 3 \
    --output-dir ./experiment_results \
    --headless
```

### 4. Fine-tune on Collected Data
```bash
# After collecting data, fine-tune SmolVLA
python -m server.vla_server.fine_tuning.train_smolvla \
    --data-dir ./dataset_language \
    --output-dir ./models/smolvla_language \
    --epochs 20 \
    --batch-size 32
```

## Language Commands Supported

### Color + Object Targets
- "go towards the red ball"
- "navigate to the blue box"
- "approach the yellow box"
- "drive to the orange cone"

### Object Targets
- "go towards the door"
- "approach the chair"
- "move towards the person"

### Person Following
- "follow the person in front of you"
- "follow the person"
- "go towards the person"

### Obstacle Avoidance
- "go around the chair"
- "avoid the obstacle"
- "navigate around the box"

### Robot/JetBot Avoidance
- "avoid colliding into the jetbot"
- "swerve around the jetbot in front of you"
- "go around the other robot"
- "avoid the jetbot ahead"
- "steer clear of the jetbot"
- "don't hit the other robot"

### Speed Modulation
- "speed up as you go towards the red ball"
- "slow down as you approach the door"
- "go fast towards the blue ball"
- "slowly approach the person"
- "quickly navigate to the red ball"
- "carefully approach the chair"
- "rush towards the green box"
- "cautiously move towards the obstacle"

### Conditional Actions
- "turn left near the person in front of you"
- "turn right when you reach the chair"
- "slow down as you go around the red ball"
- "stop when you reach the door"
- "turn around at the blue box"
- "go left after the chair"

### Complex Navigation
- "go quickly towards the red ball then slow down"
- "speed up as you go around the chair"
- "slowly approach the door while avoiding obstacles"
- "fast approach to the blue ball then stop"
- "carefully navigate around the obstacle then speed up"

### Directional
- "go forward"
- "turn left" / "turn right"
- "go backward"
- "stop"

## Output Formats

### Training Data
```
dataset_language/
├── {uuid}.jpg          # 224x224 RGB image
├── {uuid}.json         # Metadata
│   {
│     "instruction": "go towards the red ball",
│     "action": {"left_speed": 0.5, "right_speed": 0.5},
│     "command_type": "approach",
│     "target_type": "ball",
│     "target_color": "red",
│     ...
│   }
└── collection_stats.json
```

### Experiment Results
```
experiment_results/
└── {experiment_name}_{timestamp}.json
    {
      "experiment_name": "...",
      "num_trials": 18,
      "aggregate_metrics": {
        "success_rate": 0.72,
        "mean_final_distance": 0.35,
        ...
      },
      "trials": [...]
    }
```

## Evaluation Metrics
- **Success Rate**: % of trials reaching target within threshold
- **Progress Ratio**: Distance reduced / Initial distance
- **Path Efficiency**: Straight-line distance / Path length
- **Mean Final Distance**: Average distance to target at trial end
