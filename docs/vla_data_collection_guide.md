# VLA Data Collection Guide for JetBot

## Overview

Good VLA training data captures the relationship between:
1. **What the robot sees** (camera image)
2. **What it should do** (natural language instruction)
3. **How it should move** (motor commands)

## Key Principles

### 1. Continuous Actions, Not Binary Labels

**Bad:** "This image is blocked" or "This image is free"
**Good:** "When I see this and want to 'go forward', I should set motors to L=0.4, R=0.4"

The motor values should reflect the **degree** of the action:
- Gentle turn: L=0.3, R=0.4
- Sharp turn: L=-0.2, R=0.5
- Forward: L=0.4, R=0.4
- Stop: L=0.0, R=0.0

### 2. Diverse Instructions

Don't just use one instruction. Vary your language:

**Navigation instructions:**
- "go forward"
- "move ahead avoiding obstacles"
- "navigate to the open space"
- "go straight"

**Turning instructions:**
- "turn left"
- "go around the obstacle on the left"
- "rotate left 90 degrees"

**Object-directed instructions:**
- "go to the red ball"
- "approach the door"
- "move toward the chair"
- "find the toy"

**Avoidance instructions:**
- "avoid the box"
- "go around the obstacle"
- "don't hit the wall"

### 3. Scenario Coverage

Collect data for these scenarios:

| Scenario | Motor Pattern | Sample Count |
|----------|--------------|--------------|
| Clear path forward | L≈R, both positive | 150-200 |
| Obstacle on left | L > R (turn right) | 100-150 |
| Obstacle on right | L < R (turn left) | 100-150 |
| Obstacle ahead | Sharp turn or reverse | 100-150 |
| Near target object | Slow approach | 50-100 |
| At target (stop) | L=0, R=0 | 50-100 |
| Tight spaces | Slow, careful movements | 50-100 |

**Total: 600-850 samples minimum**

### 4. Position and Angle Variation

For each scenario, vary:
- **Distance** to obstacles/targets (near, medium, far)
- **Angle** of approach (head-on, 45°, 90°)
- **Robot orientation** (facing left, right, straight)

## Data Collection Process

### Step 1: Set Up Environment

Create scenarios with:
- Cardboard boxes (obstacles)
- Colored objects (targets)
- Walls/boundaries
- Open spaces

### Step 2: Collect by Instruction Type

**Session 1: Forward Navigation (200 samples)**
```
Instruction: "go forward" / "move ahead" / "navigate forward"
- Place robot in open areas
- Drive forward at various speeds
- Save samples every 0.5-1 second while moving
```

**Session 2: Left Turns (150 samples)**
```
Instruction: "turn left" / "go left" / "avoid obstacle on right"
- Place obstacles on robot's right side
- Execute left turns of varying sharpness
- Save during the turn
```

**Session 3: Right Turns (150 samples)**
```
Instruction: "turn right" / "go right" / "avoid obstacle on left"
- Place obstacles on robot's left side
- Execute right turns of varying sharpness
- Save during the turn
```

**Session 4: Obstacle Avoidance (150 samples)**
```
Instruction: "avoid the obstacle" / "go around the box"
- Place obstacles directly ahead
- Navigate around them (mix of left and right)
- Save during entire maneuver
```

**Session 5: Object Approach (100 samples)**
```
Instruction: "go to the [red ball/toy/target]"
- Place a distinctive target object
- Drive toward it from various angles
- Save during approach
```

**Session 6: Stopping (50 samples)**
```
Instruction: "stop" / "wait" / "stay"
- Robot should be stationary
- L=0, R=0
- Various backgrounds
```

### Step 3: Quality Checks

After collection, verify:
- [ ] Images are clear, not blurry
- [ ] Motor values are reasonable (-1 to 1)
- [ ] Instructions match the actions
- [ ] Good variety of scenarios

## Common Mistakes to Avoid

### ❌ Mistake 1: Binary-style Collection
```
Bad: Take 100 "blocked" photos and 100 "free" photos
```
VLA needs continuous data showing HOW to navigate, not just IF there's an obstacle.

### ❌ Mistake 2: Same Instruction for Everything
```
Bad: All 500 samples have instruction "navigate forward"
```
The model won't learn to respond to different commands.

### ❌ Mistake 3: Inconsistent Actions
```
Bad: Same scenario but wildly different motor values
```
Be consistent: similar situations should have similar responses.

### ❌ Mistake 4: Static Images Only
```
Bad: Place robot, take photo, move robot, repeat
```
Collect data WHILE MOVING for more natural action patterns.

### ❌ Mistake 5: Too Few Samples
```
Bad: 100-200 samples total
```
VLA models need 500+ samples minimum, ideally 1000+.

## Sample Data Distribution

Aim for this distribution:

```
Forward motion:     25% (200 samples)
Left turns:         18% (150 samples)
Right turns:        18% (150 samples)
Obstacle avoidance: 18% (150 samples)
Object approach:    12% (100 samples)
Stopping:            6% (50 samples)
Reversing:           3% (25 samples)
--------------------------------
Total:             100% (825 samples)
```

## Instruction Templates

Use these as starting points:

**Navigation:**
- "go forward"
- "move ahead"
- "navigate forward avoiding obstacles"
- "proceed straight"
- "drive forward"

**Turning:**
- "turn left"
- "turn right"
- "rotate left"
- "rotate right"
- "go left"
- "go right"

**Avoidance:**
- "avoid the obstacle"
- "go around the [object]"
- "don't hit the [object]"
- "navigate around the obstacle"

**Object-directed:**
- "go to the [object]"
- "approach the [object]"
- "move toward the [object]"
- "find the [object]"

**Speed modifiers:**
- "slowly go forward"
- "quickly turn left"
- "carefully approach"

**Stopping:**
- "stop"
- "wait"
- "stay here"
- "don't move"

## Recording Tips

1. **Smooth movements**: Avoid jerky controls
2. **Consistent speeds**: Use similar speeds for similar scenarios
3. **Natural flow**: Collect during continuous driving, not just snapshots
4. **Good lighting**: Ensure images are well-lit
5. **Clean lens**: Check camera for smudges

## Validation

After collecting, run this to check your data:

```python
from server.vla_server.fine_tuning.dataset import JetBotVLADataset

dataset = JetBotVLADataset('dataset_vla')
stats = dataset.get_statistics()
print(stats)
```

Check that:
- `num_samples` >= 500
- `unique_instructions` >= 10
- `action_mean` is close to [0, 0] (balanced turns)
- `action_std` is reasonable (0.2-0.5)
