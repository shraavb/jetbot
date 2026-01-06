# VLA Evaluation Report

**Timestamp**: 20260106_030500

## Configuration

- Steps per instruction: 50
- Instructions tested: 4

## Per-Instruction Analysis

### ✅ `go forward`

| Metric | Value |
|--------|-------|
| Distance | 0.236m |
| Avg Left | 0.850 |
| Avg Right | 0.940 |
| R-L Diff | 0.090 |
| Forward % | 100.0% |
| Left Turn % | 0.0% |
| Right Turn % | 0.0% |

**Expected**: Both motors positive and similar
**Actual**: L=0.85, R=0.94 (forward)

### ✅ `turn left`

| Metric | Value |
|--------|-------|
| Distance | 0.191m |
| Avg Left | 0.570 |
| Avg Right | 0.920 |
| R-L Diff | 0.350 |
| Forward % | 100.0% |
| Left Turn % | 100.0% |
| Right Turn % | 0.0% |

**Expected**: Right motor > Left motor
**Actual**: R-L diff=0.35 (turning left)

### ❌ `turn right`

| Metric | Value |
|--------|-------|
| Distance | 0.208m |
| Avg Left | 0.640 |
| Avg Right | 0.950 |
| R-L Diff | 0.310 |
| Forward % | 100.0% |
| Left Turn % | 100.0% |
| Right Turn % | 0.0% |

**Expected**: Left motor > Right motor
**Actual**: R-L diff=0.31
**Notes**: Not turning right (turning left instead?)

### ✅ `go towards the red ball`

| Metric | Value |
|--------|-------|
| Distance | 0.166m |
| Avg Left | 0.860 |
| Avg Right | 0.450 |
| R-L Diff | -0.410 |
| Forward % | 100.0% |
| Left Turn % | 0.0% |
| Right Turn % | 100.0% |

**Expected**: Movement towards target
**Actual**: Moved 0.166m

## Summary

**Accuracy**: 3/4 (75.0%)

| Instruction | Status | Avg L | Avg R | Distance |
|-------------|--------|-------|-------|----------|
| go forward | ✅ | 0.85 | 0.94 | 0.236m |
| turn left | ✅ | 0.57 | 0.92 | 0.191m |
| turn right | ❌ | 0.64 | 0.95 | 0.208m |
| go towards the red ball | ✅ | 0.86 | 0.45 | 0.166m |