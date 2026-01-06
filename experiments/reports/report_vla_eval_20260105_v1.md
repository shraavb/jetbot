# VLA Evaluation Report

**Timestamp**: 20260105_193637

## Configuration

- Steps per instruction: 50
- Instructions tested: 4

## Per-Instruction Analysis

### ✅ `go forward`

| Metric | Value |
|--------|-------|
| Distance | 0.141m |
| Avg Left | 0.552 |
| Avg Right | 0.483 |
| R-L Diff | -0.069 |
| Forward % | 90.0% |
| Left Turn % | 0.0% |
| Right Turn % | 20.0% |

**Expected**: Both motors positive and similar
**Actual**: L=0.55, R=0.48 (forward)

### ✅ `turn left`

| Metric | Value |
|--------|-------|
| Distance | 0.141m |
| Avg Left | 0.405 |
| Avg Right | 0.727 |
| R-L Diff | 0.322 |
| Forward % | 70.0% |
| Left Turn % | 40.0% |
| Right Turn % | 0.0% |

**Expected**: Right motor > Left motor
**Actual**: R-L diff=0.32 (turning left)

### ❌ `turn right`

| Metric | Value |
|--------|-------|
| Distance | 0.129m |
| Avg Left | 0.339 |
| Avg Right | 0.671 |
| R-L Diff | 0.332 |
| Forward % | 70.0% |
| Left Turn % | 50.0% |
| Right Turn % | 0.0% |

**Expected**: Left motor > Right motor
**Actual**: R-L diff=0.33
**Notes**: Not turning right (turning left instead?)

### ✅ `go towards the red ball`

| Metric | Value |
|--------|-------|
| Distance | 0.055m |
| Avg Left | 0.326 |
| Avg Right | 0.132 |
| R-L Diff | -0.194 |
| Forward % | 0.0% |
| Left Turn % | 0.0% |
| Right Turn % | 50.0% |

**Expected**: Movement towards target
**Actual**: Moved 0.055m

## Summary

**Accuracy**: 3/4 (75.0%)

| Instruction | Status | Avg L | Avg R | Distance |
|-------------|--------|-------|-------|----------|
| go forward | ✅ | 0.55 | 0.48 | 0.141m |
| turn left | ✅ | 0.40 | 0.73 | 0.141m |
| turn right | ❌ | 0.34 | 0.67 | 0.129m |
| go towards the red ball | ✅ | 0.33 | 0.13 | 0.055m |