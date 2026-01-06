"""
VLA Experiment Results Analyzer

Analyze evaluation results from vla_experiment_test.py and generate
per-instruction metrics and visualizations.

Usage:
    python experiments/analyze_results.py --results experiments/results/eval.json
    python experiments/analyze_results.py --results-dir experiments/results/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def load_results(results_path: Path) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def analyze_instruction(result: Dict) -> Dict:
    """Analyze results for a single instruction."""
    actions = result.get('actions', [])

    # Handle pre-computed averages (simplified format)
    if not actions and 'avg_left' in result:
        avg_left = result['avg_left']
        avg_right = result['avg_right']
        avg_diff = avg_right - avg_left

        return {
            'instruction': result['instruction'],
            'num_steps': result.get('num_steps', 0),
            'distance_traveled': result.get('distance_traveled', 0),
            'avg_left': float(avg_left),
            'avg_right': float(avg_right),
            'std_left': 0.0,
            'std_right': 0.0,
            'avg_diff': float(avg_diff),
            'forward_ratio': 1.0 if avg_left > 0.3 and avg_right > 0.3 else 0.0,
            'left_turn_ratio': 1.0 if avg_diff > 0.2 else 0.0,
            'right_turn_ratio': 1.0 if avg_diff < -0.2 else 0.0,
        }

    if not actions:
        return {
            'instruction': result.get('instruction', 'unknown'),
            'num_steps': 0,
            'distance_traveled': 0,
            'avg_left': 0, 'avg_right': 0,
            'std_left': 0, 'std_right': 0,
            'avg_diff': 0,
            'forward_ratio': 0, 'left_turn_ratio': 0, 'right_turn_ratio': 0
        }

    left_speeds = [a[0] for a in actions]
    right_speeds = [a[1] for a in actions]

    # Calculate metrics
    avg_left = np.mean(left_speeds)
    avg_right = np.mean(right_speeds)
    std_left = np.std(left_speeds)
    std_right = np.std(right_speeds)

    # Direction analysis
    diff = np.array(right_speeds) - np.array(left_speeds)
    avg_diff = np.mean(diff)

    # Movement classification
    forward_ratio = np.mean([1 if l > 0.3 and r > 0.3 else 0 for l, r in actions])
    left_turn_ratio = np.mean([1 if r > l + 0.2 else 0 for l, r in actions])
    right_turn_ratio = np.mean([1 if l > r + 0.2 else 0 for l, r in actions])

    return {
        'instruction': result['instruction'],
        'num_steps': len(actions),
        'distance_traveled': result.get('distance_traveled', 0),
        'avg_left': float(avg_left),
        'avg_right': float(avg_right),
        'std_left': float(std_left),
        'std_right': float(std_right),
        'avg_diff': float(avg_diff),  # positive = turning left, negative = turning right
        'forward_ratio': float(forward_ratio),
        'left_turn_ratio': float(left_turn_ratio),
        'right_turn_ratio': float(right_turn_ratio),
    }


def assess_instruction(instruction: str, metrics: Dict) -> Dict:
    """Assess whether the model correctly follows the instruction."""
    instruction_lower = instruction.lower()
    assessment = {
        'expected_behavior': '',
        'actual_behavior': '',
        'correct': False,
        'notes': []
    }

    avg_left = metrics['avg_left']
    avg_right = metrics['avg_right']
    avg_diff = metrics['avg_diff']

    if 'forward' in instruction_lower or 'straight' in instruction_lower:
        assessment['expected_behavior'] = 'Both motors positive and similar'
        if avg_left > 0.3 and avg_right > 0.3 and abs(avg_diff) < 0.2:
            assessment['actual_behavior'] = f'L={avg_left:.2f}, R={avg_right:.2f} (forward)'
            assessment['correct'] = True
        else:
            assessment['actual_behavior'] = f'L={avg_left:.2f}, R={avg_right:.2f}'
            assessment['notes'].append('Not moving forward consistently')

    elif 'left' in instruction_lower and 'turn' in instruction_lower:
        assessment['expected_behavior'] = 'Right motor > Left motor'
        if avg_diff > 0.15:  # right > left means turning left
            assessment['actual_behavior'] = f'R-L diff={avg_diff:.2f} (turning left)'
            assessment['correct'] = True
        else:
            assessment['actual_behavior'] = f'R-L diff={avg_diff:.2f}'
            assessment['notes'].append('Not turning left consistently')

    elif 'right' in instruction_lower and 'turn' in instruction_lower:
        assessment['expected_behavior'] = 'Left motor > Right motor'
        if avg_diff < -0.15:  # left > right means turning right
            assessment['actual_behavior'] = f'R-L diff={avg_diff:.2f} (turning right)'
            assessment['correct'] = True
        else:
            assessment['actual_behavior'] = f'R-L diff={avg_diff:.2f}'
            assessment['notes'].append('Not turning right (turning left instead?)')

    elif 'stop' in instruction_lower or 'wait' in instruction_lower:
        assessment['expected_behavior'] = 'Both motors near zero'
        if abs(avg_left) < 0.2 and abs(avg_right) < 0.2:
            assessment['actual_behavior'] = f'L={avg_left:.2f}, R={avg_right:.2f} (stopped)'
            assessment['correct'] = True
        else:
            assessment['actual_behavior'] = f'L={avg_left:.2f}, R={avg_right:.2f}'
            assessment['notes'].append('Not stopped')

    elif 'towards' in instruction_lower or 'approach' in instruction_lower:
        assessment['expected_behavior'] = 'Movement towards target'
        # This is harder to assess without visual context
        if metrics['distance_traveled'] > 0.05:
            assessment['actual_behavior'] = f'Moved {metrics["distance_traveled"]:.3f}m'
            assessment['correct'] = True  # Partial success if moving
        else:
            assessment['actual_behavior'] = 'Minimal movement'
            assessment['notes'].append('Need visual grounding to fully assess')

    else:
        assessment['expected_behavior'] = 'Unknown instruction type'
        assessment['actual_behavior'] = f'L={avg_left:.2f}, R={avg_right:.2f}'
        assessment['notes'].append('Cannot auto-assess this instruction type')

    return assessment


def generate_report(results: Dict, output_path: Optional[Path] = None) -> str:
    """Generate a markdown report of the evaluation results."""
    lines = []
    lines.append("# VLA Evaluation Report\n")
    lines.append(f"**Timestamp**: {results.get('timestamp', 'N/A')}\n")

    config = results.get('config', {})
    lines.append("## Configuration\n")
    lines.append(f"- Steps per instruction: {config.get('steps_per_instruction', 'N/A')}")
    lines.append(f"- Instructions tested: {len(config.get('instructions', []))}\n")

    lines.append("## Per-Instruction Analysis\n")

    all_metrics = []
    for result in results.get('results', []):
        metrics = analyze_instruction(result)
        assessment = assess_instruction(result['instruction'], metrics)
        all_metrics.append({**metrics, **assessment})

        status = "✅" if assessment['correct'] else "❌"
        lines.append(f"### {status} `{result['instruction']}`\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Distance | {metrics['distance_traveled']:.3f}m |")
        lines.append(f"| Avg Left | {metrics['avg_left']:.3f} |")
        lines.append(f"| Avg Right | {metrics['avg_right']:.3f} |")
        lines.append(f"| R-L Diff | {metrics['avg_diff']:.3f} |")
        lines.append(f"| Forward % | {metrics['forward_ratio']*100:.1f}% |")
        lines.append(f"| Left Turn % | {metrics['left_turn_ratio']*100:.1f}% |")
        lines.append(f"| Right Turn % | {metrics['right_turn_ratio']*100:.1f}% |\n")

        lines.append(f"**Expected**: {assessment['expected_behavior']}")
        lines.append(f"**Actual**: {assessment['actual_behavior']}")
        if assessment['notes']:
            lines.append(f"**Notes**: {'; '.join(assessment['notes'])}\n")
        else:
            lines.append("")

    # Summary
    lines.append("## Summary\n")
    correct = sum(1 for m in all_metrics if m['correct'])
    total = len(all_metrics)
    lines.append(f"**Accuracy**: {correct}/{total} ({100*correct/total:.1f}%)\n")

    lines.append("| Instruction | Status | Avg L | Avg R | Distance |")
    lines.append("|-------------|--------|-------|-------|----------|")
    for m in all_metrics:
        status = "✅" if m['correct'] else "❌"
        lines.append(f"| {m['instruction']} | {status} | {m['avg_left']:.2f} | {m['avg_right']:.2f} | {m['distance_traveled']:.3f}m |")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Analyze VLA evaluation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--results',
        type=Path,
        help='Path to single results JSON file'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('experiments/results'),
        help='Directory containing results JSON files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path for report (defaults to experiments/reports/)'
    )
    args = parser.parse_args()

    if args.results:
        results_files = [args.results]
    else:
        results_files = sorted(args.results_dir.glob('*.json'))

    if not results_files:
        print("No results files found")
        return

    for results_file in results_files:
        print(f"\nAnalyzing: {results_file}")
        results = load_results(results_file)

        if args.output:
            output_path = args.output
        else:
            reports_dir = Path('experiments/reports')
            reports_dir.mkdir(parents=True, exist_ok=True)
            output_path = reports_dir / f"report_{results_file.stem}.md"

        report = generate_report(results, output_path)
        print(report)


if __name__ == '__main__':
    main()
