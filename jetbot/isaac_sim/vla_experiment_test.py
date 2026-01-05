"""
Simplified VLA Experiment Test

A standalone script to test the trained VLA model in Isaac Sim
without JetBot hardware dependencies.

Usage:
    # First, start the VLA server in Terminal 1:
    cd /workspace/jetbot
    python -c "
    import os
    import zmq
    import torch
    from PIL import Image
    import io
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from server.vla_server.fine_tuning.jetbot_action_head import JetBotActionHead

    print('Loading SmolVLM-500M-Instruct...')
    model_id = 'HuggingFaceTB/SmolVLM-500M-Instruct'
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).cuda().eval()

    print('Loading action head...')
    action_head = JetBotActionHead(hidden_size=960, intermediate_size=128, dropout=0.1, action_dim=2)
    state_dict = torch.load('/workspace/jetbot/models/smolvla_jetbot/best/jetbot_action_head.pt')
    action_head.load_state_dict(state_dict)
    action_head = action_head.cuda().half().eval()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:5555')
    print('VLA server ready on port 5555!')

    while True:
        request = socket.recv_multipart()
        jpeg_bytes, instruction = request[0], request[1].decode('utf-8')
        print(f'Received: {instruction}')

        image = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB').resize((224, 224))
        prompt = f'<image>Robot instruction: {instruction}'
        inputs = processor(text=prompt, images=image, return_tensors='pt')
        inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, -1, :]
            left, right = action_head.get_actions(hidden)

        print(f'Action: L={left:.3f}, R={right:.3f}')
        socket.send_multipart([f'{left:.6f}'.encode(), f'{right:.6f}'.encode()])
    "

    # Then run this script in Terminal 2:
    python jetbot/isaac_sim/vla_experiment_test.py
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import zmq
from PIL import Image
import io

# Suppress verbose Omniverse/Carb warnings
os.environ.setdefault('CARB_LOG_LEVEL', 'error')
os.environ.setdefault('OMNI_LOG_LEVEL', 'error')


def main():
    parser = argparse.ArgumentParser(description='Simplified VLA Experiment Test')
    parser.add_argument('--host', default='localhost', help='VLA server host')
    parser.add_argument('--port', type=int, default=5555, help='VLA server port')
    parser.add_argument('--steps', type=int, default=20, help='Steps per instruction')
    parser.add_argument('--instructions', nargs='+',
                        default=['go forward', 'turn left', 'turn right', 'go towards the red ball'],
                        help='Instructions to test')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results (default: results/vla_eval_TIMESTAMP.json)')
    args = parser.parse_args()

    # Setup Isaac Sim
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({'headless': True})

    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.wheeled_robots.robots import WheeledRobot
    from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
    from omni.isaac.sensor import Camera
    from omni.isaac.core.objects import DynamicSphere

    print('Setting up world...')
    world = World()
    world.scene.add_default_ground_plane()

    # Add JetBot
    jetbot = world.scene.add(
        WheeledRobot(
            prim_path='/World/JetBot',
            name='jetbot',
            wheel_dof_names=['left_wheel_joint', 'right_wheel_joint'],
            create_robot=True,
            usd_path='/workspace/assets/jetbot.usd',
            position=np.array([0.0, 0.0, 0.0])
        )
    )

    controller = DifferentialController(name='ctrl', wheel_radius=0.03, wheel_base=0.1125)

    # Add camera
    camera = Camera(prim_path='/World/JetBot/chassis/camera', resolution=(224, 224))

    # Add a red ball target
    red_ball = world.scene.add(
        DynamicSphere(
            prim_path='/World/RedBall',
            name='red_ball',
            position=np.array([1.5, 0.5, 0.1]),
            radius=0.1,
            color=np.array([1.0, 0.0, 0.0])
        )
    )

    world.reset()
    camera.initialize()

    # Connect to VLA server
    print(f'Connecting to VLA server at {args.host}:{args.port}...')
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.connect(f'tcp://{args.host}:{args.port}')

    print(f'\n{"="*60}')
    print(f'Running VLA Experiment Test')
    print(f'Instructions: {len(args.instructions)}')
    print(f'Steps per instruction: {args.steps}')
    print(f'{"="*60}\n')

    results = []

    for idx, instruction in enumerate(args.instructions):
        print(f'[{idx+1}/{len(args.instructions)}] Testing: "{instruction}"')
        world.reset()

        start_pos, _ = jetbot.get_world_pose()
        trajectory = [start_pos.tolist()]
        actions = []

        for step in range(args.steps):
            world.step(render=True)

            # Get camera image
            rgba = camera.get_rgba()
            if rgba is None:
                continue
            rgb = rgba[:, :, :3]

            # Convert to JPEG
            img = Image.fromarray(rgb)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')

            # Send to VLA server
            try:
                socket.send_multipart([img_bytes.getvalue(), instruction.encode()])
                response = socket.recv_multipart()
                left, right = float(response[0]), float(response[1])
            except zmq.error.Again:
                print(f'  Timeout at step {step}')
                left, right = 0.0, 0.0

            actions.append((left, right))

            # Apply action
            linear = (left + right) / 2.0 * 0.3
            angular = (right - left) / 0.1125 * 0.3
            wheel_actions = controller.forward(command=[linear, angular])
            jetbot.apply_wheel_actions(wheel_actions)

            pos, _ = jetbot.get_world_pose()
            trajectory.append(pos.tolist())

            if step % 5 == 0:
                print(f'  Step {step}: pos=({pos[0]:.2f}, {pos[1]:.2f}), action=L:{left:.2f} R:{right:.2f}')

        final_pos, _ = jetbot.get_world_pose()
        distance_traveled = np.linalg.norm(np.array(final_pos[:2]) - np.array(start_pos[:2]))

        result = {
            'instruction': instruction,
            'start_pos': start_pos.tolist(),
            'final_pos': final_pos.tolist(),
            'distance_traveled': float(distance_traveled),
            'num_steps': len(actions),
            'avg_left': float(np.mean([a[0] for a in actions])),
            'avg_right': float(np.mean([a[1] for a in actions])),
            'actions': [(float(a[0]), float(a[1])) for a in actions],
            'trajectory': trajectory
        }
        results.append(result)

        print(f'  Completed: traveled {distance_traveled:.3f}m, avg action=({result["avg_left"]:.2f}, {result["avg_right"]:.2f})\n')

    # Summary
    print(f'{"="*60}')
    print('EXPERIMENT SUMMARY')
    print(f'{"="*60}')
    for r in results:
        print(f'  "{r["instruction"]}": traveled {r["distance_traveled"]:.3f}m')
    print(f'{"="*60}')

    # Save results to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('results') / f'vla_eval_{timestamp}.json'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_data = {
        'timestamp': timestamp,
        'config': {
            'host': args.host,
            'port': args.port,
            'steps_per_instruction': args.steps,
            'instructions': args.instructions
        },
        'results': results,
        'summary': {
            'total_instructions': len(results),
            'avg_distance': float(np.mean([r['distance_traveled'] for r in results])),
            'total_distance': float(sum(r['distance_traveled'] for r in results))
        }
    }

    with open(output_path, 'w') as f:
        json.dump(experiment_data, f, indent=2)

    print(f'\nResults saved to: {output_path}')
    print('\nVLA evaluation complete!')
    simulation_app.close()


if __name__ == '__main__':
    main()
