"""
Capture Portfolio Photo of JetBot in Isaac Sim

Creates a visually appealing scene with the JetBot and various objects,
then captures a high-resolution image for portfolio use.

Usage (run from RunPod terminal):
    python -m jetbot.isaac_sim.capture_portfolio_photo --output jetbot_portfolio.png

    # With custom camera position
    python -m jetbot.isaac_sim.capture_portfolio_photo --camera-pos 1.5 1.5 0.8

    # Higher resolution
    python -m jetbot.isaac_sim.capture_portfolio_photo --resolution 3840 2160
"""

import argparse
import numpy as np
from pathlib import Path


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (radians) to quaternion [w, x, y, z]."""
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)

    return np.array([
        cr * cp * cy + sr * sp * sy,  # w
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy   # z
    ])


def look_at_quaternion(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Calculate quaternion for camera to look at target position."""
    direction = target_pos - camera_pos
    direction = direction / np.linalg.norm(direction)

    # Calculate yaw and pitch
    yaw = np.arctan2(direction[1], direction[0])
    pitch = -np.arcsin(direction[2])

    return euler_to_quaternion(0, pitch, yaw)


def capture_portfolio_photo(
    output_path: str = "jetbot_portfolio.png",
    resolution: tuple = (1920, 1080),
    camera_position: tuple = (1.2, 1.2, 0.6),
    look_at: tuple = (0.0, 0.0, 0.1)
):
    """
    Capture a portfolio photo of the JetBot scene.

    Args:
        output_path: Path to save the image
        resolution: Image resolution (width, height)
        camera_position: Camera position (x, y, z)
        look_at: Point the camera looks at (x, y, z)
    """
    from isaacsim import SimulationApp

    # Start simulation with rendering enabled
    print("Starting Isaac Sim...")
    simulation_app = SimulationApp({
        "headless": True,
        "width": resolution[0],
        "height": resolution[1],
        "anti_aliasing": 3,  # FXAA
    })

    # Now import Isaac Sim modules
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicSphere, VisualCuboid, VisualCylinder
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.sensor import Camera
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    import omni.kit.commands
    import omni.usd

    print("Creating world...")
    world = World(physics_dt=1/60.0, rendering_dt=1/60.0)

    # ========================================
    # Create Ground Plane
    # ========================================
    print("Adding ground plane...")
    ground = world.scene.add(
        VisualCuboid(
            prim_path="/World/ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.01]),
            scale=np.array([10.0, 10.0, 0.02]),
            color=np.array([0.3, 0.3, 0.35])  # Nice gray floor
        )
    )

    # ========================================
    # Add JetBot Robot
    # ========================================
    print("Adding JetBot...")
    try:
        assets_root = get_assets_root_path()
        jetbot_usd = f"{assets_root}/Isaac/Robots/Jetbot/jetbot.usd"

        omni.kit.commands.execute(
            'CreateReferenceCommand',
            usd_context=omni.usd.get_context(),
            path_to="/World/JetBot",
            asset_path=jetbot_usd
        )

        jetbot_xform = XFormPrim(prim_path="/World/JetBot")
        jetbot_xform.set_world_pose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        print("  JetBot loaded from USD")
    except Exception as e:
        print(f"  Warning: Could not load JetBot USD ({e}), creating visual approximation...")
        # Create visual approximation
        world.scene.add(
            VisualCuboid(
                prim_path="/World/JetBot/body",
                name="jetbot_body",
                position=np.array([0.0, 0.0, 0.04]),
                scale=np.array([0.12, 0.10, 0.04]),
                color=np.array([0.1, 0.1, 0.1])
            )
        )

    # ========================================
    # Add Red Ball
    # ========================================
    print("Adding red ball...")
    red_ball = world.scene.add(
        DynamicSphere(
            prim_path="/World/Objects/red_ball",
            name="red_ball",
            position=np.array([0.8, -0.3, 0.12]),
            radius=0.12,
            color=np.array([0.9, 0.1, 0.1]),  # Red
            mass=0.1
        )
    )

    # ========================================
    # Add Blue Ball
    # ========================================
    print("Adding blue ball...")
    blue_ball = world.scene.add(
        DynamicSphere(
            prim_path="/World/Objects/blue_ball",
            name="blue_ball",
            position=np.array([0.5, 0.6, 0.10]),
            radius=0.10,
            color=np.array([0.1, 0.3, 0.9]),  # Blue
            mass=0.1
        )
    )

    # ========================================
    # Add Green Ball
    # ========================================
    print("Adding green ball...")
    green_ball = world.scene.add(
        DynamicSphere(
            prim_path="/World/Objects/green_ball",
            name="green_ball",
            position=np.array([-0.4, 0.5, 0.08]),
            radius=0.08,
            color=np.array([0.1, 0.8, 0.2]),  # Green
            mass=0.1
        )
    )

    # ========================================
    # Add Chair
    # ========================================
    print("Adding chair...")
    chair_color = np.array([0.55, 0.27, 0.07])  # Brown
    chair_pos = [1.0, 0.5, 0.0]
    seat_height = 0.45

    # Seat
    world.scene.add(
        VisualCuboid(
            prim_path="/World/Objects/chair_seat",
            name="chair_seat",
            position=np.array([chair_pos[0], chair_pos[1], seat_height]),
            scale=np.array([0.4, 0.4, 0.05]),
            color=chair_color
        )
    )
    # Backrest
    world.scene.add(
        VisualCuboid(
            prim_path="/World/Objects/chair_back",
            name="chair_back",
            position=np.array([chair_pos[0] - 0.175, chair_pos[1], seat_height + 0.25]),
            scale=np.array([0.05, 0.4, 0.5]),
            color=chair_color
        )
    )
    # Legs
    leg_positions = [
        [chair_pos[0] - 0.15, chair_pos[1] - 0.15, seat_height / 2],
        [chair_pos[0] - 0.15, chair_pos[1] + 0.15, seat_height / 2],
        [chair_pos[0] + 0.15, chair_pos[1] - 0.15, seat_height / 2],
        [chair_pos[0] + 0.15, chair_pos[1] + 0.15, seat_height / 2],
    ]
    for i, leg_pos in enumerate(leg_positions):
        world.scene.add(
            VisualCuboid(
                prim_path=f"/World/Objects/chair_leg{i}",
                name=f"chair_leg{i}",
                position=np.array(leg_pos),
                scale=np.array([0.04, 0.04, seat_height]),
                color=chair_color
            )
        )

    # ========================================
    # Add Person (Cylinder + Head)
    # ========================================
    print("Adding person...")
    person_pos = [-0.6, -0.4, 0.0]
    person_color = np.array([0.2, 0.4, 0.8])  # Blue clothing
    body_height = 1.2

    # Body
    world.scene.add(
        VisualCylinder(
            prim_path="/World/Objects/person_body",
            name="person_body",
            position=np.array([person_pos[0], person_pos[1], body_height / 2]),
            radius=0.18,
            height=body_height,
            color=person_color
        )
    )
    # Head
    world.scene.add(
        VisualCylinder(
            prim_path="/World/Objects/person_head",
            name="person_head",
            position=np.array([person_pos[0], person_pos[1], body_height + 0.12]),
            radius=0.12,
            height=0.24,
            color=np.array([0.9, 0.75, 0.65])  # Skin tone
        )
    )

    # ========================================
    # Add Door with Frame
    # ========================================
    print("Adding door...")
    door_pos = [2.0, 0.0, 0.0]
    door_color = np.array([0.45, 0.25, 0.1])  # Brown door
    door_width = 0.9
    door_height = 2.0

    # Door panel
    world.scene.add(
        VisualCuboid(
            prim_path="/World/Objects/door",
            name="door",
            position=np.array([door_pos[0], door_pos[1], door_height / 2]),
            scale=np.array([0.05, door_width, door_height]),
            color=door_color
        )
    )
    # Door frame
    frame_color = np.array([0.25, 0.25, 0.25])
    world.scene.add(
        VisualCuboid(
            prim_path="/World/Objects/door_frame_top",
            name="door_frame_top",
            position=np.array([door_pos[0], door_pos[1], door_height + 0.04]),
            scale=np.array([0.1, door_width + 0.2, 0.08]),
            color=frame_color
        )
    )
    for side, y_offset in [('left', -door_width/2 - 0.05), ('right', door_width/2 + 0.05)]:
        world.scene.add(
            VisualCuboid(
                prim_path=f"/World/Objects/door_frame_{side}",
                name=f"door_frame_{side}",
                position=np.array([door_pos[0], door_pos[1] + y_offset, door_height / 2]),
                scale=np.array([0.1, 0.1, door_height]),
                color=frame_color
            )
        )

    # ========================================
    # Add Orange Cone
    # ========================================
    print("Adding orange cone...")
    world.scene.add(
        VisualCylinder(
            prim_path="/World/Objects/orange_cone",
            name="orange_cone",
            position=np.array([0.3, -0.6, 0.15]),
            radius=0.12,
            height=0.30,
            color=np.array([1.0, 0.5, 0.0])  # Orange
        )
    )

    # ========================================
    # Add Yellow Box
    # ========================================
    print("Adding yellow box...")
    world.scene.add(
        VisualCuboid(
            prim_path="/World/Objects/yellow_box",
            name="yellow_box",
            position=np.array([-0.5, 0.0, 0.1]),
            scale=np.array([0.2, 0.2, 0.2]),
            color=np.array([0.95, 0.85, 0.1])  # Yellow
        )
    )

    # ========================================
    # Add Another JetBot (as obstacle)
    # ========================================
    print("Adding second JetBot...")
    try:
        omni.kit.commands.execute(
            'CreateReferenceCommand',
            usd_context=omni.usd.get_context(),
            path_to="/World/Objects/other_jetbot",
            asset_path=jetbot_usd
        )

        other_jetbot = XFormPrim(prim_path="/World/Objects/other_jetbot")
        # Position it facing the main JetBot
        other_jetbot.set_world_pose(
            position=np.array([0.6, 0.2, 0.0]),
            orientation=euler_to_quaternion(0, 0, np.pi)  # Facing opposite direction
        )
        print("  Second JetBot loaded")
    except Exception as e:
        print(f"  Could not add second JetBot: {e}")

    # ========================================
    # Create Portfolio Camera
    # ========================================
    print("Setting up portfolio camera...")
    camera_pos = np.array(camera_position)
    target_pos = np.array(look_at)
    camera_quat = look_at_quaternion(camera_pos, target_pos)

    portfolio_camera = Camera(
        prim_path="/World/PortfolioCamera",
        position=camera_pos,
        orientation=camera_quat,
        frequency=30,
        resolution=resolution
    )
    portfolio_camera.initialize()
    portfolio_camera.set_focal_length(24.0)  # Wide angle for scene capture

    # ========================================
    # Render and Capture
    # ========================================
    print("Initializing world and rendering...")
    world.reset()

    # Step multiple times to ensure proper rendering
    print("Warming up renderer (this may take a moment)...")
    for i in range(30):
        world.step(render=True)
        if (i + 1) % 10 == 0:
            print(f"  Render step {i + 1}/30")

    # Capture the image
    print("Capturing image...")
    rgba = portfolio_camera.get_rgba()

    if rgba is None:
        print("ERROR: Camera returned None. Trying again...")
        for _ in range(10):
            world.step(render=True)
        rgba = portfolio_camera.get_rgba()

    if rgba is not None:
        rgb = rgba[:, :, :3]

        # Save with PIL
        from PIL import Image
        img = Image.fromarray(rgb)

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        img.save(str(output_file), quality=95)
        print(f"\nSaved portfolio image to: {output_file.absolute()}")
        print(f"Resolution: {resolution[0]}x{resolution[1]}")
    else:
        print("ERROR: Failed to capture image from camera")

    # Cleanup
    print("Closing simulation...")
    simulation_app.close()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Capture a portfolio photo of JetBot in Isaac Sim',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic capture
  python -m jetbot.isaac_sim.capture_portfolio_photo

  # Custom output path
  python -m jetbot.isaac_sim.capture_portfolio_photo --output /path/to/photo.png

  # Custom camera position (x y z)
  python -m jetbot.isaac_sim.capture_portfolio_photo --camera-pos 2.0 2.0 1.0

  # High resolution (4K)
  python -m jetbot.isaac_sim.capture_portfolio_photo --resolution 3840 2160

  # Look at specific point
  python -m jetbot.isaac_sim.capture_portfolio_photo --look-at 0.5 0.0 0.2
        """
    )

    parser.add_argument(
        '--output', '-o',
        default='jetbot_portfolio.png',
        help='Output image path (default: jetbot_portfolio.png)'
    )
    parser.add_argument(
        '--resolution', '-r',
        nargs=2,
        type=int,
        default=[1920, 1080],
        metavar=('WIDTH', 'HEIGHT'),
        help='Image resolution (default: 1920 1080)'
    )
    parser.add_argument(
        '--camera-pos', '-c',
        nargs=3,
        type=float,
        default=[1.2, 1.2, 0.6],
        metavar=('X', 'Y', 'Z'),
        help='Camera position (default: 1.2 1.2 0.6)'
    )
    parser.add_argument(
        '--look-at', '-l',
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.1],
        metavar=('X', 'Y', 'Z'),
        help='Point camera looks at (default: 0.0 0.0 0.1)'
    )

    args = parser.parse_args()

    capture_portfolio_photo(
        output_path=args.output,
        resolution=tuple(args.resolution),
        camera_position=tuple(args.camera_pos),
        look_at=tuple(args.look_at)
    )


if __name__ == '__main__':
    main()