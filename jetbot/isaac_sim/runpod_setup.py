#!/usr/bin/env python3
"""
RunPod Setup Script for JetBot VLA Simulation

Self-contained script that runs Isaac Sim without JetBot hardware dependencies.
Includes domain randomization for visual diversity.

Usage:
    python runpod_setup.py --download-assets
    python runpod_setup.py --test-sim
    python runpod_setup.py --collect-data --episodes 100 --output /workspace/sim_data
"""

import os
import sys
import json
import time
import uuid
import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


# Default navigation instructions for data collection
DEFAULT_INSTRUCTIONS = [
    "go forward",
    "move forward",
    "drive straight ahead",
    "turn left",
    "turn right",
    "rotate left",
    "rotate right",
    "go backward",
    "move back",
    "stop",
    "halt",
    "avoid the obstacle",
    "go around the obstacle",
    "approach the red cube",
    "approach the blue cube",
    "approach the green cube",
    "move toward the target",
    "navigate to the goal",
    "go to the red object",
    "go to the blue object",
    "turn around",
    "make a u-turn"
]

# Colors for domain randomization (RGB normalized 0-1)
OBSTACLE_COLORS = {
    'red': (0.9, 0.1, 0.1),
    'green': (0.1, 0.8, 0.1),
    'blue': (0.1, 0.1, 0.9),
    'yellow': (0.9, 0.9, 0.1),
    'orange': (0.9, 0.5, 0.1),
    'purple': (0.6, 0.1, 0.8),
    'cyan': (0.1, 0.8, 0.8),
    'white': (0.9, 0.9, 0.9),
}

GROUND_COLORS = [
    (0.3, 0.3, 0.3),   # Gray
    (0.4, 0.3, 0.2),   # Brown
    (0.2, 0.2, 0.25),  # Dark gray
    (0.5, 0.5, 0.45),  # Light gray
    (0.3, 0.25, 0.2),  # Dark brown
    (0.6, 0.55, 0.5),  # Beige
]

# Isaac Sim Nucleus assets - paths relative to assets root
# These will be tried if Nucleus is available; fallback to primitives if not
NUCLEUS_ASSETS = {
    # Robots (relative paths from Isaac assets root)
    'robots': [
        '/Isaac/Robots/Jetbot/jetbot.usd',
        '/Isaac/Robots/Carter/carter_v1.usd',
        '/Isaac/Robots/Transporter/transporter.usd',
        '/Isaac/Robots/Kaya/kaya.usd',
    ],
    # People/Characters
    'people': [
        '/Isaac/People/Characters/original_male_adult_police_04/male_adult_police_04.usd',
        '/Isaac/People/Characters/original_male_adult_medical_01/male_adult_medical_01.usd',
        '/Isaac/People/Characters/original_female_adult_police_02/female_adult_police_02.usd',
        '/Isaac/People/Characters/original_male_adult_construction_03/male_adult_construction_03.usd',
    ],
    # Props and furniture (common warehouse/industrial items)
    'props': [
        '/Isaac/Props/Forklift/forklift.usd',
        '/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd',
        '/Isaac/Environments/Simple_Warehouse/Props/SM_BarrelPlastic_A_01.usd',
        '/Isaac/Environments/Simple_Warehouse/Props/SM_BarrelPlastic_A_02.usd',
        '/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd',
        '/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd',
        '/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd',
        '/Isaac/Environments/Simple_Warehouse/Props/SM_PushcartA_02.usd',
        '/Isaac/Environments/Simple_Warehouse/Props/SM_RackShelf_01.usd',
    ],
    # Traffic/safety items
    'traffic': [
        '/Isaac/Environments/Simple_Warehouse/Props/SM_TrafficConeA_01.usd',
        '/Isaac/Environments/Simple_Warehouse/Props/SM_SafetyBarrierA_01.usd',
    ],
}

# Cache for Nucleus asset availability (checked once per session)
_nucleus_assets_cache = {'checked': False, 'available': [], 'root_path': None}


def check_nucleus_assets() -> dict:
    """
    Check which Nucleus assets are available.
    Results are cached for the session.

    Returns dict with:
        - 'available': list of full asset paths that exist
        - 'root_path': the assets root path
    """
    global _nucleus_assets_cache

    if _nucleus_assets_cache['checked']:
        return _nucleus_assets_cache

    try:
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        import omni.client

        root_path = get_assets_root_path()
        if root_path is None:
            print("Warning: Nucleus assets root path not available")
            _nucleus_assets_cache['checked'] = True
            return _nucleus_assets_cache

        _nucleus_assets_cache['root_path'] = root_path
        available = []

        # Check each asset category
        for category, paths in NUCLEUS_ASSETS.items():
            for rel_path in paths:
                full_path = root_path + rel_path
                # Check if asset exists
                result, entry = omni.client.stat(full_path)
                if result == omni.client.Result.OK:
                    available.append({
                        'category': category,
                        'path': full_path,
                        'rel_path': rel_path
                    })

        _nucleus_assets_cache['available'] = available
        _nucleus_assets_cache['checked'] = True

        print(f"Found {len(available)} Nucleus assets available:")
        for cat in ['robots', 'people', 'props', 'traffic']:
            count = len([a for a in available if a['category'] == cat])
            if count > 0:
                print(f"  - {cat}: {count} assets")

    except Exception as e:
        print(f"Warning: Could not check Nucleus assets: {e}")
        _nucleus_assets_cache['checked'] = True

    return _nucleus_assets_cache


def spawn_nucleus_asset(stage, asset_path: str, prim_path: str, position: tuple, rotation_deg: float = 0, scale: float = 1.0) -> bool:
    """
    Spawn a Nucleus asset at the given position.

    Args:
        stage: USD stage
        asset_path: Full path to the USD asset
        prim_path: Path for the new prim
        position: (x, y, z) world position
        rotation_deg: Rotation around Z axis in degrees
        scale: Scale factor (Nucleus assets are often in cm, need 0.01 for m)

    Returns:
        True if successful, False otherwise
    """
    try:
        from pxr import UsdGeom, Gf, Sdf
        from omni.isaac.core.utils.prims import create_prim, delete_prim

        # Create a parent Xform for our transform, then add child with reference
        # This avoids conflicts with xform ops in the referenced USD file
        parent_prim = create_prim(prim_path, "Xform")

        # Set transform on parent (no conflicts since it's our own prim)
        xform = UsdGeom.Xformable(parent_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, rotation_deg))
        xform.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))  # Use Vec3d to match typical USD precision

        # Add reference as a child prim to avoid xform op conflicts
        child_path = prim_path + "/asset"
        child_prim = create_prim(child_path, "Xform")
        child_prim.GetReferences().AddReference(asset_path)

        return True

    except Exception as e:
        print(f"Warning: Could not spawn Nucleus asset {asset_path}: {e}")
        # Clean up the partially created prim so fallback can use this path
        try:
            from omni.isaac.core.utils.prims import delete_prim
            delete_prim(prim_path)
        except:
            pass  # If delete fails, the fallback will handle it
        return False


def download_jetbot_asset(output_dir: str = "/workspace/assets") -> str:
    """Download JetBot USD asset from NVIDIA."""
    import urllib.request

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jetbot_url = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Robots/Jetbot/jetbot.usd"
    jetbot_path = output_path / "jetbot.usd"

    if jetbot_path.exists():
        print(f"JetBot asset already exists at {jetbot_path}")
        return str(jetbot_path)

    print(f"Downloading JetBot asset to {jetbot_path}...")
    urllib.request.urlretrieve(jetbot_url, jetbot_path)
    print("Download complete!")

    return str(jetbot_path)


def create_random_obstacles(stage, num_obstacles: int = 3, robot_pos: tuple = (0, 0), robot_yaw: float = 0, jetbot_asset_path: str = None, use_nucleus_assets: bool = True) -> List[str]:
    """
    Create random colored obstacles in the scene, positioned in front of the robot.
    Includes diverse shapes: basic geometry, cones, compound objects, other JetBots,
    and Nucleus assets (people, furniture, warehouse props) if available.

    Args:
        stage: USD stage
        num_obstacles: Number of obstacles to create
        robot_pos: (x, y) position of robot in world coords
        robot_yaw: Robot's yaw angle in radians (0 = facing +X)
        jetbot_asset_path: Path to JetBot USD for spawning other robots
        use_nucleus_assets: Try to use Nucleus assets (people, furniture, etc.)

    Returns list of prim paths for cleanup.
    """
    from pxr import UsdGeom, Gf, UsdShade, Sdf
    from omni.isaac.core.utils.prims import create_prim, delete_prim

    def safe_create_prim(prim_path: str, prim_type: str):
        """Create a prim, deleting any existing prim at the path first."""
        # Check if prim exists and delete it
        existing = stage.GetPrimAtPath(prim_path)
        if existing and existing.IsValid():
            delete_prim(prim_path)
        return create_prim(prim_path, prim_type)

    obstacle_paths = []

    # Check for available Nucleus assets
    nucleus_cache = None
    nucleus_available = []
    if use_nucleus_assets:
        nucleus_cache = check_nucleus_assets()
        nucleus_available = nucleus_cache.get('available', [])

    # Extended shape types with weights (more common shapes have higher weight)
    shape_types = [
        ('Cube', 12),           # Basic cube
        ('Cylinder', 12),       # Basic cylinder
        ('Sphere', 12),         # Basic sphere
        ('Cone', 10),           # Traffic cone shape
        ('TallCylinder', 6),    # Pole/barrier
        ('FlatBox', 6),         # Low obstacle like a box on ground
        ('TallBox', 6),         # Chair-like tall object
        ('Capsule', 5),         # Rounded cylinder (pill shape)
        ('Pyramid', 4),         # Pyramid shape (using cone)
        ('OtherJetBot', 7),     # Another JetBot robot
    ]

    # Add Nucleus asset types if available
    if nucleus_available:
        # Add categories based on what's available
        people_assets = [a for a in nucleus_available if a['category'] == 'people']
        props_assets = [a for a in nucleus_available if a['category'] == 'props']
        traffic_assets = [a for a in nucleus_available if a['category'] == 'traffic']
        robot_assets = [a for a in nucleus_available if a['category'] == 'robots']

        if people_assets:
            shape_types.append(('NucleusPerson', 8))  # People/humans
        if props_assets:
            shape_types.append(('NucleusProp', 10))   # Barrels, boxes, furniture
        if traffic_assets:
            shape_types.append(('NucleusTraffic', 6)) # Traffic cones, barriers
        if robot_assets:
            shape_types.append(('NucleusRobot', 5))   # Other robots

    # Create weighted choice list
    shapes = []
    weights = []
    for shape, weight in shape_types:
        shapes.append(shape)
        weights.append(weight)
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    for i in range(num_obstacles):
        # Random position in robot's local frame (in front of robot)
        local_x = np.random.uniform(0.3, 1.5)  # Forward distance (extended range)
        local_y = np.random.uniform(-0.8, 0.8)  # Lateral spread (wider)

        # Transform to world coordinates based on robot position and yaw
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        x = robot_pos[0] + local_x * cos_yaw - local_y * sin_yaw
        y = robot_pos[1] + local_x * sin_yaw + local_y * cos_yaw
        z = 0.0  # At ground level

        # Random color
        color_name = np.random.choice(list(OBSTACLE_COLORS.keys()))
        color = OBSTACLE_COLORS[color_name]

        # Random shape with weighted selection
        shape_type = np.random.choice(shapes, p=weights)

        prim_path = f"/World/Obstacle_{i}"

        # Create the shape based on type
        if shape_type == 'Cube':
            size = np.random.uniform(0.05, 0.12)
            prim = create_prim(prim_path, "Cube")
            cube = UsdGeom.Cube(prim)
            cube.GetSizeAttr().Set(size * 2)
            z_offset = size

        elif shape_type == 'Cylinder':
            size = np.random.uniform(0.04, 0.10)
            prim = create_prim(prim_path, "Cylinder")
            cylinder = UsdGeom.Cylinder(prim)
            cylinder.GetRadiusAttr().Set(size)
            cylinder.GetHeightAttr().Set(size * 2)
            z_offset = size

        elif shape_type == 'Sphere':
            size = np.random.uniform(0.04, 0.12)
            prim = create_prim(prim_path, "Sphere")
            sphere = UsdGeom.Sphere(prim)
            sphere.GetRadiusAttr().Set(size)
            z_offset = size

        elif shape_type == 'Cone':
            # Traffic cone - tall and narrow
            radius = np.random.uniform(0.03, 0.06)
            height = np.random.uniform(0.15, 0.25)
            prim = create_prim(prim_path, "Cone")
            cone = UsdGeom.Cone(prim)
            cone.GetRadiusAttr().Set(radius)
            cone.GetHeightAttr().Set(height)
            # Cone colors: orange, red, yellow for traffic cones
            color = random.choice([
                OBSTACLE_COLORS['orange'],
                OBSTACLE_COLORS['red'],
                OBSTACLE_COLORS['yellow']
            ])
            z_offset = height / 2

        elif shape_type == 'TallCylinder':
            # Pole or barrier - thin and tall
            radius = np.random.uniform(0.02, 0.04)
            height = np.random.uniform(0.2, 0.35)
            prim = create_prim(prim_path, "Cylinder")
            cylinder = UsdGeom.Cylinder(prim)
            cylinder.GetRadiusAttr().Set(radius)
            cylinder.GetHeightAttr().Set(height)
            z_offset = height / 2

        elif shape_type == 'FlatBox':
            # Low flat box - like a package on ground
            width = np.random.uniform(0.08, 0.15)
            height = np.random.uniform(0.03, 0.06)
            prim = create_prim(prim_path, "Cube")
            cube = UsdGeom.Cube(prim)
            cube.GetSizeAttr().Set(width)
            # Clear existing xform ops and add scale to make it flat
            xform_temp = UsdGeom.Xformable(prim)
            xform_temp.ClearXformOpOrder()
            xform_temp.AddTranslateOp().Set(Gf.Vec3d(x, y, z + height / 2))
            xform_temp.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, height / width))
            gprim = UsdGeom.Gprim(prim)
            gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            obstacle_paths.append(prim_path)
            continue  # Skip default position handling

        elif shape_type == 'TallBox':
            # Chair-like tall box
            width = np.random.uniform(0.06, 0.10)
            height = np.random.uniform(0.15, 0.25)
            prim = create_prim(prim_path, "Cube")
            cube = UsdGeom.Cube(prim)
            cube.GetSizeAttr().Set(width)
            # Clear existing xform ops and add scale to make it tall
            xform_temp = UsdGeom.Xformable(prim)
            xform_temp.ClearXformOpOrder()
            xform_temp.AddTranslateOp().Set(Gf.Vec3d(x, y, z + height / 2))
            xform_temp.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, height / width))
            gprim = UsdGeom.Gprim(prim)
            gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            obstacle_paths.append(prim_path)
            continue  # Skip default position handling

        elif shape_type == 'Capsule':
            # Pill/capsule shape
            radius = np.random.uniform(0.03, 0.06)
            height = np.random.uniform(0.08, 0.15)
            prim = create_prim(prim_path, "Capsule")
            capsule = UsdGeom.Capsule(prim)
            capsule.GetRadiusAttr().Set(radius)
            capsule.GetHeightAttr().Set(height)
            z_offset = radius + height / 2

        elif shape_type == 'Pyramid':
            # Pyramid using cone with wider base
            radius = np.random.uniform(0.06, 0.10)
            height = np.random.uniform(0.08, 0.15)
            prim = create_prim(prim_path, "Cone")
            cone = UsdGeom.Cone(prim)
            cone.GetRadiusAttr().Set(radius)
            cone.GetHeightAttr().Set(height)
            z_offset = height / 2

        elif shape_type == 'OtherJetBot':
            # Spawn another JetBot as obstacle
            if jetbot_asset_path and Path(jetbot_asset_path).exists():
                try:
                    prim = create_prim(prim_path, "Xform")
                    prim.GetReferences().AddReference(jetbot_asset_path)
                    z_offset = 0.0
                    # Random rotation for the other JetBot
                    random_yaw = np.random.uniform(-np.pi, np.pi)
                    xform = UsdGeom.Xformable(prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
                    xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, np.degrees(random_yaw)))
                    obstacle_paths.append(prim_path)
                    continue  # Skip the rest of the loop for JetBot
                except Exception as e:
                    # Fall back to a simple shape if JetBot loading fails
                    size = np.random.uniform(0.05, 0.10)
                    prim = safe_create_prim(prim_path, "Cube")
                    cube = UsdGeom.Cube(prim)
                    cube.GetSizeAttr().Set(size * 2)
                    z_offset = size
                    color = OBSTACLE_COLORS['green']  # JetBot is green
            else:
                # No JetBot asset, use green cube as placeholder
                size = np.random.uniform(0.08, 0.12)
                prim = safe_create_prim(prim_path, "Cube")
                cube = UsdGeom.Cube(prim)
                cube.GetSizeAttr().Set(size * 2)
                z_offset = size
                color = OBSTACLE_COLORS['green']

        elif shape_type == 'NucleusPerson':
            # Spawn a person/human from Nucleus assets
            people_assets = [a for a in nucleus_available if a['category'] == 'people']
            if people_assets:
                asset = np.random.choice(people_assets)
                random_yaw = np.random.uniform(-180, 180)
                # People assets are in cm, scale to meters (0.01)
                # Also make them smaller to fit scene (additional 0.5 scale)
                if spawn_nucleus_asset(stage, asset['path'], prim_path,
                                       (x, y, z), random_yaw, scale=0.005):
                    obstacle_paths.append(prim_path)
                    continue
            # Fallback to tall cylinder (person-like)
            prim = safe_create_prim(prim_path, "Capsule")
            capsule = UsdGeom.Capsule(prim)
            capsule.GetRadiusAttr().Set(0.05)
            capsule.GetHeightAttr().Set(0.3)
            z_offset = 0.2
            color = (0.8, 0.6, 0.5)  # Skin-ish color

        elif shape_type == 'NucleusProp':
            # Spawn furniture/warehouse props from Nucleus
            props_assets = [a for a in nucleus_available if a['category'] == 'props']
            if props_assets:
                asset = np.random.choice(props_assets)
                random_yaw = np.random.uniform(-180, 180)
                # Props are in cm, scale to meters
                if spawn_nucleus_asset(stage, asset['path'], prim_path,
                                       (x, y, z), random_yaw, scale=0.01):
                    obstacle_paths.append(prim_path)
                    continue
            # Fallback to box (furniture-like)
            size = np.random.uniform(0.08, 0.15)
            prim = safe_create_prim(prim_path, "Cube")
            cube = UsdGeom.Cube(prim)
            cube.GetSizeAttr().Set(size * 2)
            z_offset = size
            color = (0.6, 0.4, 0.2)  # Brown/wood color

        elif shape_type == 'NucleusTraffic':
            # Spawn traffic cones/barriers from Nucleus
            traffic_assets = [a for a in nucleus_available if a['category'] == 'traffic']
            if traffic_assets:
                asset = np.random.choice(traffic_assets)
                random_yaw = np.random.uniform(-180, 180)
                if spawn_nucleus_asset(stage, asset['path'], prim_path,
                                       (x, y, z), random_yaw, scale=0.01):
                    obstacle_paths.append(prim_path)
                    continue
            # Fallback to cone
            prim = safe_create_prim(prim_path, "Cone")
            cone = UsdGeom.Cone(prim)
            cone.GetRadiusAttr().Set(0.05)
            cone.GetHeightAttr().Set(0.2)
            z_offset = 0.1
            color = OBSTACLE_COLORS['orange']

        elif shape_type == 'NucleusRobot':
            # Spawn other robots from Nucleus
            robot_assets = [a for a in nucleus_available if a['category'] == 'robots']
            if robot_assets:
                asset = np.random.choice(robot_assets)
                random_yaw = np.random.uniform(-180, 180)
                if spawn_nucleus_asset(stage, asset['path'], prim_path,
                                       (x, y, z), random_yaw, scale=1.0):
                    obstacle_paths.append(prim_path)
                    continue
            # Fallback to cube
            size = np.random.uniform(0.08, 0.12)
            prim = safe_create_prim(prim_path, "Cube")
            cube = UsdGeom.Cube(prim)
            cube.GetSizeAttr().Set(size * 2)
            z_offset = size
            color = OBSTACLE_COLORS['cyan']

        else:
            # Default to cube
            size = np.random.uniform(0.05, 0.12)
            prim = create_prim(prim_path, "Cube")
            cube = UsdGeom.Cube(prim)
            cube.GetSizeAttr().Set(size * 2)
            z_offset = size

        # Set position (skip if already handled by Nucleus assets or OtherJetBot)
        if shape_type not in ['OtherJetBot', 'NucleusPerson', 'NucleusProp', 'NucleusTraffic', 'NucleusRobot', 'FlatBox', 'TallBox']:
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z + z_offset))

            # Set color via displayColor
            gprim = UsdGeom.Gprim(prim)
            gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

        obstacle_paths.append(prim_path)

    return obstacle_paths


def remove_obstacles(stage, obstacle_paths: List[str]):
    """Remove obstacles from the scene."""
    for path in obstacle_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)


def randomize_lighting(stage):
    """Randomize scene lighting for visual diversity."""
    from pxr import UsdLux, Gf, UsdGeom

    # Find or create distant light
    light_path = "/World/DistantLight"
    light_prim = stage.GetPrimAtPath(light_path)

    if not light_prim.IsValid():
        light_prim = stage.DefinePrim(light_path, "DistantLight")

    light = UsdLux.DistantLight(light_prim)

    # Randomize intensity
    intensity = np.random.uniform(500, 2000)
    light.GetIntensityAttr().Set(intensity)

    # Randomize color temperature (warm to cool)
    color_temp = np.random.uniform(4000, 8000)
    # Convert to RGB approximation
    if color_temp < 6500:
        # Warm (yellowish)
        r, g, b = 1.0, 0.9 + 0.1 * (color_temp - 4000) / 2500, 0.8 + 0.2 * (color_temp - 4000) / 2500
    else:
        # Cool (bluish)
        r, g, b = 0.9 - 0.1 * (color_temp - 6500) / 1500, 0.95, 1.0
    light.GetColorAttr().Set(Gf.Vec3f(r, g, b))

    # Randomize angle
    xform = UsdGeom.Xformable(light_prim)
    xform.ClearXformOpOrder()
    angle_x = np.random.uniform(30, 70)
    angle_y = np.random.uniform(-45, 45)
    xform.AddRotateXYZOp().Set(Gf.Vec3d(angle_x, angle_y, 0))


def randomize_ground_color(stage):
    """Randomize ground plane color."""
    from pxr import UsdGeom, Gf

    # Find ground plane
    ground_path = "/World/defaultGroundPlane/GroundPlane/CollisionMesh"
    ground_prim = stage.GetPrimAtPath(ground_path)

    if ground_prim.IsValid():
        color = GROUND_COLORS[np.random.randint(len(GROUND_COLORS))]
        gprim = UsdGeom.Gprim(ground_prim)
        gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])


# Note: UsdGeom is imported inside functions to avoid import errors before Isaac Sim is initialized


def get_scripted_action(instruction: str) -> Tuple[float, float]:
    """Get action based on instruction keywords with noise."""
    instruction = instruction.lower()
    noise = np.random.normal(0, 0.1, 2)

    if 'forward' in instruction or 'straight' in instruction:
        action = (0.5, 0.5)
    elif 'left' in instruction:
        action = (0.2, 0.5)
    elif 'right' in instruction:
        action = (0.5, 0.2)
    elif 'backward' in instruction or 'back' in instruction:
        action = (-0.3, -0.3)
    elif 'stop' in instruction or 'halt' in instruction:
        action = (0.0, 0.0)
    elif 'avoid' in instruction or 'around' in instruction:
        if np.random.random() > 0.5:
            action = (0.3, 0.5)
        else:
            action = (0.5, 0.3)
    else:
        action = (0.3, 0.3)

    return (
        float(np.clip(action[0] + noise[0], -1.0, 1.0)),
        float(np.clip(action[1] + noise[1], -1.0, 1.0))
    )


def test_simulation() -> bool:
    """Test that Isaac Sim and JetBot work correctly."""
    print("Testing Isaac Sim with JetBot...")

    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": True})

        from omni.isaac.core import World
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

        world = World()
        world.scene.add_default_ground_plane()

        jetbot_path = "/workspace/assets/jetbot.usd"
        if not os.path.exists(jetbot_path):
            jetbot_path = download_jetbot_asset()

        jetbot = world.scene.add(
            WheeledRobot(
                prim_path="/World/JetBot",
                name="jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_path,
            )
        )

        controller = DifferentialController(
            name="diff_controller",
            wheel_radius=0.0325,
            wheel_base=0.1
        )

        world.reset()

        # Run a few steps
        for i in range(20):
            wheel_velocities = controller.forward([0.3, 0.1])
            jetbot.apply_wheel_actions(wheel_velocities)
            world.step(render=False)
            if i % 5 == 0:
                pos, _ = jetbot.get_world_pose()
                print(f"Step {i}: Pos=[{pos[0]:.3f}, {pos[1]:.3f}]")

        print("Test PASSED! JetBot simulation working.")
        simulation_app.close()
        return True

    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def collect_synthetic_data(
    output_dir: str = "/workspace/sim_data",
    num_episodes: int = 10,
    steps_per_episode: int = 50,
    instructions: List[str] = None,
    domain_randomization: bool = True,
    num_obstacles: int = 20,
    use_nucleus_assets: bool = True
):
    """
    Collect synthetic training data with domain randomization.

    Args:
        output_dir: Directory to save data
        num_episodes: Number of episodes to collect
        steps_per_episode: Steps per episode
        instructions: List of instructions to use
        domain_randomization: Enable visual diversity (obstacles, lighting, colors)
        num_obstacles: Number of obstacles per episode (5-20)
        use_nucleus_assets: Try to use Nucleus assets (people, furniture, etc.)
    """
    print(f"Collecting synthetic data with domain randomization...")
    print(f"  Episodes: {num_episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Domain randomization: {domain_randomization}")
    print(f"  Obstacles per scene: {num_obstacles}")
    print(f"  Use Nucleus assets: {use_nucleus_assets}")
    print(f"  Output: {output_dir}")

    if instructions is None:
        instructions = DEFAULT_INSTRUCTIONS

    # Create output directory
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": True})

        from omni.isaac.core import World
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.sensor import Camera
        from PIL import Image

        # Create world
        world = World()
        world.scene.add_default_ground_plane()

        # Load JetBot
        jetbot_path = "/workspace/assets/jetbot.usd"
        if not os.path.exists(jetbot_path):
            jetbot_path = download_jetbot_asset()

        jetbot = world.scene.add(
            WheeledRobot(
                prim_path="/World/JetBot",
                name="jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_path,
            )
        )

        controller = DifferentialController(
            name="diff_controller",
            wheel_radius=0.0325,
            wheel_base=0.1
        )

        # Create camera ATTACHED TO ROBOT CHASSIS for first-person view
        # This is CRITICAL - camera must move with the robot
        import omni.usd
        from pxr import UsdGeom, Gf
        from omni.isaac.core.utils.prims import create_prim

        stage = omni.usd.get_context().get_stage()

        # Create camera as child of robot chassis (first-person view)
        camera_prim_path = "/World/JetBot/chassis/front_camera"
        print(f"Creating first-person camera at: {camera_prim_path}")
        create_prim(camera_prim_path, "Camera")

        # Position camera on front of robot, looking forward
        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        xform = UsdGeom.Xformable(camera_prim)
        xform.ClearXformOpOrder()
        # Position: front of robot (x=0.1), centered (y=0), elevated (z=0.06)
        # Camera at front of robot, slightly elevated to see over robot body
        xform.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.0, 0.06))
        # Rotation: USD camera looks down -Z by default, with +Y up
        # Isaac Sim world uses +Z up, +X forward
        # To make camera look forward along +X with +Z up:
        # Use rotation (90, 0, -90) which:
        # - 90 around X: rotates camera so its -Z points toward +Y
        # - -90 around Z: rotates so -Z now points toward +X (forward)
        # This aligns the camera to look forward along world +X axis
        xform.AddRotateXYZOp().Set(Gf.Vec3d(90, 0, -90))

        # Set camera properties for wide-angle navigation view
        camera_geom = UsdGeom.Camera(camera_prim)
        camera_geom.GetFocalLengthAttr().Set(18.0)  # Wide angle
        camera_geom.GetHorizontalApertureAttr().Set(20.955)
        camera_geom.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10.0))

        # IMPORTANT: Reset world BEFORE camera initialization
        world.reset()

        # Create Camera wrapper
        camera = Camera(
            prim_path=camera_prim_path,
            resolution=(224, 224),
            frequency=30
        )
        camera.initialize()

        # Warm-up frames for camera to initialize properly
        print("Warming up camera (first-person view attached to robot)...")
        for _ in range(20):
            world.step(render=True)

        total_samples = 0
        obstacle_paths = []

        for episode in range(num_episodes):
            # Remove previous obstacles
            if obstacle_paths:
                remove_obstacles(stage, obstacle_paths)
                obstacle_paths = []

            # Reset robot position
            world.reset()

            # Default robot pose (origin, facing +X)
            rand_x, rand_y, rand_yaw = 0.0, 0.0, 0.0

            # Randomize robot starting pose for diversity
            if domain_randomization:
                # Random position within a small area
                rand_x = np.random.uniform(-0.3, 0.3)
                rand_y = np.random.uniform(-0.3, 0.3)

                # Random orientation (yaw only, in radians)
                rand_yaw = np.random.uniform(-np.pi, np.pi)

                # Convert yaw to quaternion (w, x, y, z format)
                # For rotation around Z-axis: q = [cos(yaw/2), 0, 0, sin(yaw/2)]
                quat_w = np.cos(rand_yaw / 2)
                quat_z = np.sin(rand_yaw / 2)
                orientation = np.array([quat_w, 0.0, 0.0, quat_z])

                jetbot.set_world_pose(
                    position=np.array([rand_x, rand_y, 0.05]),
                    orientation=orientation
                )

            # Domain randomization for this episode
            if domain_randomization:
                # Add random obstacles (always at least 5)
                # Position them in front of the robot based on its pose
                actual_num_obstacles = np.random.randint(5, num_obstacles + 1)
                obstacle_paths = create_random_obstacles(
                    stage,
                    actual_num_obstacles,
                    robot_pos=(rand_x, rand_y),
                    robot_yaw=rand_yaw,
                    jetbot_asset_path=jetbot_path,
                    use_nucleus_assets=use_nucleus_assets
                )

                # Randomize lighting
                try:
                    randomize_lighting(stage)
                except Exception as e:
                    pass  # Lighting randomization is optional

                # Randomize ground color
                try:
                    randomize_ground_color(stage)
                except Exception as e:
                    pass  # Ground color randomization is optional

            # Let simulation settle after changes
            for _ in range(10):
                world.step(render=True)

            # Select random instruction for this episode
            instruction = np.random.choice(instructions)

            # Track obstacle info for metadata
            scene_info = {
                'num_obstacles': len(obstacle_paths),
                'has_obstacles': len(obstacle_paths) > 0
            }

            for step in range(steps_per_episode):
                # Get camera image
                world.step(render=True)
                rgba = camera.get_rgba()

                if rgba is None:
                    print(f"Warning: No camera image at episode {episode}, step {step}")
                    continue

                rgb = rgba[:, :, :3]

                # Get action from scripted policy
                left_speed, right_speed = get_scripted_action(instruction)

                # Save sample
                sample_id = str(uuid.uuid4())

                # Save image
                img_pil = Image.fromarray(rgb.astype(np.uint8))
                img_pil.save(save_path / f"{sample_id}.jpg", quality=95)

                # Save metadata (including scene info)
                metadata = {
                    'instruction': instruction,
                    'action': {
                        'left_speed': left_speed,
                        'right_speed': right_speed
                    },
                    'episode': episode,
                    'step': step,
                    'timestamp': time.time(),
                    'source': 'isaac_sim',
                    'domain_randomization': domain_randomization,
                    'scene': scene_info
                }
                with open(save_path / f"{sample_id}.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Apply action
                linear = (left_speed + right_speed) / 2.0 * 0.3
                angular = (right_speed - left_speed) / 0.1 * 0.3
                wheel_velocities = controller.forward([linear, angular])
                jetbot.apply_wheel_actions(wheel_velocities)

                total_samples += 1

            print(f"Episode {episode+1}/{num_episodes} complete "
                  f"({total_samples} samples, {len(obstacle_paths)} obstacles)")

        # Cleanup
        if obstacle_paths:
            remove_obstacles(stage, obstacle_paths)

        print(f"\nData collection complete: {total_samples} samples saved to {save_path}")
        simulation_app.close()

    except Exception as e:
        print(f"Data collection failed: {e}")
        import traceback
        traceback.print_exc()


def test_vla_in_simulation(
    vla_host: str = 'localhost',
    vla_port: int = 5555,
    instruction: str = 'go forward',
    max_steps: int = 100
):
    """Test VLA model in Isaac Sim simulation."""
    import zmq
    import io
    from PIL import Image

    print(f"Testing VLA in simulation...")
    print(f"  VLA server: {vla_host}:{vla_port}")
    print(f"  Instruction: {instruction}")
    print(f"  Max steps: {max_steps}")

    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": True})

        from omni.isaac.core import World
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.sensor import Camera

        # Create world
        world = World()
        world.scene.add_default_ground_plane()

        # Load JetBot
        jetbot_path = "/workspace/assets/jetbot.usd"
        if not os.path.exists(jetbot_path):
            jetbot_path = download_jetbot_asset()

        jetbot = world.scene.add(
            WheeledRobot(
                prim_path="/World/JetBot",
                name="jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_path,
            )
        )

        controller = DifferentialController(
            name="diff_controller",
            wheel_radius=0.0325,
            wheel_base=0.1
        )

        # Find or create camera
        import omni.usd
        from pxr import UsdGeom, Gf
        from omni.isaac.core.utils.prims import create_prim

        stage = omni.usd.get_context().get_stage()
        camera_prim_path = None
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                camera_prim_path = str(prim.GetPath())
                break

        if camera_prim_path is None:
            camera_prim_path = "/World/Camera"
            create_prim(camera_prim_path, "Camera")
            camera_prim = stage.GetPrimAtPath(camera_prim_path)
            xform = UsdGeom.Xformable(camera_prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.3))
            xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 30, 180))

        world.reset()

        camera = Camera(
            prim_path=camera_prim_path,
            resolution=(224, 224),
            frequency=30
        )
        camera.initialize()

        # Add colored obstacles for complex instruction testing
        print("Adding colored obstacles to scene...")
        from pxr import UsdShade, Sdf

        # Create a red sphere (ball) in front of the robot
        red_ball_path = "/World/RedBall"
        create_prim(red_ball_path, "Sphere")
        red_ball = stage.GetPrimAtPath(red_ball_path)
        sphere = UsdGeom.Sphere(red_ball)
        sphere.GetRadiusAttr().Set(0.05)
        xform = UsdGeom.Xformable(red_ball)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.2, 0.05))  # In front and slightly right
        gprim = UsdGeom.Gprim(red_ball)
        gprim.GetDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.1, 0.1)])  # Red

        # Create a blue cube
        blue_cube_path = "/World/BlueCube"
        create_prim(blue_cube_path, "Cube")
        blue_cube = stage.GetPrimAtPath(blue_cube_path)
        cube = UsdGeom.Cube(blue_cube)
        cube.GetSizeAttr().Set(0.08)
        xform = UsdGeom.Xformable(blue_cube)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.4, -0.3, 0.04))  # In front and left
        gprim = UsdGeom.Gprim(blue_cube)
        gprim.GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.1, 0.9)])  # Blue

        # Create a green cylinder
        green_cyl_path = "/World/GreenCylinder"
        create_prim(green_cyl_path, "Cylinder")
        green_cyl = stage.GetPrimAtPath(green_cyl_path)
        cyl = UsdGeom.Cylinder(green_cyl)
        cyl.GetRadiusAttr().Set(0.04)
        cyl.GetHeightAttr().Set(0.1)
        xform = UsdGeom.Xformable(green_cyl)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.7, 0.0, 0.05))  # Farther ahead
        gprim = UsdGeom.Gprim(green_cyl)
        gprim.GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.8, 0.1)])  # Green

        print("  Added: Red ball at (0.5, 0.2), Blue cube at (0.4, -0.3), Green cylinder at (0.7, 0.0)")

        # Warm-up
        for _ in range(20):
            world.step(render=True)

        # Connect to VLA server
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{vla_host}:{vla_port}")
        print(f"Connected to VLA server at {vla_host}:{vla_port}")

        # Run VLA-guided navigation
        positions = []
        actions = []

        for step in range(max_steps):
            world.step(render=True)
            rgba = camera.get_rgba()

            if rgba is None:
                print(f"Warning: No camera image at step {step}")
                continue

            rgb = rgba[:, :, :3]

            # Convert to PNG for sending (JPEG has PIL version issues in Isaac Sim)
            img_pil = Image.fromarray(rgb.astype(np.uint8))
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

            # Send to VLA server
            socket.send_multipart([image_bytes, instruction.encode('utf-8')])

            # Receive response
            response = socket.recv_multipart()
            if response[0] == b'ERROR':
                print(f"VLA error: {response[1].decode('utf-8')}")
                break

            left_speed = float(response[0].decode('utf-8'))
            right_speed = float(response[1].decode('utf-8'))

            # Apply action
            linear = (left_speed + right_speed) / 2.0 * 0.3
            angular = (right_speed - left_speed) / 0.1 * 0.3
            wheel_velocities = controller.forward([linear, angular])
            jetbot.apply_wheel_actions(wheel_velocities)

            # Record
            pos, _ = jetbot.get_world_pose()
            positions.append(pos.tolist())
            actions.append((left_speed, right_speed))

            if step % 10 == 0:
                print(f"Step {step}: pos=[{pos[0]:.3f}, {pos[1]:.3f}], "
                      f"action=[{left_speed:.3f}, {right_speed:.3f}]")

        print(f"\nVLA test complete!")
        print(f"  Total steps: {len(positions)}")
        if positions:
            start = positions[0]
            end = positions[-1]
            distance = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            print(f"  Distance traveled: {distance:.3f}m")

        socket.close()
        context.term()
        simulation_app.close()

    except Exception as e:
        print(f"VLA test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='RunPod Setup for JetBot VLA Simulation (Self-contained)'
    )
    parser.add_argument(
        '--download-assets',
        action='store_true',
        help='Download JetBot USD asset'
    )
    parser.add_argument(
        '--test-sim',
        action='store_true',
        help='Test Isaac Sim with JetBot'
    )
    parser.add_argument(
        '--collect-data',
        action='store_true',
        help='Collect synthetic training data'
    )
    parser.add_argument(
        '--test-vla',
        action='store_true',
        help='Test VLA model in simulation'
    )
    parser.add_argument(
        '--vla-host',
        default='localhost',
        help='VLA server host'
    )
    parser.add_argument(
        '--vla-port',
        type=int,
        default=5555,
        help='VLA server port'
    )
    parser.add_argument(
        '--instruction',
        default='go forward',
        help='Navigation instruction for VLA test'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of episodes for data collection'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Steps per episode'
    )
    parser.add_argument(
        '--output',
        default='/workspace/sim_data',
        help='Output directory for data'
    )
    parser.add_argument(
        '--no-domain-randomization',
        action='store_true',
        help='Disable domain randomization (obstacles, lighting, colors)'
    )
    parser.add_argument(
        '--obstacles',
        type=int,
        default=20,
        help='Max number of obstacles per scene (5-20 will be randomly selected)'
    )
    parser.add_argument(
        '--no-nucleus-assets',
        action='store_true',
        help='Disable Nucleus assets (people, furniture, warehouse props)'
    )

    args = parser.parse_args()

    if args.download_assets:
        download_jetbot_asset()

    if args.test_sim:
        success = test_simulation()
        sys.exit(0 if success else 1)

    if args.collect_data:
        collect_synthetic_data(
            output_dir=args.output,
            num_episodes=args.episodes,
            steps_per_episode=args.steps,
            domain_randomization=not args.no_domain_randomization,
            num_obstacles=args.obstacles,
            use_nucleus_assets=not args.no_nucleus_assets
        )

    if args.test_vla:
        test_vla_in_simulation(
            vla_host=args.vla_host,
            vla_port=args.vla_port,
            instruction=args.instruction,
            max_steps=args.steps
        )

    if not any([args.download_assets, args.test_sim, args.collect_data, args.test_vla]):
        parser.print_help()


if __name__ == '__main__':
    main()
