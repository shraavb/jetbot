"""
Experimental Scene Setup for VLA Language Grounding

Creates a configurable scene with colored objects for testing VLA model
responses to natural language commands like:
- "go towards the red ball"
- "follow the person in front of you"
- "go around the chair"
- "go towards the door"

Usage:
    from jetbot.isaac_sim.experiment_scene import ExperimentScene

    scene = ExperimentScene()
    scene.setup()
    scene.add_red_ball(position=[2.0, 0.0, 0.15])
    scene.add_chair(position=[1.5, 1.0, 0.0])
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class ObjectColor(Enum):
    """Standard colors for objects in the scene."""
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0)
    ORANGE = (1.0, 0.5, 0.0)
    PURPLE = (0.5, 0.0, 0.5)
    WHITE = (1.0, 1.0, 1.0)
    BLACK = (0.1, 0.1, 0.1)
    BROWN = (0.6, 0.3, 0.1)
    GRAY = (0.5, 0.5, 0.5)


@dataclass
class SceneObject:
    """Represents an object in the experiment scene."""
    name: str
    prim_path: str
    position: np.ndarray
    object_type: str  # 'ball', 'chair', 'door', 'person', 'box', etc.
    color: Optional[ObjectColor] = None
    scale: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    usd_path: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'prim_path': self.prim_path,
            'position': self.position.tolist(),
            'object_type': self.object_type,
            'color': self.color.name if self.color else None,
            'scale': self.scale.tolist()
        }


@dataclass
class SceneConfig:
    """Configuration for an experiment scene."""
    name: str
    description: str
    objects: List[Dict] = field(default_factory=list)
    robot_start_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    robot_start_orientation: float = 0.0  # Yaw angle in radians
    ground_size: Tuple[float, float] = (10.0, 10.0)
    ground_color: Tuple[float, float, float] = (0.4, 0.4, 0.4)


class ExperimentScene:
    """
    Creates and manages experiment scenes with colored objects for VLA testing.

    The scene includes a ground plane and various objects that can be referenced
    by color and type in natural language commands.
    """

    def __init__(self):
        """Initialize experiment scene manager."""
        self._world = None
        self._objects: Dict[str, SceneObject] = {}
        self._ground = None
        self._is_initialized = False

        # Isaac Sim modules (lazy loaded)
        self._DynamicSphere = None
        self._DynamicCuboid = None
        self._VisualCuboid = None
        self._VisualCylinder = None
        self._XFormPrim = None

    def _lazy_import_isaac(self):
        """Lazy import Isaac Sim modules."""
        if self._DynamicSphere is None:
            try:
                from omni.isaac.core.objects import (
                    DynamicSphere,
                    DynamicCuboid,
                    VisualCuboid,
                    VisualCylinder
                )
                from omni.isaac.core.prims import XFormPrim

                self._DynamicSphere = DynamicSphere
                self._DynamicCuboid = DynamicCuboid
                self._VisualCuboid = VisualCuboid
                self._VisualCylinder = VisualCylinder
                self._XFormPrim = XFormPrim
            except ImportError as e:
                raise ImportError(
                    f"Isaac Sim not available. Please ensure Isaac Sim is installed. Error: {e}"
                )

    def setup(
        self,
        world,
        ground_size: Tuple[float, float] = (10.0, 10.0),
        ground_color: Tuple[float, float, float] = (0.4, 0.4, 0.4)
    ) -> None:
        """
        Setup the experiment scene with a ground plane.

        Args:
            world: Isaac Sim World instance
            ground_size: Size of ground plane (width, depth)
            ground_color: RGB color of ground
        """
        self._lazy_import_isaac()
        self._world = world

        # Create ground plane
        self._ground = self._world.scene.add(
            self._VisualCuboid(
                prim_path="/World/Experiment/ground",
                name="ground",
                position=np.array([0.0, 0.0, -0.01]),
                scale=np.array([ground_size[0], ground_size[1], 0.02]),
                color=np.array(ground_color)
            )
        )

        self._is_initialized = True

    def add_ball(
        self,
        color: ObjectColor,
        position: List[float],
        radius: float = 0.15,
        name: Optional[str] = None
    ) -> SceneObject:
        """
        Add a colored ball to the scene.

        Args:
            color: Ball color
            position: [x, y, z] position
            radius: Ball radius in meters
            name: Optional custom name (defaults to "{color}_ball")
        """
        self._lazy_import_isaac()

        if name is None:
            name = f"{color.name.lower()}_ball"

        # Ensure unique name
        base_name = name
        counter = 1
        while name in self._objects:
            name = f"{base_name}_{counter}"
            counter += 1

        prim_path = f"/World/Experiment/{name}"

        ball = self._world.scene.add(
            self._DynamicSphere(
                prim_path=prim_path,
                name=name,
                position=np.array(position),
                radius=radius,
                color=np.array(color.value),
                mass=0.1
            )
        )

        obj = SceneObject(
            name=name,
            prim_path=prim_path,
            position=np.array(position),
            object_type='ball',
            color=color,
            scale=np.array([radius, radius, radius])
        )
        self._objects[name] = obj

        return obj

    def add_red_ball(self, position: List[float], radius: float = 0.15) -> SceneObject:
        """Add a red ball at the specified position."""
        return self.add_ball(ObjectColor.RED, position, radius)

    def add_blue_ball(self, position: List[float], radius: float = 0.15) -> SceneObject:
        """Add a blue ball at the specified position."""
        return self.add_ball(ObjectColor.BLUE, position, radius)

    def add_green_ball(self, position: List[float], radius: float = 0.15) -> SceneObject:
        """Add a green ball at the specified position."""
        return self.add_ball(ObjectColor.GREEN, position, radius)

    def add_box(
        self,
        color: ObjectColor,
        position: List[float],
        size: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        name: Optional[str] = None
    ) -> SceneObject:
        """
        Add a colored box to the scene.

        Args:
            color: Box color
            position: [x, y, z] position
            size: (width, depth, height) in meters
            name: Optional custom name
        """
        self._lazy_import_isaac()

        if name is None:
            name = f"{color.name.lower()}_box"

        # Ensure unique name
        base_name = name
        counter = 1
        while name in self._objects:
            name = f"{base_name}_{counter}"
            counter += 1

        prim_path = f"/World/Experiment/{name}"

        box = self._world.scene.add(
            self._DynamicCuboid(
                prim_path=prim_path,
                name=name,
                position=np.array(position),
                scale=np.array(size),
                color=np.array(color.value),
                mass=0.5
            )
        )

        obj = SceneObject(
            name=name,
            prim_path=prim_path,
            position=np.array(position),
            object_type='box',
            color=color,
            scale=np.array(size)
        )
        self._objects[name] = obj

        return obj

    def add_chair(
        self,
        position: List[float],
        color: ObjectColor = ObjectColor.BROWN,
        name: str = "chair"
    ) -> SceneObject:
        """
        Add a simple chair representation (box-based) to the scene.

        Args:
            position: [x, y, z] position
            color: Chair color
            name: Object name
        """
        self._lazy_import_isaac()

        # Ensure unique name
        base_name = name
        counter = 1
        while name in self._objects:
            name = f"{base_name}_{counter}"
            counter += 1

        prim_path = f"/World/Experiment/{name}"

        # Create chair as a simple box (seat)
        # In a real setup, you'd load a chair USD asset
        seat_height = 0.45
        seat_size = (0.4, 0.4, 0.05)

        chair = self._world.scene.add(
            self._VisualCuboid(
                prim_path=prim_path,
                name=name,
                position=np.array([position[0], position[1], seat_height]),
                scale=np.array(seat_size),
                color=np.array(color.value)
            )
        )

        # Add backrest
        backrest_path = f"{prim_path}_back"
        self._world.scene.add(
            self._VisualCuboid(
                prim_path=backrest_path,
                name=f"{name}_back",
                position=np.array([position[0] - 0.175, position[1], seat_height + 0.25]),
                scale=np.array([0.05, 0.4, 0.5]),
                color=np.array(color.value)
            )
        )

        # Add legs
        leg_positions = [
            [position[0] - 0.15, position[1] - 0.15, seat_height / 2],
            [position[0] - 0.15, position[1] + 0.15, seat_height / 2],
            [position[0] + 0.15, position[1] - 0.15, seat_height / 2],
            [position[0] + 0.15, position[1] + 0.15, seat_height / 2],
        ]

        for i, leg_pos in enumerate(leg_positions):
            leg_path = f"{prim_path}_leg{i}"
            self._world.scene.add(
                self._VisualCuboid(
                    prim_path=leg_path,
                    name=f"{name}_leg{i}",
                    position=np.array(leg_pos),
                    scale=np.array([0.04, 0.04, seat_height]),
                    color=np.array(color.value)
                )
            )

        obj = SceneObject(
            name=name,
            prim_path=prim_path,
            position=np.array(position),
            object_type='chair',
            color=color,
            scale=np.array([0.4, 0.4, 0.9])
        )
        self._objects[name] = obj

        return obj

    def add_door(
        self,
        position: List[float],
        color: ObjectColor = ObjectColor.BROWN,
        width: float = 0.9,
        height: float = 2.0,
        name: str = "door"
    ) -> SceneObject:
        """
        Add a door representation to the scene.

        Args:
            position: [x, y, z] position (bottom center of door)
            color: Door color
            width: Door width in meters
            height: Door height in meters
            name: Object name
        """
        self._lazy_import_isaac()

        # Ensure unique name
        base_name = name
        counter = 1
        while name in self._objects:
            name = f"{base_name}_{counter}"
            counter += 1

        prim_path = f"/World/Experiment/{name}"

        # Create door as a thin cuboid
        door = self._world.scene.add(
            self._VisualCuboid(
                prim_path=prim_path,
                name=name,
                position=np.array([position[0], position[1], height / 2]),
                scale=np.array([0.05, width, height]),
                color=np.array(color.value)
            )
        )

        # Add door frame
        frame_color = np.array([0.3, 0.3, 0.3])
        frame_thickness = 0.08

        # Top frame
        self._world.scene.add(
            self._VisualCuboid(
                prim_path=f"{prim_path}_frame_top",
                name=f"{name}_frame_top",
                position=np.array([position[0], position[1], height + frame_thickness / 2]),
                scale=np.array([0.1, width + 0.2, frame_thickness]),
                color=frame_color
            )
        )

        # Side frames
        for side, y_offset in [('left', -width/2 - 0.05), ('right', width/2 + 0.05)]:
            self._world.scene.add(
                self._VisualCuboid(
                    prim_path=f"{prim_path}_frame_{side}",
                    name=f"{name}_frame_{side}",
                    position=np.array([position[0], position[1] + y_offset, height / 2]),
                    scale=np.array([0.1, 0.1, height]),
                    color=frame_color
                )
            )

        obj = SceneObject(
            name=name,
            prim_path=prim_path,
            position=np.array(position),
            object_type='door',
            color=color,
            scale=np.array([0.05, width, height])
        )
        self._objects[name] = obj

        return obj

    def add_person(
        self,
        position: List[float],
        color: ObjectColor = ObjectColor.BLUE,
        name: str = "person"
    ) -> SceneObject:
        """
        Add a simple person representation (cylinder-based) to the scene.

        Args:
            position: [x, y, z] position
            color: Clothing color
            name: Object name
        """
        self._lazy_import_isaac()

        # Ensure unique name
        base_name = name
        counter = 1
        while name in self._objects:
            name = f"{base_name}_{counter}"
            counter += 1

        prim_path = f"/World/Experiment/{name}"

        # Body (cylinder)
        body_height = 1.4
        body_radius = 0.2

        body = self._world.scene.add(
            self._VisualCylinder(
                prim_path=prim_path,
                name=name,
                position=np.array([position[0], position[1], body_height / 2]),
                radius=body_radius,
                height=body_height,
                color=np.array(color.value)
            )
        )

        # Head (sphere approximated with small cylinder)
        head_radius = 0.12
        self._world.scene.add(
            self._VisualCylinder(
                prim_path=f"{prim_path}_head",
                name=f"{name}_head",
                position=np.array([position[0], position[1], body_height + head_radius]),
                radius=head_radius,
                height=head_radius * 2,
                color=np.array([0.9, 0.7, 0.6])  # Skin tone
            )
        )

        obj = SceneObject(
            name=name,
            prim_path=prim_path,
            position=np.array(position),
            object_type='person',
            color=color,
            scale=np.array([body_radius * 2, body_radius * 2, body_height + head_radius * 2])
        )
        self._objects[name] = obj

        return obj

    def add_cone(
        self,
        color: ObjectColor,
        position: List[float],
        radius: float = 0.15,
        height: float = 0.3,
        name: Optional[str] = None
    ) -> SceneObject:
        """
        Add a colored cone/traffic cone to the scene.

        Args:
            color: Cone color
            position: [x, y, z] position
            radius: Base radius in meters
            height: Cone height in meters
            name: Optional custom name
        """
        self._lazy_import_isaac()

        if name is None:
            name = f"{color.name.lower()}_cone"

        # Ensure unique name
        base_name = name
        counter = 1
        while name in self._objects:
            name = f"{base_name}_{counter}"
            counter += 1

        prim_path = f"/World/Experiment/{name}"

        # Approximate cone with cylinder (cone prim may not be available)
        cone = self._world.scene.add(
            self._VisualCylinder(
                prim_path=prim_path,
                name=name,
                position=np.array([position[0], position[1], height / 2]),
                radius=radius,
                height=height,
                color=np.array(color.value)
            )
        )

        obj = SceneObject(
            name=name,
            prim_path=prim_path,
            position=np.array(position),
            object_type='cone',
            color=color,
            scale=np.array([radius, radius, height])
        )
        self._objects[name] = obj

        return obj

    def add_wall(
        self,
        position: List[float],
        size: Tuple[float, float, float] = (0.1, 2.0, 1.0),
        color: ObjectColor = ObjectColor.GRAY,
        name: str = "wall"
    ) -> SceneObject:
        """
        Add a wall segment to the scene.

        Args:
            position: [x, y, z] position (center)
            size: (thickness, length, height) in meters
            color: Wall color
            name: Object name
        """
        self._lazy_import_isaac()

        # Ensure unique name
        base_name = name
        counter = 1
        while name in self._objects:
            name = f"{base_name}_{counter}"
            counter += 1

        prim_path = f"/World/Experiment/{name}"

        wall = self._world.scene.add(
            self._VisualCuboid(
                prim_path=prim_path,
                name=name,
                position=np.array([position[0], position[1], size[2] / 2]),
                scale=np.array(size),
                color=np.array(color.value)
            )
        )

        obj = SceneObject(
            name=name,
            prim_path=prim_path,
            position=np.array(position),
            object_type='wall',
            color=color,
            scale=np.array(size)
        )
        self._objects[name] = obj

        return obj

    def get_object(self, name: str) -> Optional[SceneObject]:
        """Get object by name."""
        return self._objects.get(name)

    def get_objects_by_type(self, object_type: str) -> List[SceneObject]:
        """Get all objects of a specific type."""
        return [obj for obj in self._objects.values() if obj.object_type == object_type]

    def get_objects_by_color(self, color: ObjectColor) -> List[SceneObject]:
        """Get all objects of a specific color."""
        return [obj for obj in self._objects.values() if obj.color == color]

    def get_all_objects(self) -> Dict[str, SceneObject]:
        """Get all objects in the scene."""
        return self._objects.copy()

    def remove_object(self, name: str) -> bool:
        """Remove an object from the scene."""
        if name in self._objects:
            # Would need to remove from world scene as well
            del self._objects[name]
            return True
        return False

    def clear_objects(self) -> None:
        """Remove all objects from the scene (except ground)."""
        self._objects.clear()

    def get_scene_description(self) -> str:
        """Generate a natural language description of the scene."""
        if not self._objects:
            return "The scene is empty."

        descriptions = []
        for obj in self._objects.values():
            color_str = obj.color.name.lower() if obj.color else ""
            pos_str = f"at position ({obj.position[0]:.1f}, {obj.position[1]:.1f})"

            if color_str:
                descriptions.append(f"a {color_str} {obj.object_type} {pos_str}")
            else:
                descriptions.append(f"a {obj.object_type} {pos_str}")

        return "The scene contains: " + ", ".join(descriptions) + "."

    def export_config(self) -> SceneConfig:
        """Export current scene configuration."""
        return SceneConfig(
            name="experiment_scene",
            description=self.get_scene_description(),
            objects=[obj.to_dict() for obj in self._objects.values()]
        )


# Predefined scene configurations
SCENE_CONFIGS = {
    'color_navigation': SceneConfig(
        name='color_navigation',
        description='Scene with colored objects for testing color-based navigation commands',
        objects=[
            {'type': 'ball', 'color': 'RED', 'position': [2.0, 0.0, 0.15]},
            {'type': 'ball', 'color': 'BLUE', 'position': [0.0, 2.0, 0.15]},
            {'type': 'ball', 'color': 'GREEN', 'position': [-2.0, 0.0, 0.15]},
            {'type': 'box', 'color': 'YELLOW', 'position': [0.0, -2.0, 0.1]},
        ],
        robot_start_position=np.array([0.0, 0.0, 0.0])
    ),

    'obstacle_course': SceneConfig(
        name='obstacle_course',
        description='Scene with chairs and obstacles for avoidance testing',
        objects=[
            {'type': 'chair', 'position': [1.5, 0.5, 0.0]},
            {'type': 'chair', 'position': [1.5, -0.5, 0.0]},
            {'type': 'box', 'color': 'RED', 'position': [3.0, 0.0, 0.1]},
        ],
        robot_start_position=np.array([0.0, 0.0, 0.0])
    ),

    'room_navigation': SceneConfig(
        name='room_navigation',
        description='Room-like scene with door and furniture',
        objects=[
            {'type': 'door', 'position': [3.0, 0.0, 0.0]},
            {'type': 'chair', 'position': [1.0, 1.5, 0.0]},
            {'type': 'person', 'position': [2.0, -1.0, 0.0]},
            {'type': 'ball', 'color': 'RED', 'position': [1.5, 0.0, 0.15]},
        ],
        robot_start_position=np.array([0.0, 0.0, 0.0])
    ),

    'multi_target': SceneConfig(
        name='multi_target',
        description='Multiple targets of different colors and types',
        objects=[
            {'type': 'ball', 'color': 'RED', 'position': [2.0, 1.0, 0.15]},
            {'type': 'ball', 'color': 'BLUE', 'position': [2.0, -1.0, 0.15]},
            {'type': 'box', 'color': 'GREEN', 'position': [-1.5, 1.5, 0.1]},
            {'type': 'box', 'color': 'YELLOW', 'position': [-1.5, -1.5, 0.1]},
            {'type': 'cone', 'color': 'ORANGE', 'position': [0.0, 2.5, 0.0]},
            {'type': 'person', 'position': [1.0, 0.0, 0.0]},
            {'type': 'chair', 'position': [-2.0, 0.0, 0.0]},
            {'type': 'door', 'position': [4.0, 0.0, 0.0]},
        ],
        robot_start_position=np.array([0.0, 0.0, 0.0])
    ),
}


def load_scene_from_config(
    experiment_scene: ExperimentScene,
    config: SceneConfig
) -> None:
    """
    Load objects from a scene configuration.

    Args:
        experiment_scene: ExperimentScene instance (already setup with world)
        config: Scene configuration to load
    """
    for obj_config in config.objects:
        obj_type = obj_config['type']
        position = obj_config['position']
        color = ObjectColor[obj_config['color']] if 'color' in obj_config else None
        name = obj_config.get('name')

        if obj_type == 'ball':
            experiment_scene.add_ball(color, position, name=name)
        elif obj_type == 'box':
            experiment_scene.add_box(color, position, name=name)
        elif obj_type == 'chair':
            experiment_scene.add_chair(position, color=color or ObjectColor.BROWN, name=name or 'chair')
        elif obj_type == 'door':
            experiment_scene.add_door(position, color=color or ObjectColor.BROWN, name=name or 'door')
        elif obj_type == 'person':
            experiment_scene.add_person(position, color=color or ObjectColor.BLUE, name=name or 'person')
        elif obj_type == 'cone':
            experiment_scene.add_cone(color, position, name=name)
        elif obj_type == 'wall':
            size = obj_config.get('size', (0.1, 2.0, 1.0))
            experiment_scene.add_wall(position, size, color or ObjectColor.GRAY, name=name or 'wall')
