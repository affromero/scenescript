# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .base_entity import BaseEntity
from .wall import WallEntity


class DoorEntity(BaseEntity):
    COMMAND_STRING = "make_door"

    PARAMS_DEFINITION = {
        "id": {"dtype": int, "normalise": None},
        "wall0_id": {"dtype": int, "normalise": None},
        "wall1_id": {"dtype": int, "normalise": None},
        "position_x": {"dtype": float, "normalise": "world"},
        "position_y": {"dtype": float, "normalise": "world"},
        "position_z": {"dtype": float, "normalise": "world"},
        "width": {"dtype": float, "normalise": "width"},
        "height": {"dtype": float, "normalise": "height"},
    }

    TOKEN = 2

    def __init__(self, parameters: dict[str, torch.Tensor], parent_wall_entity: WallEntity | None = None) -> None:
        """
        Args:
            parameters: Dict with keys specified in DoorEntity.PARAMS_DEFINITION.
            parent_wall_entity: Optional[WallEntity instance].
        """
        re_ordered_parameters = {}
        for param_key in DoorEntity.PARAMS_DEFINITION:
            if param_key in parameters:
                re_ordered_parameters[param_key] = parameters[param_key]
        self.params = re_ordered_parameters

        self.assign_parent_wall_entity(parent_wall_entity)

    def assign_parent_wall_entity(self, parent_wall_entity: WallEntity | None = None) -> None:
        """
        Args:
            parent_wall_entity: WallEntity instance.
        """
        self.parent_wall_entity = parent_wall_entity
        if parent_wall_entity is not None:
            # Write to wall0_id and wall1_id. Legacy parameters
            self.params["wall0_id"] = parent_wall_entity.params["id"]
            self.params["wall1_id"] = parent_wall_entity.params["id"]

    def extent(self) -> dict[str, float]:
        """Compute extent of door entity.

        Returns:
            Dict with the following keys: {min/max/size}_{x/y/z}.
                Values are floats.
        """
        if self.parent_wall_entity is None:
            raise ValueError("Parent wall entity is not assigned.")
        wall_start = np.array(
            [
                self.parent_wall_entity.params["a_x"],
                self.parent_wall_entity.params["a_y"],
            ]
        )
        wall_end = np.array(
            [
                self.parent_wall_entity.params["b_x"],
                self.parent_wall_entity.params["b_y"],
            ]
        )
        wall_length = np.linalg.norm(wall_end - wall_start)
        wall_xy_unit_vec = (wall_end - wall_start) / wall_length  # [2]
        wall_xy_unit_vec = np.nan_to_num(wall_xy_unit_vec, nan=0)

        door_center = np.array(
            [
                self.params["position_x"],
                self.params["position_y"],
                self.params["position_z"],
            ]
        )  # [3]
        offset = 0.5 * np.concatenate(
            [wall_xy_unit_vec * self.params["width"], np.array([self.params["height"]])]
        )  # [3]
        door_start_xyz = door_center - offset  # [3]
        door_end_xyz = door_center + offset  # [3]

        min_x = min(door_start_xyz[0], door_end_xyz[0])
        max_x = max(door_start_xyz[0], door_end_xyz[0])
        min_y = min(door_start_xyz[1], door_end_xyz[1])
        max_y = max(door_start_xyz[1], door_end_xyz[1])
        min_z = min(door_start_xyz[2], door_end_xyz[2])
        max_z = max(door_start_xyz[2], door_end_xyz[2])

        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "min_z": min_z,
            "max_z": max_z,
            "size_x": max(max_x - min_x, 0),
            "size_y": max(max_y - min_y, 0),
            "size_z": max(max_z - min_z, 0),
        }

    def rotate(self, rotation_angle: float) -> None:
        """Rotate door entity.

        Args:
            rotation_angle: float. Angle to rotate in degrees about the Z-axis.
        """
        door_center = np.array(
            [
                self.params["position_x"],
                self.params["position_y"],
                self.params["position_z"],
            ]
        )  # [3]

        rot_mat = Rotation.from_euler(
            "ZYX", [rotation_angle, 0, 0], degrees=True
        ).as_matrix()
        new_door_center = rot_mat @ door_center

        self.params["position_x"] = new_door_center[0]
        self.params["position_y"] = new_door_center[1]
        self.params["position_z"] = new_door_center[2]

    def translate(self, translation: torch.Tensor) -> None:
        """Translate door entity.

        Args:
            translation: [3] torch.Tensor of XYZ translation.
        """
        door_center = torch.as_tensor(
            [
                self.params["position_x"],
                self.params["position_y"],
                self.params["position_z"],
            ],
        ).float()  # [3]
        new_door_center = door_center + translation

        self.params["position_x"] = float(new_door_center[0])
        self.params["position_y"] = float(new_door_center[1])
        self.params["position_z"] = float(new_door_center[2])

    def lex_sort_key(self) -> np.ndarray:
        """Compute sorting key for lexicographic sorting.

        Returns:
            a [3] np.ndarray.
        """
        door_center = np.array(
            [
                self.params["position_x"],
                self.params["position_y"],
                self.params["position_z"],
            ]
        )  # [3]
        return door_center

    def random_sort_key(self) -> np.ndarray:
        """Compute sorting key for random sorting.

        Returns:
            a [1] np.ndarray.
        """
        return np.random.rand(1)  # [1]

    def to_seq_value(self) -> list[Any]:
        return [
            self.TOKEN,
            self.params["wall0_id"],
            self.params["wall1_id"],
            self.params["position_x"],
            self.params["position_y"],
            self.params["position_z"],
            self.params["width"],
            self.params["height"],
        ]
