# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypeAlias

from .bbox import BboxEntity
from .door import DoorEntity
from .wall import WallEntity
from .window import WindowEntity

ALL_ENTITY_CLASSES_TYPE: TypeAlias = list[WallEntity | DoorEntity | WindowEntity | BboxEntity]
ALL_ENTITY_CLASSES: ALL_ENTITY_CLASSES_TYPE = [
    WallEntity,  # type: ignore[list-item]
    DoorEntity,  # type: ignore[list-item]
    WindowEntity,  # type: ignore[list-item]
    BboxEntity,  # type: ignore[list-item]
]

def get_entity_class_from_token(command_value: int) -> WallEntity | DoorEntity | WindowEntity | BboxEntity:
    """Get the entity class from the integer token."""
    for entity_class in ALL_ENTITY_CLASSES:
        if command_value == entity_class.TOKEN:
            return entity_class
    raise ValueError(f"Unknown command token: {command_value}")


def get_entity_class_from_string(command_string: str) -> WallEntity | DoorEntity | WindowEntity | BboxEntity:
    """Get the entity class from the integer token."""
    for entity_class in ALL_ENTITY_CLASSES:
        if command_string == entity_class.COMMAND_STRING:
            return entity_class
    raise ValueError(f"Unknown command token: {command_string}")
