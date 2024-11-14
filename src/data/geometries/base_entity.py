# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseEntity(ABC):
    @property
    @abstractmethod
    def COMMAND_STRING(self) -> str:
        pass

    @property
    @abstractmethod
    def PARAMS_DEFINITION(self) -> dict[str, Any]:
        pass

    @property
    @abstractmethod
    def TOKEN(self) -> int:
        pass

    @abstractmethod
    def extent(self) -> dict[str, float]:
        pass

    @abstractmethod
    def rotate(self, rotation_angle: float) -> None:
        pass

    @abstractmethod
    def translate(self, translation: torch.Tensor) -> None:
        pass

    @abstractmethod
    def lex_sort_key(self) -> np.ndarray:
        pass

    def random_sort_key(self) -> np.ndarray:
        """Compute sorting key for random sorting.

        Returns:
            a [1] np.ndarray.
        """
        return np.random.rand(1)  # [1]

    def sort_key(self, sort_type: str) -> np.ndarray:
        """Compute sorting key.

        Args:
            sort_type: str.

        Returns:
            an np.ndarray.
        """
        assert sort_type in ["lex", "random"]
        if sort_type == "lex":
            return self.lex_sort_key()
        elif sort_type == "random":
            return self.random_sort_key()

    @abstractmethod
    def to_seq_value(self) -> list[Any]:
        pass
