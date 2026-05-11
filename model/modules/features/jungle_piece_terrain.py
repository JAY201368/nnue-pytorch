import torch
from torch import nn

from .input_feature import InputFeature
from .jungle_piece_square import (
    BOARD_FILES,
    NUM_PIECE_TYPES,
    NUM_RELATIVE_OWNERS,
    NUM_SQUARES,
    orient_square,
    relative_owner,
)

TERRAIN_LAND = 0
TERRAIN_WATER = 1
TERRAIN_OWN_TRAP = 2
TERRAIN_ENEMY_TRAP = 3
TERRAIN_OWN_DEN = 4
TERRAIN_ENEMY_DEN = 5
TERRAIN_OWN_DEN_ADJACENT = 6
TERRAIN_ENEMY_DEN_ADJACENT = 7

NUM_TERRAIN_TAGS = 8
MAX_TERRAIN_TAGS_PER_PIECE = 3


def _sq(file: int, rank: int) -> int:
    return rank * BOARD_FILES + file


# Squares are expressed in oriented coordinates: the POV player's den is on rank 0.
OWN_DEN = _sq(3, 0)
ENEMY_DEN = _sq(3, 8)

OWN_TRAPS = frozenset({_sq(2, 0), _sq(4, 0), _sq(3, 1)})
ENEMY_TRAPS = frozenset({_sq(2, 8), _sq(4, 8), _sq(3, 7)})

WATER = frozenset(
    _sq(file, rank)
    for file in (1, 2, 4, 5)
    for rank in (3, 4, 5)
)

OWN_DEN_ADJACENT = frozenset({_sq(2, 0), _sq(4, 0), _sq(3, 1)})
ENEMY_DEN_ADJACENT = frozenset({_sq(2, 8), _sq(4, 8), _sq(3, 7)})


def terrain_tags_for_oriented_square(square: int) -> tuple[int, ...]:
    if not 0 <= square < NUM_SQUARES:
        raise ValueError(f"square must be in [0, {NUM_SQUARES}), got {square}")

    tags: list[int] = [TERRAIN_WATER if square in WATER else TERRAIN_LAND]

    if square in OWN_TRAPS:
        tags.append(TERRAIN_OWN_TRAP)
    if square in ENEMY_TRAPS:
        tags.append(TERRAIN_ENEMY_TRAP)
    if square == OWN_DEN:
        tags.append(TERRAIN_OWN_DEN)
    if square == ENEMY_DEN:
        tags.append(TERRAIN_ENEMY_DEN)
    if square in OWN_DEN_ADJACENT:
        tags.append(TERRAIN_OWN_DEN_ADJACENT)
    if square in ENEMY_DEN_ADJACENT:
        tags.append(TERRAIN_ENEMY_DEN_ADJACENT)

    return tuple(tags)


def piece_terrain_indices(
    is_white_pov: bool,
    square: int,
    piece_type: int,
    piece_is_white: bool,
) -> tuple[int, ...]:
    if not 0 <= piece_type < NUM_PIECE_TYPES:
        raise ValueError(f"piece_type must be in [0, {NUM_PIECE_TYPES}), got {piece_type}")

    owner = relative_owner(is_white_pov, piece_is_white)
    plane = piece_type + NUM_PIECE_TYPES * owner
    base = NUM_TERRAIN_TAGS * plane
    oriented_square = orient_square(is_white_pov, square)

    return tuple(base + tag for tag in terrain_tags_for_oriented_square(oriented_square))


class JunglePieceTerrain(InputFeature):
    HASH = 0xC7B64A15
    FEATURE_NAME = "JunglePieceTerrain"
    INPUT_FEATURE_NAME = "JunglePieceTerrain"
    MAX_ACTIVE_FEATURES = 16 * MAX_TERRAIN_TAGS_PER_PIECE

    NUM_TERRAIN = NUM_TERRAIN_TAGS
    NUM_PT = NUM_PIECE_TYPES * NUM_RELATIVE_OWNERS
    NUM_INPUTS = NUM_TERRAIN * NUM_PT
    NUM_REAL_FEATURES = NUM_INPUTS

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        L1 = self.num_outputs - num_psqt_buckets
        for i in range(num_psqt_buckets):
            self.weight[:, L1 + i] = 0.0

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)
