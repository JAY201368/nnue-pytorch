import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules import JUNGLE_BASE_FEATURE_SET, get_available_features, get_feature_cls
from model.modules.features.jungle_piece_square import (
    ELEPHANT,
    NUM_SQUARES,
    PANTHER,
    JunglePieceSquare,
    orient_square,
    piece_square_index,
)
from model.modules.features.jungle_piece_terrain import (
    ENEMY_TRAPS,
    NUM_TERRAIN_TAGS,
    TERRAIN_ENEMY_DEN_ADJACENT,
    TERRAIN_ENEMY_TRAP,
    TERRAIN_LAND,
    TERRAIN_WATER,
    WATER,
    JunglePieceTerrain,
    piece_terrain_indices,
    terrain_tags_for_oriented_square,
)


def test_jungle_feature_set_registered_and_composes():
    assert "JunglePieceSquare" in get_available_features()
    assert "JunglePieceTerrain" in get_available_features()

    feature_cls = get_feature_cls(JUNGLE_BASE_FEATURE_SET)
    transformer = feature_cls(16)

    assert transformer.FEATURE_NAME == JUNGLE_BASE_FEATURE_SET
    assert transformer.INPUT_FEATURE_NAME == JUNGLE_BASE_FEATURE_SET
    assert transformer.NUM_INPUTS == (
        JunglePieceSquare.NUM_INPUTS + JunglePieceTerrain.NUM_INPUTS
    )
    assert transformer.NUM_REAL_FEATURES == transformer.NUM_INPUTS
    assert transformer.MAX_ACTIVE_FEATURES == (
        JunglePieceSquare.MAX_ACTIVE_FEATURES + JunglePieceTerrain.MAX_ACTIVE_FEATURES
    )


def test_jungle_piece_square_orientation_and_indexing():
    assert orient_square(True, 0) == 0
    assert orient_square(False, 0) == NUM_SQUARES - 1

    white_elephant_a1 = piece_square_index(True, 0, ELEPHANT, True)
    black_pov_white_elephant_a1 = piece_square_index(False, 0, ELEPHANT, True)

    assert white_elephant_a1 == 0
    assert black_pov_white_elephant_a1 == (NUM_SQUARES - 1) + NUM_SQUARES * 8


def test_jungle_piece_square_psqt_initializes_material_columns():
    feature = JunglePieceSquare(num_outputs=10)
    with torch.no_grad():
        feature.weight.zero_()
    feature.init_weights(num_psqt_buckets=2, nnue2score=100.0)

    own_elephant = piece_square_index(True, 0, ELEPHANT, True)
    enemy_elephant = piece_square_index(True, 0, ELEPHANT, False)

    assert torch.all(feature.weight[own_elephant, -2:] == 8.0)
    assert torch.all(feature.weight[enemy_elephant, -2:] == -8.0)


def test_jungle_terrain_tags_are_relative_to_pov():
    enemy_trap = next(iter(ENEMY_TRAPS))
    tags = terrain_tags_for_oriented_square(enemy_trap)

    assert TERRAIN_LAND in tags
    assert TERRAIN_ENEMY_TRAP in tags
    assert TERRAIN_ENEMY_DEN_ADJACENT in tags

    water_square = next(iter(WATER))
    assert terrain_tags_for_oriented_square(water_square) == (TERRAIN_WATER,)


def test_jungle_piece_terrain_indices_use_piece_owner_plane():
    enemy_trap = next(iter(ENEMY_TRAPS))
    indices = piece_terrain_indices(True, enemy_trap, PANTHER, True)

    assert set(indices) == {
        NUM_TERRAIN_TAGS * PANTHER + TERRAIN_LAND,
        NUM_TERRAIN_TAGS * PANTHER + TERRAIN_ENEMY_TRAP,
        NUM_TERRAIN_TAGS * PANTHER + TERRAIN_ENEMY_DEN_ADJACENT,
    }

    black_pov_indices = piece_terrain_indices(False, enemy_trap, PANTHER, True)
    assert all(index >= NUM_TERRAIN_TAGS * (PANTHER + 8) for index in black_pov_indices)
