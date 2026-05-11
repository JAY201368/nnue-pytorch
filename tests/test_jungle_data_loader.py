import os
import struct
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules import JUNGLE_BASE_FEATURE_SET
from model.modules.features.jungle_piece_square import (
    ELEPHANT,
    RAT,
    piece_square_index,
)
from model.modules.features.jungle_piece_terrain import piece_terrain_indices


def _write_jungle_binpack(path, records):
    payload = b"".join(records)
    with open(path, "wb") as f:
        f.write(b"BINP")
        f.write(struct.pack("<I", len(payload)))
        f.write(payload)


def _packed_record(piece_squares, side_to_move=0, score=32, ply=12, result=1, flags=0):
    return struct.pack(
        "<16sBhHbB",
        bytes(piece_squares),
        side_to_move,
        score,
        ply,
        result,
        flags,
    )


def test_jungle_binpack_sparse_batch_matches_python_feature_formula(tmp_path):
    try:
        from data_loader import DataloaderSkipConfig, SparseBatchProvider
    except ImportError as exc:
        pytest.skip(f"native data loader is not built: {exc}")

    captured = 63
    piece_squares = [captured] * 16
    piece_squares[ELEPHANT] = 0
    piece_squares[8 + RAT] = 62

    data_path = tmp_path / "sample.binpack"
    _write_jungle_binpack(data_path, [_packed_record(piece_squares)])

    config = DataloaderSkipConfig(
        filtered=False,
        wld_filtered=False,
        random_fen_skipping=0,
        early_fen_skipping=-1,
        soft_early_fen_skipping=0,
        simple_eval_skipping=-1,
    )
    provider = SparseBatchProvider(
        JUNGLE_BASE_FEATURE_SET,
        [str(data_path)],
        batch_size=1,
        cyclic=False,
        num_workers=1,
        config=config,
    )

    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = next(provider)

    assert us.item() == 1.0
    assert them.item() == 0.0
    assert outcome.item() == 1.0
    assert score.item() == 32
    assert psqt_indices.item() == 0
    assert layer_stack_indices.item() == 0

    assert white_indices.shape == (1, 64)
    assert black_indices.shape == (1, 64)
    assert white_values[0, :4].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert black_values[0, :4].tolist() == [1.0, 1.0, 1.0, 1.0]

    expected_white = [
        piece_square_index(True, 0, ELEPHANT, True),
        piece_square_index(True, 62, RAT, False),
        1008 + piece_terrain_indices(True, 0, ELEPHANT, True)[0],
        1008 + piece_terrain_indices(True, 62, RAT, False)[0],
    ]
    expected_black = [
        piece_square_index(False, 0, ELEPHANT, True),
        piece_square_index(False, 62, RAT, False),
        1008 + piece_terrain_indices(False, 0, ELEPHANT, True)[0],
        1008 + piece_terrain_indices(False, 62, RAT, False)[0],
    ]

    assert white_indices[0, :4].tolist() == expected_white
    assert black_indices[0, :4].tolist() == expected_black
    assert white_indices[0, 4:].eq(-1).all()
    assert black_indices[0, 4:].eq(-1).all()
