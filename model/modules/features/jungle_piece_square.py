import torch
from torch import nn

from .input_feature import InputFeature

BOARD_FILES = 7
BOARD_RANKS = 9
NUM_SQUARES = BOARD_FILES * BOARD_RANKS

NUM_PIECE_TYPES = 8
NUM_RELATIVE_OWNERS = 2

ELEPHANT = 0
LION = 1
TIGER = 2
PANTHER = 3
WOLF = 4
DOG = 5
CAT = 6
RAT = 7

PIECE_VALUES = [800, 700, 600, 500, 400, 300, 200, 100]


def orient_square(is_white_pov: bool, square: int) -> int:
    """Orient a physical square so the POV player's home side is rank 0."""
    if not 0 <= square < NUM_SQUARES:
        raise ValueError(f"square must be in [0, {NUM_SQUARES}), got {square}")
    return square if is_white_pov else NUM_SQUARES - 1 - square


def relative_owner(is_white_pov: bool, piece_is_white: bool) -> int:
    return 0 if piece_is_white == is_white_pov else 1


def piece_square_index(
    is_white_pov: bool,
    square: int,
    piece_type: int,
    piece_is_white: bool,
) -> int:
    """Index for a piece on a square from one player's perspective."""
    if not 0 <= piece_type < NUM_PIECE_TYPES:
        raise ValueError(f"piece_type must be in [0, {NUM_PIECE_TYPES}), got {piece_type}")
    owner = relative_owner(is_white_pov, piece_is_white)
    return orient_square(is_white_pov, square) + NUM_SQUARES * (
        piece_type + NUM_PIECE_TYPES * owner
    )


class JunglePieceSquare(InputFeature):
    HASH = 0xA318C9D1
    FEATURE_NAME = "JunglePieceSquare"
    INPUT_FEATURE_NAME = "JunglePieceSquare"
    MAX_ACTIVE_FEATURES = 16

    NUM_SQ = NUM_SQUARES
    NUM_PT = NUM_PIECE_TYPES * NUM_RELATIVE_OWNERS
    NUM_INPUTS = NUM_SQ * NUM_PT
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
        scale = 1.0 / nnue2score
        L1 = self.num_outputs - num_psqt_buckets

        initial_values = torch.tensor(
            self.piece_square_psqts(),
            device=self.weight.device,
            dtype=self.weight.dtype,
        ).mul_(scale)

        for i in range(num_psqt_buckets):
            self.weight[:, L1 + i] = initial_values

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)

    @staticmethod
    def piece_square_psqts() -> list[int]:
        values = [0] * JunglePieceSquare.NUM_INPUTS

        for owner in range(NUM_RELATIVE_OWNERS):
            sign = 1 if owner == 0 else -1
            for piece_type, piece_value in enumerate(PIECE_VALUES):
                offset = NUM_SQUARES * (piece_type + NUM_PIECE_TYPES * owner)
                for square in range(NUM_SQUARES):
                    values[offset + square] = sign * piece_value

        return values
