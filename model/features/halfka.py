from collections import OrderedDict

import chess  # TODO: handcraft our own chess lib
import torch

from .feature_block import FeatureBlock

RANKS = 9  # 9行
FILES = 7  # 7列
NUM_SQ = FILES * RANKS  # 63格
NUM_PT = 16  # 双方各8种
NUM_PLANES = NUM_SQ * NUM_PT + 1  # 1009
NUM_ATTACK_BUCKETS = 8  # 攻击桶数量


def orient(
        is_white_pov: bool,
        sq: int | chess.Square
) -> int:
    # 56 = 0b111000
    # 对称变换(关于横中轴镜像)
    # 原写法
    # return (56 * (not is_white_pov)) ^ sq
    # 原写法等效
    # if is_white_pov:
    #     return sq
    # else:
    #     return 56 ^ sq
    # 改为斗兽棋
    if is_white_pov:
        return sq
    rank = sq // FILES
    file = sq % FILES
    flipped_rank = RANKS - 1 - rank
    return flipped_rank * FILES + file

def halfaa_idx(
        is_white_pov: bool,
        # king_sq: int,  不再需要王位
        sq: int,
        p: chess.Piece,
        attack_bucket: int
):
    """
    定位棋子 + 位置在特征向量中的索引
    """
    # 白方偶数, 黑方奇数
    p_idx = (p.piece_type - 1) * 2 if p.color == is_white_pov else (p.piece_type - 1) * 2 + 1
    # 王不动, 删除王桶, 改为用攻击桶
    return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + attack_bucket * NUM_PLANES

def classify_attack_bucket(board: chess.Board) -> int:
    # TODO: classify attack buckets
    return 0

def halfaa_psqts():
    """
    填写psqt值
    """
    # values copied from stockfish, in stockfish internal units
    piece_values = {
        # chess.PAWN: 126,
        # chess.KNIGHT: 781,
        # chess.BISHOP: 825,
        # chess.ROOK: 1276,
        # chess.QUEEN: 2538,
        # TODO: fill up piece-square table values
        chess.ELEPHANT: 0,
        chess.WOLF: 0,
        chess.CHEETAH: 0,
        chess.MOUSE: 0,
        chess.CAT: 0,
        chess.DOG: 0,
        chess.TIGER: 0,
        chess.LION: 0,
    }

    values = [0] * (NUM_PLANES * NUM_ATTACK_BUCKETS)

    # for ksq in range(64):
    for attack_bucket in range(NUM_ATTACK_BUCKETS):
        for s in range(NUM_SQ):
            for pt, val in piece_values.items():
                idxw = halfaa_idx(True, s, chess.Piece(pt, chess.WHITE), attack_bucket)
                idxb = halfaa_idx(True, s, chess.Piece(pt, chess.BLACK), attack_bucket)
                values[idxw] = val
                values[idxb] = -val

    return values


class Features(FeatureBlock):
    def __init__(self):
        super().__init__(
            "HalfAA", 0x5F134CB8, OrderedDict([("HalfAA", NUM_PLANES * NUM_ATTACK_BUCKETS)])
        )

    def get_active_features(
        self, board: chess.Board
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        接收棋盘
        返回黑白两个视角的特征向量
        """
        def piece_features(turn):
            indices = torch.zeros(NUM_PLANES)
            # ksq = board.king(turn)
            # assert ksq is not None
            attack_bucket = classify_attack_bucket(board)
            for sq, p in board.piece_map().items():
                indices[halfaa_idx(turn, sq, p, attack_bucket)] = 1.0
            return indices

        return piece_features(chess.WHITE), piece_features(chess.BLACK)

    def get_initial_psqt_features(self) -> list[int]:
        return halfaa_psqts()


class FactorizedFeatures(FeatureBlock):
    def __init__(self):
        super().__init__(
            "HalfAA^",
            0x5F134CB8,
            OrderedDict([("HalfAA", NUM_PLANES * NUM_ATTACK_BUCKETS), ("A", NUM_SQ * NUM_PT)]),
        )

    def get_active_features(self, board: chess.Board):
        raise Exception(
            "Not supported yet, you must use the c++ data loader for factorizer support during training"
        )

    def get_feature_factors(self, idx: int) -> list[int]:
        if idx >= self.num_real_features:
            raise Exception("Feature must be real")

        a_idx = idx % NUM_PLANES - 1

        return [idx, self.get_factor_base_feature("A") + a_idx]

    def get_initial_psqt_features(self) -> list[int]:
        return halfaa_psqts() + [0] * (NUM_SQ * NUM_PT)


"""
This is used by the features module for discovery of feature blocks.
"""


def get_feature_block_clss() -> list[type[FeatureBlock]]:
    return [Features, FactorizedFeatures]
