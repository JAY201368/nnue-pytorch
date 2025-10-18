from collections import OrderedDict

import chess
import torch

from .feature_block import FeatureBlock


NUM_SQ = 64
NUM_PT = 10
NUM_PLANES = NUM_SQ * NUM_PT + 1


def orient(is_white_pov: bool, sq: int) -> int:
    # white: sq -> sq
    # black: sq -> 0b111111 ^ sq
    # 中心对称(旋转)
    return (63 * (not is_white_pov)) ^ sq


def halfkp_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece) -> int:
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
    return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES


class Features(FeatureBlock):
    def __init__(self):
        super().__init__(
            "HalfKP", 0x5D69D5B8, OrderedDict([("HalfKP", NUM_PLANES * NUM_SQ)])
        )

    def get_active_features(
        self, board: chess.Board
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        make feature tensor according to current board
        :return: feature tensors of both two povs
        """
        def piece_features(turn):
            indices = torch.zeros(NUM_PLANES * NUM_SQ)
            for sq, p in board.piece_map().items():
                if p.piece_type == chess.KING:
                    continue
                ksq = board.king(turn)
                assert ksq is not None
                indices[halfkp_idx(turn, orient(turn, ksq), sq, p)] = 1.0
            return indices

        return (piece_features(chess.WHITE), piece_features(chess.BLACK))

    def get_initial_psqt_features(self):
        raise Exception("Not supported yet. See HalfKA")


class FactorizedFeatures(FeatureBlock):
    def __init__(self):
        super().__init__(
            "HalfKP^",
            0x5D69D5B8,
            OrderedDict(
                [("HalfKP", NUM_PLANES * NUM_SQ), ("HalfK", NUM_SQ), ("P", NUM_SQ * 10)]
            ),
        )
        self.base = Features()

    def get_active_features(
        self, board: chess.Board
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 调用原始HalfKP特征集的get_active_features方法, 获得白方和黑方的HalfKP特征张量
        white, black = self.base.get_active_features(board)

        def piece_features(base, color):
            indices = torch.zeros(NUM_SQ * 11)  # 创建大小为704的零张量, 对应HalfK(64) + P(640) = 704个虚拟特征
            piece_count = 0
            # P feature
            for sq, p in board.piece_map().items():
                if p.piece_type == chess.KING:
                    continue
                piece_count += 1
                p_idx = (p.piece_type - 1) * 2 + (p.color != color)  # pieceType index
                indices[(p_idx + 1) * NUM_SQ + orient(color, sq)] = 1.0  # first 64 slots are reserved for HalfK
            # HalfK feature
            # piece_count: 当前颜色方非王棋子的数量
            ksq = board.king(color)
            assert ksq is not None
            indices[orient(color, ksq)] = piece_count
            # 将原始HalfKP特征(base)与虚拟特征(indices)拼接
            # 最终张量大小：NUM_PLANES * NUM_SQ + NUM_SQ + NUM_SQ * 10 = 41024 + 64 + 640 = 41728
            return torch.cat((base, indices))

        return (piece_features(white, chess.WHITE), piece_features(black, chess.BLACK))

    def get_feature_factors(self, idx: int) -> list[int]:
        if idx >= self.num_real_features:
            raise Exception("Feature must be real")

        k_idx = idx // NUM_PLANES  # idx // 641 -> king pos
        p_idx = idx % NUM_PLANES - 1  # idx % 641 - 1 -> pieceType & piecePos

        return [
            idx,
            self.get_factor_base_feature("HalfK") + k_idx,
            self.get_factor_base_feature("P") + p_idx,
        ]

    def get_initial_psqt_features(self):
        raise Exception("Not supported yet. See HalfKA^")


"""
This is used by the features module for discovery of feature blocks.
"""


def get_feature_block_clss() -> list[type[FeatureBlock]]:
    return [Features, FactorizedFeatures]
