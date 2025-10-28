# 特征集定义其部分(特征块)及其大小, 为每个特征提供初始的 PSQT 值, 并允许检索每个特征的因子(与给定实特征对应的所有特征索引)
import chess  # 需要仿制斗兽棋的棋盘类
import torch

from .feature_block import FeatureBlock


def _calculate_features_hash(features):
    if len(features) == 1:
        return features[0].hash

    tail_hash = _calculate_features_hash(features[1:])
    return features[0].hash ^ (tail_hash << 1) ^ (tail_hash >> 1) & 0xFFFFFFFF  # 何意味


class FeatureSet:
    """
    A feature set is nothing more than a list of named FeatureBlocks.
    It itself functions similarly to a feature block, but we don't want to be
    explicit about it as we don't want it to be used as a building block for other
    feature sets. You can think of this class as a composite, but not the full extent.
    It is basically a concatenation of feature blocks.
    """

    def __init__(
            self,
            features  # List of feature blocks
    ):
        for feature in features:
            if not isinstance(feature, FeatureBlock):
                raise Exception("All features must subclass FeatureBlock")

        self.features = features
        self.hash = _calculate_features_hash(features)
        self.name = "+".join(feature.name for feature in features)
        self.num_real_features = sum(feature.num_real_features for feature in features)
        self.num_virtual_features = sum(
            feature.num_virtual_features for feature in features
        )
        self.num_features = sum(feature.num_features for feature in features)

    def get_virtual_feature_ranges(self) -> list[tuple[int, int]]:
        """
        This method returns the feature ranges for the virtual factors of the
        underlying feature blocks. This is useful to know during initialization,
        when we want to zero initialize the virtual feature weights, but give some other
        values to the real feature weights.
        """
        ranges = []
        offset = 0
        for feature in self.features:
            if feature.num_virtual_features:
                ranges.append(
                    (offset + feature.num_real_features, offset + feature.num_features)
                )
            offset += feature.num_features

        return ranges

    def get_real_feature_ranges(self) -> list[tuple[int, int]]:
        """
        返回一个列表, 列表元素为每个特征块的实特征在特征向量中的下标范围
        """
        ranges = []
        offset = 0
        for feature in self.features:
            ranges.append((offset, offset + feature.num_real_features))
            offset += feature.num_features

        return ranges

    def get_active_features(
        self,
        board: chess.Board
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method goes over all the feature blocks and gathers the active features.
        Each block has its own index space assigned so the features from two different
        blocks will never have the same index here. Basically the thing you would expect
        to happen after concatenating many feature blocks.
        根据棋盘状态, 得到特征张量(由各特征块的特征张量拼接而成)
        """
        w = torch.zeros(0)
        b = torch.zeros(0)

        offset = 0
        for feature in self.features:
            w_local, b_local = feature.get_active_features(board)
            w_local += offset
            b_local += offset
            w = torch.cat([w, w_local])
            b = torch.cat([b, b_local])
            offset += feature.num_features

        return w, b

    def get_feature_factors(
            self,
            idx: int  # 实特征索引
    ) -> list[int]:
        """
        This method takes a feature idx and looks for the block that owns it.
        If it found the block it asks it to factorize the index, otherwise
        it throws and Exception. The idx must refer to a real feature.
        输入一个实特征索引, 返回其所有因子化特征索引
        :param idx: real feature index
        :return: all factorized feature index
        """
        offset = 0
        for feature in self.features:
            if idx < offset + feature.num_real_features:
                return [offset + i for i in feature.get_feature_factors(idx - offset)]
            offset += feature.num_features

        raise Exception("No feature block to factorize {}".format(idx))

    def get_virtual_to_real_features_gather_indices(self) -> list[list[int]]:
        """
        This method does what get_feature_factors does but for all
        valid features at the same time. It returns a list of length
        self.num_real_features with ith element being a list of factors
        of the ith feature.
        This method is technically redundant, but it allows to perform the operation
        slightly faster when there's many feature blocks. It might be worth
        to add a similar method to the FeatureBlock itself - to make it faster
        for feature blocks with many factors.
        """
        indices = []
        real_offset = 0
        offset = 0
        # 遍历feature set中的每个feature block
        for feature in self.features:
            # 遍历每个feature block中的所有real_feature, 将其分解
            for i_real in range(feature.num_real_features):
                i_fact = feature.get_feature_factors(i_real)
                # 将分解后下标加入列表
                indices.append([offset + i for i in i_fact])
            real_offset += feature.num_real_features
            offset += feature.num_features
        return indices

    def get_initial_psqt_features(self) -> list[int]:
        init = []
        for feature in self.features:
            init += feature.get_initial_psqt_features()
        return init
