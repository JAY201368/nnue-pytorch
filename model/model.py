"""
包含网络架构的定义并描述推理过程
模型被定义为一个名为 NNUE 的 pl.LightningModule. 
模型定义了推理过程并选择要优化的参数. 为了考虑量化, 它在每一步后将某些层的权重裁剪到支持的范围内. 训练循环由 pytorch-lightning 处理, 它通过模型的某些方法与模型对接. 
模型还包含代码, 允许它在加载现有模型时在某些特征集之间进行转换. 值得注意的是, 在加载现有模型后, 可以添加虚拟特征. 模型模块还公开了一个函数, 用于将这些虚拟特征权重合并为真实特征权重. 
"""
from typing import Generator

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .config import ModelConfig
from .feature_transformer import DoubleFeatureTransformerSlice
from .features import FeatureSet
from .quantize import QuantizationConfig, QuantizationManager

# 线性层类
class StackedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, count: int):
        """
        count -> 桶(并行计算的特征向量)的数量
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.count = count
        self.linear = nn.Linear(in_features, out_features * count)

        self._init_uniformly()

    @torch.no_grad()
    def _init_uniformly(self) -> None:
        init_weight = self.linear.weight[0 : self.out_features, :]
        init_bias = self.linear.bias[0 : self.out_features]

        self.linear.weight.copy_(init_weight.repeat(self.count, 1))
        self.linear.bias.copy_(init_bias.repeat(self.count))

    def forward(self, x: Tensor, ls_indices: Tensor) -> Tensor:
        stacked_output = self.linear(x)

        return self.select_output(stacked_output, ls_indices)

    def select_output(self, stacked_output: Tensor, ls_indices: Tensor) -> Tensor:
        reshaped_output = stacked_output.reshape(-1, self.out_features)

        idx_offset = torch.arange(
            0,
            ls_indices.shape[0] * self.count,
            self.count,
            device=stacked_output.device,
        )
        indices = ls_indices.flatten() + idx_offset

        selected_output = reshaped_output[indices]

        return selected_output

    @torch.no_grad()
    def at_index(self, index: int) -> nn.Linear:
        layer = nn.Linear(self.in_features, self.out_features)

        begin = index * self.out_features
        end = (index + 1) * self.out_features

        layer.weight.copy_(self.linear.weight[begin:end, :])
        layer.bias.copy_(self.linear.bias[begin:end])

        return layer

class FactorizedStackedLinear(StackedLinear):
    def __init__(self, in_features: int, out_features: int, count: int):
        super().__init__(in_features, out_features, count)

        self.factorized_linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            self.factorized_linear.weight.zero_()
            self.factorized_linear.bias.zero_()

    def forward(self, x: Tensor, ls_indices: Tensor) -> Tensor:
        # coalesce?
        merged_weight = self.linear.weight + self.factorized_linear.weight.repeat(
            self.count, 1
        )
        merged_bias = self.linear.bias + self.factorized_linear.bias.repeat(self.count)

        stacked_output = F.linear(x, merged_weight, merged_bias)

        return self.select_output(stacked_output, ls_indices)

    @torch.no_grad()
    def at_index(self, index: int) -> nn.Linear:
        layer = super().at_index(index)

        layer.weight.add_(self.factorized_linear.weight)
        layer.bias.add_(self.factorized_linear.bias)

        return layer

    @torch.no_grad()
    def coalesce_weights(self) -> None:
        """
        将虚拟特征对应的权重合并到真实特征上, 即将虚拟特征下标对应的权重行累加到真实特征下标对应的权重行
        """
        for i in range(self.count):
            begin = i * self.out_features
            end = (i + 1) * self.out_features

            self.linear.weight[begin:end, :].add_(self.factorized_linear.weight)
            self.linear.bias[begin:end].add_(self.factorized_linear.bias)

        self.factorized_linear.weight.zero_()
        self.factorized_linear.bias.zero_()

# 浮点神经网络
class LayerStacks(nn.Module):
    def __init__(self, count: int, config: ModelConfig):
        """
        count -> 桶(并行计算的特征向量)的数量
        """
        super().__init__()

        self.count = count
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        # Factorizer only for the first layer because later
        # there's a non-linearity and factorization breaks.
        # This is by design. The weights in the further layers should be
        # able to diverge a lot.
        # 每桶分得 L2 + 1 维首层输出, 其中 L2 走激活, +1 是直通项, 后面直接加到输出
        self.l1 = FactorizedStackedLinear(2 * self.L1 // 2, self.L2 + 1, count)    # 奇数变偶数? 线性层l1: 3072 -> (15+1) * count
        self.l2 = StackedLinear(self.L2 * 2, self.L3, count)  # count个L2池化成一个, fact单独保留(?), 最终剩下两倍L2的输入维数. 线性层l2: 15 * 2 -> 32 * count
        self.output = StackedLinear(self.L3, 1, count)   # 输出层, 线性层output: 32 -> 1 * count

        with torch.no_grad():
            self.output.linear.bias.zero_()

    def forward(self, x: Tensor, ls_indices: Tensor):
        l1c_ = self.l1(x, ls_indices)
        l1x_, l1x_out = l1c_.split(self.L2, dim=1)
        # multiply sqr crelu result by (127/128) to match quantized version
        l1x_ = torch.clamp(
            torch.cat([torch.pow(l1x_, 2.0) * (127 / 128), l1x_], dim=1), 0.0, 1.0
        )

        l2c_ = self.l2(l1x_, ls_indices)
        l2x_ = torch.clamp(l2c_, 0.0, 1.0)

        l3c_ = self.output(l2x_, ls_indices)
        l3x_ = l3c_ + l1x_out

        return l3x_

    @torch.no_grad()
    def get_coalesced_layer_stacks(
        self,
    ) -> Generator[tuple[nn.Linear, nn.Linear, nn.Linear], None, None]:
        # During training the buckets are represented by a single, wider, layer.
        # This representation needs to be transformed into individual layers
        # for the serializer, because the buckets are interpreted as separate layers.
        for i in range(self.count):
            yield self.l1.at_index(i), self.l2.at_index(i), self.output.at_index(i)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self) -> None:
        self.l1.coalesce_weights()

# 整数网络
class NNUEModel(nn.Module):
    def __init__(
        self,
        feature_set: FeatureSet,
        config: ModelConfig,
        quantize_config: QuantizationConfig,
        num_psqt_buckets: int = 8,
        num_ls_buckets: int = 8,
    ):
        super().__init__()

        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.num_psqt_buckets = num_psqt_buckets
        self.num_ls_buckets = num_ls_buckets

        self.input = DoubleFeatureTransformerSlice(
            feature_set.num_features, self.L1 + self.num_psqt_buckets
        )
        self.feature_set = feature_set
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config)

        self.quantization = QuantizationManager(quantize_config)
        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self._init_layers()

    def _init_layers(self):
        self._zero_virtual_feature_weights()
        self._init_psqt()

    def _zero_virtual_feature_weights(self):
        """
        We zero all virtual feature weights because there's not need for them
        to be initialized; they only aid the training of correlated features.
        """
        weights = self.input.weight
        with torch.no_grad():
            for a, b in self.feature_set.get_virtual_feature_ranges():
                weights[a:b, :] = 0.0
        self.input.weight = nn.Parameter(weights)

    def _init_psqt(self):
        input_weights = self.input.weight
        input_bias = self.input.bias
        # 1.0 / kPonanzaConstant
        scale = 1 / self.quantization.nnue2score

        with torch.no_grad():
            initial_values = self.feature_set.get_initial_psqt_features()
            assert len(initial_values) == self.feature_set.num_features

            new_weights = (
                torch.tensor(
                    initial_values,
                    device=input_weights.device,
                    dtype=input_weights.dtype,
                )
                * scale
            )

            for i in range(self.num_psqt_buckets):
                input_weights[:, self.L1 + i] = new_weights
                # Bias doesn't matter because it cancels out during
                # inference during perspective averaging. We set it to 0
                # just for the sake of it. It might still diverge away from 0
                # due to gradient imprecision but it won't change anything.
                input_bias[self.L1 + i] = 0.0

        self.input.weight = nn.Parameter(input_weights)
        self.input.bias = nn.Parameter(input_bias)

    def clip_weights(self):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        根据量化准则, 缩放模型各层参数
        准则对应init()中定义的剪裁配置列表weight_clipping
        self.weight_clipping = [
            {
                "params": [self.layer_stacks.l1.weight],
                "min_weight": -max_hidden_weight,
                "max_weight": max_hidden_weight,
                "virtual_params": self.layer_stacks.l1_fact.weight,
            },
            {
                "params": [self.layer_stacks.l2.weight],
                "min_weight": -max_hidden_weight,
                "max_weight": max_hidden_weight,
            },
            {
                "params": [self.layer_stacks.output.weight],
                "min_weight": -max_out_weight,
                "max_weight": max_out_weight,
            },
        ]
        """
        for group in self.weight_clipping:
            for p in group["params"]:
                if "min_weight" in group or "max_weight" in group:
                    p_data_fp32 = p.data
                    min_weight = group["min_weight"]
                    max_weight = group["max_weight"]
                    if "virtual_params" in group:
                        virtual_params = group["virtual_params"]
                        xs = p_data_fp32.shape[0] // virtual_params.shape[0]
                        ys = p_data_fp32.shape[1] // virtual_params.shape[1]
                        expanded_virtual_layer = virtual_params.repeat(xs, ys)
                        if min_weight is not None:
                            min_weight_t = (
                                p_data_fp32.new_full(p_data_fp32.shape, min_weight)
                                - expanded_virtual_layer
                            )
                            p_data_fp32 = torch.max(p_data_fp32, min_weight_t)
                        if max_weight is not None:
                            max_weight_t = (
                                p_data_fp32.new_full(p_data_fp32.shape, max_weight)
                                - expanded_virtual_layer
                            )
                            p_data_fp32 = torch.min(p_data_fp32, max_weight_t)
                    else:
                        if min_weight is not None and max_weight is not None:
                            p_data_fp32.clamp_(min_weight, max_weight)
                        else:
                            raise Exception("Not supported.")
                    p.data.copy_(p_data_fp32)

    def set_feature_set(self, new_feature_set: FeatureSet):
        """
        This method attempts to convert the model from using the self.feature_set
        to new_feature_set. Currently only works for adding virtual features.
        """
        if self.feature_set.name == new_feature_set.name:
            return

        # TODO: Implement this for more complicated conversions.
        #       Currently we support only a single feature block.
        if len(self.feature_set.features) > 1:
            raise Exception(
                "Cannot change feature set from {} to {}.".format(
                    self.feature_set.name, new_feature_set.name
                )
            )

        # Currently we only support conversion for feature sets with
        # one feature block each so we'll dig the feature blocks directly
        # and forget about the set.
        old_feature_block = self.feature_set.features[0]
        new_feature_block = new_feature_set.features[0]

        # next(iter(new_feature_block.factors)) is the way to get the
        # first item in a OrderedDict. (the ordered dict being str : int
        # mapping of the factor name to its size).
        # It is our new_feature_factor_name.
        # For example old_feature_block.name == "HalfKP"
        # and new_feature_factor_name == "HalfKP^"
        # We assume here that the "^" denotes factorized feature block
        # and we would like feature block implementers to follow this convention.
        # So if our current feature_set matches the first factor in the new_feature_set
        # we only have to add the virtual feature on top of the already existing real ones.
        if old_feature_block.name == next(iter(new_feature_block.factors)):
            # We can just extend with zeros since it's unfactorized -> factorized
            weights = self.input.weight
            padding = weights.new_zeros(
                (new_feature_block.num_virtual_features, weights.shape[1])
            )
            weights = torch.cat([weights, padding], dim=0)
            self.input.weight = nn.Parameter(weights)
            self.feature_set = new_feature_set
        else:
            raise Exception(
                "Cannot change feature set from {} to {}.".format(
                    self.feature_set.name, new_feature_set.name
                )
            )

    def forward(
        self,
        us: Tensor,
        them: Tensor,
        white_indices: Tensor,
        white_values: Tensor,
        black_indices: Tensor,
        black_values: Tensor,
        psqt_indices: Tensor,
        layer_stack_indices: Tensor,
    ):
        # wp, bp各自表示由DoubleFeatureTransformerSlice生成的两侧视角的稠密输入向量
        # 形状均为[batch_size, L1 + num_psqt_buckets]
        wp, bp = self.input(white_indices, white_values, black_indices, black_values)  # 黑白双方视角
        # 分离主特征和PSQT输入
        w, wpsqt = torch.split(wp, self.L1, dim=1)
        b, bpsqt = torch.split(bp, self.L1, dim=1)
        # 依据当前行动方权重融合两侧主特征, 己方和敌方使用不同加权
        # torch.cat拼接张量, 加号作用为逐元素相加. 
        l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        l0_ = torch.clamp(l0_, 0.0, 1.0)  # 裁剪输入特征向量

        l0_s = torch.split(l0_, self.L1 // 2, dim=1)  # 将l0_按照每块L1//2切成四块
        l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]  # 两两逐元素相乘, 再拼接为新的输入(池化)
        # We multiply by 127/128 because in the quantized network 1.0 is represented by 127
        # and it's more efficient to divide by 128 instead.
        # 量化网络里1.0用127表示, 
        # 用127/128做轻微缩放, 避免饱和并便于用移位近似除以128(?)
        l0_ = torch.cat(l0_s1, dim=1) * (127 / 128)

        # 每个样本选择的PSQT桶索引. 本步操作将psqt_index从一行[B]变为一列[B][1]
        psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
        # 根据psqt_indices列表选取psqt桶, 选择前形状为[B][num_psqt_buckets], 选择后形状变为[B][1]
        wpsqt = wpsqt.gather(1, psqt_indices_unsq)
        bpsqt = bpsqt.gather(1, psqt_indices_unsq)
        # The PSQT values are averaged over perspectives. "Their" perspective
        # has a negative influence (us-0.5 is 0.5 for white and -0.5 for black,
        # which does both the averaging and sign flip for black to move)
        x = self.layer_stacks(l0_, layer_stack_indices) + (wpsqt - bpsqt) * (us - 0.5)

        return x
