import lightning as L
import ranger21
import torch
from torch import Tensor, nn

from .config import LossParams, ModelConfig
from .features import FeatureSet
from .model import NNUEModel
from .quantize import QuantizationConfig


def _get_parameters(layers: list[nn.Module]):
    return [p for layer in layers for p in layer.parameters()]


class NNUE(L.LightningModule):
    """
    feature_set - an instance of FeatureSet defining the input features

    lambda_ = 0.0 - purely based on game results
    0.0 < lambda_ < 1.0 - interpolated score and result
    lambda_ = 1.0 - purely based on search scores

    gamma - the multiplicative factor applied to the learning rate after each epoch

    lr - the initial learning rate
    """

    def __init__(
        self,
        feature_set: FeatureSet,
        config: ModelConfig,
        quantize_config: QuantizationConfig,
        max_epoch=800,  # 一般400轮达到饱和
        num_batches_per_epoch=int(100_000_000 / 16384),
        gamma=0.992,    # 学习率衰减因子(lr *= gamma 每轮递减)
        lr=8.75e-4,     # 初始学习率
        param_index=0,
        num_psqt_buckets=8,
        num_ls_buckets=8,
        loss_params=LossParams(),
    ):
        super().__init__()
        self.model: NNUEModel = NNUEModel(
            feature_set, config, quantize_config, num_psqt_buckets, num_ls_buckets
        )
        self.loss_params = loss_params
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.gamma = gamma
        self.lr = lr
        self.param_index = param_index

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step_(self, batch: tuple[Tensor, ...], batch_idx, loss_type):
        """
        单步训练的行为
        """
        _ = batch_idx  # unused, but required by pytorch-lightning

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
        ) = batch  # 从batch对象中解包以上信息

        # 前向传播
        # 调用NNUE的forward pass得到网络评分, 并缩放转换为棋力分
        scorenet = (
            self.model(
                us,
                them,
                white_indices,
                white_values,
                black_indices,
                black_values,
                psqt_indices,
                layer_stack_indices,
            )
            * self.model.quantization.nnue2score
        )

        # 对网络分做缩放
        p = self.loss_params
        # convert the network and search scores to an estimate match result
        # based on the win_rate_model, with scalings and offsets optimized
        q = (scorenet - p.in_offset) / p.in_scaling  # used to compute the chance of a win
        qm = (-scorenet - p.in_offset) / p.in_scaling  # used to compute the chance of a loss
        qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())  # estimated match result (using win, loss and draw probs).

        # 引擎搜索分(来自训练数据?)
        s = (score - p.out_offset) / p.out_scaling
        sm = (-score - p.out_offset) / p.out_scaling
        pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

        # blend that eval based score with the actual game outcome
        t = outcome  # 对局结果
        # 根据轮次调整lambda的值(从start_lambda向end_lambda均匀逼近)
        actual_lambda = p.start_lambda + (p.end_lambda - p.start_lambda) * (
            self.current_epoch / self.max_epoch
        )
        # actual_lambda实际意义: pf和outcome在pt分中的加权占比, 每轮次逐渐变化
        # 一开始偏向于对局结果, 后来越来越偏向于引擎搜索分
        # 渐进式学习：
        # 初期：网络主要学习真实对局结果, 建立基本的胜负判断能力
        # 中期：逐渐引入引擎评分, 学习更精细的位置评估
        # 后期：主要依赖引擎评分(搜索分), 学习高水平的棋力评估
        pt = pf * actual_lambda + t * (1.0 - actual_lambda)

        # use a MSE-like loss function
        # 损失为 |pt - qf|^2.5 的均值, 记录日志, 返回损失(就可以在优化器中自动进行优化?)
        loss = torch.pow(torch.abs(pt - qf), p.pow_exp)
        if p.qp_asymmetry != 0.0:
            loss = loss * ((qf > pt) * p.qp_asymmetry + 1)
        loss = loss.mean()

        self.log(loss_type, loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        """
        训练周期的单步
        指定日志名为train_loss并返回损失给优化器
        """
        return self.step_(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        """
        验证周期的单步
        只做前向传播和日志记录, 不返回值, 不参与反向传播
        """
        self.step_(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx):
        """
        测试周期的单步
        不参与反向传播
        """
        self.step_(batch, batch_idx, "test_loss")

    def configure_optimizers(self):
        """
        配置优化器与学习率计划
        """
        LR = self.lr
        train_params = [
            {"params": _get_parameters([self.model.input]), "lr": LR, "gc_dim": 0},
            {"params": [self.model.layer_stacks.l1.factorized_linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.l1.factorized_linear.bias], "lr": LR},
            {"params": [self.model.layer_stacks.l1.linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.l1.linear.bias], "lr": LR},
            {"params": [self.model.layer_stacks.l2.linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.l2.linear.bias], "lr": LR},
            {"params": [self.model.layer_stacks.output.linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.output.linear.bias], "lr": LR},
        ]

        # 使用Ranger优化器(RAdam + LookAhead 变体配置)
        optimizer = ranger21.Ranger21(
            train_params,
            lr=1.0,
            betas=(0.9, 0.999),
            eps=1.0e-7,
            using_gc=False,
            using_normgc=False,
            weight_decay=0.0,
            num_batches_per_epoch=self.num_batches_per_epoch,
            num_epochs=self.max_epoch,
            warmdown_active=False,
            use_warmup=False,
            use_adaptive_gradient_clipping=False,
            softplus=False,
            pnm_momentum_factor=0.0,
        )

        # 设置按轮递减的StepLR(每轮乘以gamma)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma
        )

        # 返回优化器与调度器给PyTorch Lightning框架以备调用
        return [optimizer], [scheduler]
