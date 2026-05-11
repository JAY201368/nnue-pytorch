import { Card, CardBody, CardHeader, Divider, Grid, H1, H2, Pill, Stack, Stat, Table, Text } from 'cursor/canvas';

const moduleRows = [
  ['配置入口', 'config.py, model/config.py', 'TrainingConfig 聚合数据、硬件、训练周期、模型、loss、optimizer 等参数，并由 tyro 暴露为 CLI。'],
  ['训练编排', 'train.py', '检查数据路径，构造 NNUE LightningModule，创建 DataLoader、logger、checkpoint callback 和 Lightning Trainer。'],
  ['数据加载', 'data_loader/', 'SparseBatchDataset 调 C++ 原生 stream 读取 .binpack，FixedNumBatchesDataset 控制每个 epoch 的 batch 数并做预取。'],
  ['网络定义', 'model/model.py, model/modules/', 'NNUEModel 由输入特征变换器、LayerStacks、PSQT bucket、量化/裁剪配置组成。'],
  ['训练逻辑', 'model/lightning_module.py', 'NNUE LightningModule 定义 forward、training_step、validation_step、loss、optimizer 参数组。'],
  ['优化器', 'model/optimizers/', 'OptimizerConfig 在 ranger21 与 schedulefree 之间选择，并设置 lr 与不同参数组的 weight decay。'],
  ['导出与验证', 'serialize.py, cross_check_eval.py, run_games.py', '将 checkpoint/model 导出为 .nnue，并支持与引擎评估或批量下遍筛网。'],
  ['分布式辅助', 'ddp_launcher.py, ddp_utils/', 'torchrun 前置包装，计算线程、worker、rank/world size 等 DDP 运行参数。'],
];

const flowRows = [
  ['1', 'CLI 参数', 'tyro.cli(TrainingConfig) 解析 datasets、batch_size、accelerator、features、loss、optimizer 等。'],
  ['2', '模型实例', 'M.NNUE(config=nnue_lightning_config) 包住 NNUEModel，并传入 max_epoch、num_batches_per_epoch。'],
  ['3', '数据流', 'SparseBatchDataset -> C++ sparse batch stream -> FixedNumBatchesDataset -> torch DataLoader(batch_size=None)。'],
  ['4', '训练器', 'Lightning Trainer 管理 DDP、日志、checkpoint、回调、compile、fit 生命周期。'],
  ['5', '单步训练', 'batch 解包为稀疏特征、outcome、score、bucket indices，模型输出 scorenet。'],
  ['6', 'loss', '把 score 与预测都映射到胜率空间，再按 lambda 混合搜索分与对局结果，计算加权幂次误差。'],
  ['7', '产物', 'logs/ 下生成 TensorBoard、CSV、checkpoint，以及 training_finished 标记；serialize.py 可导出 .nnue。'],
];

const readingRows = [
  ['1', 'README.md 与 docs/', '先建立 NNUE、特征和数据格式背景。'],
  ['2', 'config.py 与 model/config.py', '理解所有训练参数如何嵌套和从命令行进入程序。'],
  ['3', 'train.py', '掌握从参数解析到 Trainer.fit 的主调用链。'],
  ['4', 'model/lightning_module.py', '看清 batch 语义、loss、optimizer 和 Lightning hooks。'],
  ['5', 'model/model.py 与 model/modules/', '理解真实网络结构、特征变换和后半 dense stack。'],
  ['6', 'data_loader/dataset.py 与 stream.py', '理解 .binpack 到 tensor batch 的路径。'],
  ['7', 'serialize.py 与 cross_check_eval.py', '理解训练后如何导出、验证和对接引擎。'],
];

export default function NNUEPytorchArchitecture() {
  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <H1>NNUE PyTorch 训练项目架构</H1>
        <Text>
          这个仓库的核心是一个 PyTorch Lightning 训练管线：用原生 C++ loader 高速读取稀疏棋局特征，用 NNUEModel 定义网络，用 NNUE LightningModule 封装 loss 与优化器，再由 train.py 统一编排训练、日志和 checkpoint。
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value="train.py" label="训练入口" />
        <Stat value="data_loader/" label="数据加载模块" />
        <Stat value="model/" label="模型与训练逻辑" />
        <Stat value=".ckpt / .nnue" label="主要产物" />
      </Grid>

      <Divider />

      <H2>主链路</H2>
      <Table
        headers={['步骤', '阶段', '职责']}
        rows={flowRows}
      />

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>数据侧重点</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Text>训练数据不是普通 PyTorch Dataset 独立完成，而是 Python Dataset 包装 C++ sparse batch stream。</Text>
              <Text>PyTorch DataLoader 的 batch_size 设为 None，实际 batch 已由 SparseBatchDataset 按配置产出。</Text>
              <Text>FixedNumBatchesDataset 把无限/循环数据流切成固定长度 epoch，并负责 pin memory 与 CUDA 预取。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>模型侧重点</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Text>NNUEModel 先根据 feature name 选择特征类，构造 feature transformer。</Text>
              <Text>前向中按白/黑视角得到特征，拼接 us/them 视角，再进入 LayerStacks。</Text>
              <Text>量化配置同时影响分数缩放和权重裁剪，WeightClippingCallback 会在训练批次前应用约束。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>模块划分</H2>
      <Table
        headers={['模块', '关键路径', '说明']}
        rows={moduleRows}
      />

      <H2>建议阅读顺序</H2>
      <Table
        headers={['顺序', '文件/目录', '目的']}
        rows={readingRows}
      />

      <Divider />

      <Stack gap={8}>
        <H2>一句话总结</H2>
        <Text>
          可以把这个项目理解为四层：配置层负责把命令行变成结构化参数，数据层负责把 .binpack 变成稀疏 tensor batch，模型层负责 NNUE 前向、loss 和优化器，训练编排层负责 Lightning Trainer、日志、checkpoint、导出和分布式辅助。
        </Text>
        <Pill tone="info">{'推荐先读 config.py -> train.py -> model/lightning_module.py -> model/model.py -> data_loader/dataset.py'}</Pill>
      </Stack>
    </Stack>
  );
}

