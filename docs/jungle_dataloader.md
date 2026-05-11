# 斗兽棋 Data Loader 设计与迁移说明

本文记录斗兽棋训练数据加载层的当前设计。目标是在尽量保留原 NNUE 训练框架的前提下，将 `data_loader` 的 C++ 数据流从国际象棋 `.binpack` 迁移为斗兽棋 `.binpack`，并把局面转成 `JunglePieceSquare+JunglePieceTerrain` 所需的稀疏特征索引。

## 设计目标

Python 侧接口保持不变：

```text
SparseBatchDataset
  -> SparseBatchProvider
  -> C++ sparse batch stream
  -> SparseBatch.get_tensors()
```

训练侧仍接收同一组张量：

```text
us
them
white_indices
white_values
black_indices
black_values
outcome
score
psqt_indices
layer_stack_indices
```

C++ 内部不再复用国际象棋的 `chess::Position`、king bucket、movegen、将军、王车易位、en-passant 等逻辑，而是切换到斗兽棋专用的 `jungle::TrainingDataEntry`。

## 数据流

```mermaid
flowchart LR
    binpackFile["Jungle .binpack"] --> chunkReader["BINP chunk reader"]
    chunkReader --> fixedRecordReader["Fixed-size jungle records"]
    fixedRecordReader --> entry["jungle::TrainingDataEntry"]
    entry --> skipFilter["sampling filters"]
    skipFilter --> featureExtractor["JunglePieceSquare+JunglePieceTerrain"]
    featureExtractor --> sparseBatch["SparseBatch ABI"]
    sparseBatch --> pytorchTensors["PyTorch tensors"]
```

对应实现路径：

- `data_loader/cpp/lib/jungle_types.h`：斗兽棋颜色、棋子类型、棋盘坐标、`Position`、`TrainingDataEntry`。
- `data_loader/cpp/lib/jungle_binpack.h`：斗兽棋 fixed record、`BINP` chunk 读取、pack/unpack。
- `data_loader/cpp/lib/nnue_training_data_stream.h`：保留原 stream 入口名称，内部读取斗兽棋 `.binpack`。
- `data_loader/cpp/training_data_loader.cpp`：斗兽棋特征抽取、组合特征、`SparseBatch` 填充、skip predicate。
- `data_loader/cpp/training_data_loader_abi.cpp`：保持 C ABI，`get_sparse_batch_from_fens` 改为斗兽棋 FEN 调试入口。

## `.binpack` 文件格式

斗兽棋 `.binpack` 沿用原项目的 chunk 容器：

```text
4 bytes magic: "BINP"
4 bytes little-endian uint32 chunk_size
chunk payload
```

每个 chunk 的 payload 是若干个固定长度 `PackedJungleTrainingDataEntry` 连续排列。当前每条记录固定为 `23` 字节。

```cpp
struct PackedJungleTrainingDataEntry {
    uint8_t piece_squares[16];
    uint8_t side_to_move;
    int16_t score;
    uint16_t ply;
    int8_t result;
    uint8_t flags;
};
```

字段语义：

- `piece_squares[16]`：前 8 个为白方棋子，后 8 个为黑方棋子。
- 棋子顺序固定为 `ELEPHANT, LION, TIGER, PANTHER, WOLF, DOG, CAT, RAT`。
- square 编号为 `square = rank * 7 + file`，范围 `0..62`。
- `63` 表示该棋子已被吃掉，不产生活跃特征。
- `side_to_move`：`0` 表示白方行棋，`1` 表示黑方行棋。
- `score`：训练目标分值，标尺由数据生成器与 loss 配置共同约定。
- `ply`：当前半回合数，用于早期局面过滤。
- `result`：当前实现按 side-to-move 视角解释，`-1` 负、`0` 和、`1` 胜。
- `flags`：预留字段，后续可记录 capture、terminal 或格式版本等信息。

## 稀疏特征编码

当前默认特征集为：

```text
JunglePieceSquare+JunglePieceTerrain
```

编码必须与 `docs/feature_redesign.md` 保持一致。

### `JunglePieceSquare`

```text
oriented_square = square      # white POV
oriented_square = 62 - square # black POV

relative_owner = 0 # POV 己方棋子
relative_owner = 1 # POV 敌方棋子

index = oriented_square + 63 * (piece_type + 8 * relative_owner)
```

维度：

```text
NUM_INPUTS = 63 * 16 = 1008
MAX_ACTIVE_FEATURES = 16
```

### `JunglePieceTerrain`

先按 POV 旋转 physical square，再计算地形标签。

地形标签顺序：

```text
0 TERRAIN_LAND
1 TERRAIN_WATER
2 TERRAIN_OWN_TRAP
3 TERRAIN_ENEMY_TRAP
4 TERRAIN_OWN_DEN
5 TERRAIN_ENEMY_DEN
6 TERRAIN_OWN_DEN_ADJACENT
7 TERRAIN_ENEMY_DEN_ADJACENT
```

索引公式：

```text
plane = piece_type + 8 * relative_owner
index = 8 * plane + terrain_tag
```

维度：

```text
NUM_INPUTS = 2 * 8 * 8 = 128
MAX_ACTIVE_FEATURES = 16 * 3 = 48
```

组合特征按 `JunglePieceSquare` 在前、`JunglePieceTerrain` 在后的顺序输出，因此 terrain 索引进入 batch 时会自动加上 `1008` 的输入偏移。

组合后：

```text
NUM_INPUTS = 1136
MAX_ACTIVE_FEATURES = 64
```

## Batch 字段

`SparseBatch` 的 ABI 布局保持不变。

- `is_white`：当前行棋方是否为白方。
- `outcome`：由 `result` 映射为 `(result + 1) / 2`。
- `score`：直接来自 packed record。
- `white` / `black`：分别是白方 POV、黑方 POV 的 active feature indices。
- `white_values` / `black_values`：当前 active feature value 均为 `1.0f`。
- 未使用的 feature 槽位保持 index `-1`、value `0.0f`。
- `psqt_indices` / `layer_stack_indices`：当前用存活棋子数分桶：

```text
bucket = clamp((piece_count - 1) / 2, 0, 7)
```

这适配默认 `8` 个 PSQT bucket 和 layer-stack bucket。若后续模型配置允许 bucket 数变化，应将该计算改成参数化。

## 过滤策略

第一版只保留与棋种无关的采样过滤：

- `random_fen_skipping`
- `early_fen_skipping`
- `soft_early_fen_skipping`
- `wld_filtered`
- piece count 分布采样

已移除或禁用国际象棋专属过滤：

- `isCapturingMove()`
- `isInCheck()`
- `simple_eval()`

当前 `score_result_prob()` 在斗兽棋侧使用简化 sigmoid，仅用于 `wld_filtered` 采样。真实训练前需要结合斗兽棋引擎分值标尺重新校准。

## 伪数据生成与验证

仓库提供伪数据生成脚本：

```bash
.venv/bin/python scripts/generate_jungle_dummy_binpack.py .pgo/jungle_dummy.binpack --records 256 --seed 20260511
```

该脚本会生成符合当前 fixed record 格式的 pseudo positions，用于验证 dataloader 端到端链路，不保证局面来自合法走子序列。

可用下面的方式快速验证从 `.binpack` 到 PyTorch tensor：

```bash
.venv/bin/python -m pytest tests/test_jungle_data_loader.py
```

完整的斗兽棋特征与 loader 基础测试：

```bash
.venv/bin/python -m pytest tests/test_jungle_features.py tests/test_jungle_data_loader.py
```

## 注意事项

- 数据生成器必须严格按 23 字节 fixed record 写入，并确保 chunk payload 大小是记录大小的整数倍。
- `result` 当前按 side-to-move 视角解释。如果上游输出白方视角结果，需要在写入侧或 reader 中统一转换。
- `score` 的标尺会影响 loss、WLD 过滤和初始训练稳定性；真实数据接入后需要单独校准。
- `FenBatchProvider` 和 `get_sparse_batch_from_fens` 现在只适合作为斗兽棋 FEN 调试路径，不代表真实训练数据格式。
