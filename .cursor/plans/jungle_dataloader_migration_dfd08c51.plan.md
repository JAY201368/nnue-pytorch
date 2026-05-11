---
name: jungle dataloader migration
overview: 将 `data_loader` 的 C++ 内部从国际象棋 `.binpack` 解析与特征抽取迁移为斗兽棋专用实现，同时保留 Python Dataset、ctypes ABI、batch tensor 结构和训练主链路不变。
todos:
  - id: define-jungle-record
    content: 定义斗兽棋 fixed-size packed record、piece 顺序、result/score/ply 语义和 BINP chunk 读取约定
    status: completed
  - id: add-jungle-types-stream
    content: 新增 `jungle` C++ 棋局表示和 `.binpack` parallel reader，保持现有 stream/fill 模式
    status: completed
  - id: port-feature-extractors
    content: 实现并注册 `JunglePieceSquare`、`JunglePieceTerrain` C++ extractor，确保组合偏移与 Python 特征一致
    status: completed
  - id: adapt-batch-fields-filters
    content: 调整 `SparseBatch::fill_entry`、bucket、skip predicate，移除国际象棋专属依赖
    status: completed
  - id: test-loader-end-to-end
    content: 添加人工 `.binpack` fixture 或测试 writer，验证 sparse indices、batch shape 和基础训练前向
    status: completed
isProject: false
---

# 斗兽棋 Data Loader 迁移方案

## 目标边界

保留 Python 侧数据接口：[`data_loader/dataset.py`](data_loader/dataset.py)、[`data_loader/stream.py`](data_loader/stream.py)、[`data_loader/_native.py`](data_loader/_native.py) 继续返回现有 batch tuple：`us/them`、白/黑 POV sparse indices、values、`outcome`、`score`、`psqt_indices`、`layer_stack_indices`。

C++ 内部新增斗兽棋实现，不再复用国际象棋 `chess::Position`、king bucket、movegen、check/castling/en-passant 等逻辑。现有 [`data_loader/cpp/training_data_loader.cpp`](data_loader/cpp/training_data_loader.cpp) 中的 `IFeatureExtractor`、`ComposedFeatureExtractor`、`SparseBatch`、`FeaturedBatchStream` 结构可以保留，但类型参数和实现切到 `jungle::TrainingDataEntry`。

## 建议数据流

```mermaid
flowchart LR
    binpackFile["Jungle .binpack"] --> chunkReader["BINP chunk reader"]
    chunkReader --> fixedRecordReader["Fixed-size jungle records"]
    fixedRecordReader --> entry["jungle::TrainingDataEntry"]
    entry --> skipFilter["sampling filters"]
    skipFilter --> featureExtractor["JunglePieceSquare+JunglePieceTerrain"]
    featureExtractor --> sparseBatch["SparseBatch ABI"]
    sparseBatch --> pythonTensors["Python tensors"]
```

## 固定记录格式

沿用原 `CompressedTrainingDataFile` 的 `BINP + chunkSize` 容器，但 chunk 内不再使用国际象棋 movelist 续局压缩。新增固定长度记录，建议第一版只承载静态局面和训练标签：

```cpp
struct PackedJungleTrainingDataEntry {
    uint8_t piece_squares[16]; // white 8 + black 8, 0..62, 63 = captured
    uint8_t side_to_move;      // 0 white, 1 black
    int16_t score;
    uint16_t ply;
    int8_t result;             // -1/0/1 from side-to-move or fixed white POV,需在写入侧统一
    uint8_t flags;             // reserved: capture/terminal/format flags
};
```

实现时应加 `static_assert` 固定大小，并在格式注释中明确 piece 顺序与 `result` 语义。piece 顺序必须与 [`docs/feature_redesign.md`](docs/feature_redesign.md) 保持一致：`ELEPHANT, LION, TIGER, PANTHER, WOLF, DOG, CAT, RAT`。

## C++ 模块拆分

新增独立头/实现，避免继续膨胀 [`data_loader/cpp/lib/nnue_training_data_formats.h`](data_loader/cpp/lib/nnue_training_data_formats.h)：

- `data_loader/cpp/lib/jungle_types.h`：定义 `jungle::Color`、`PieceType`、`Position`、square 常量、piece 迭代工具。
- `data_loader/cpp/lib/jungle_binpack.h`：定义 fixed record、record 到 `TrainingDataEntry` 的转换、chunk reader。
- `data_loader/cpp/lib/jungle_training_data_stream.h`：实现 parallel input stream，接口对齐现有 `BasicSfenInputStream` 的 `next/fill/fill_threadsafe/eof` 模式。
- `data_loader/cpp/jungle_feature_extractors.cpp` 或并入 `training_data_loader.cpp` 初期实现：注册 `JunglePieceSquare`、`JunglePieceTerrain` 和组合特征。

## 稀疏索引编码

在 C++ 侧严格复刻 Python 特征公式：

- `JunglePieceSquare`：`index = oriented_square + 63 * (piece_type + 8 * relative_owner)`，`inputs = 1008`，`max_active = 16`。
- `JunglePieceTerrain`：先把 physical square 转成 POV square，再按地形 tag 输出多个 index；`inputs = 128`，`max_active = 48`。
- 组合顺序沿用 [`docs/feature_redesign.md`](docs/feature_redesign.md)：先 `JunglePieceSquare`，后 `JunglePieceTerrain`，第二段自动加 `1008` 偏移。
- 每个 active value 都填 `1.0f`，未用槽保持 `-1` 和 `0.0f`。

## Batch 派生字段

`is_white` 继续表示 side to move 是否为白方。

`outcome` 继续转为 `(result + 1) / 2`，但要先确认写入侧 `result` 是“当前行棋方视角”还是“白方视角”。为了少改 loss，推荐写入/读取后统一成当前行棋方视角。

`psqt_indices` 和 `layer_stack_indices` 原来由国际象棋子力数映射为 `(piece_count - 1) / 4`。斗兽棋第一版建议先用存活棋子数 bucket：`clamp((piece_count - 1) / 2, 0, num_buckets - 1)`，因为最多 16 子，能较均匀覆盖 8 buckets。若后续模型配置改 bucket 数，再把 bucket 计算参数化。

## 过滤策略

第一版只保留与棋种无关的过滤：随机跳过、早期 ply 硬/软过滤、piece count 分布采样、WLD 一致性采样。移除或禁用国际象棋专属过滤：`isCapturingMove()`、`isInCheck()`、`simple_eval()`。

如需要捕获局面过滤，等 fixed record 的 `flags` 或写入侧提供 last move/capture 标志后再恢复。

## 验证计划

新增 C++/Python 联合测试，优先覆盖编码一致性：

- 用几个人工 `PackedJungleTrainingDataEntry` 写入临时 `.binpack`，通过 `SparseBatchDataset` 读取。
- 对照 [`tests/test_jungle_features.py`](tests/test_jungle_features.py) 的 Python 公式，检查白/黑 POV indices、terrain offset、`MAX_ACTIVE_FEATURES=64`。
- 检查 captured piece 不产生活跃特征，黑方 POV square 使用 `62 - square`。
- 检查 batch shape 与 [`model/model.py`](model/model.py) 前向需要一致。

## 风险点

最大风险是 `result` 视角和 score 标尺。如果数据生成器输出的是白方视角结果，而现有 loss 假设 batch 的 `outcome` 与 side-to-move/score 标尺一致，需要在 reader 中翻转。

第二个风险是将 `training_data_loader.cpp` 从 `binpack::TrainingDataEntry` 改为 `jungle::TrainingDataEntry` 会影响 `get_sparse_batch_from_fens` 和 `FenBatchProvider`。建议第一版保留 ABI 名称但把 FEN 调试接口改成斗兽棋 FEN，或者暂时只保证 sparse stream 可用，并在 Python 测试里直接走 `.binpack`。