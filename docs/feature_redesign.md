# 斗兽棋 NNUE 特征重设计

第一阶段采用静态局面特征，沿用原项目的 `InputFeature + ComposedFeatureTransformer` 框架。特征集默认设计为：

```text
JunglePieceSquare+JunglePieceTerrain
```

斗兽棋没有王，因此不沿用国际象棋 `HalfKAv2_hm^` 的 king bucket。当前主干改为“相对视角的棋子-格子特征”，再叠加一个轻量地形泛化块。历史相关规则，例如 7-3、17-5、重复局面和无进展计数，暂不进入网络输入，由规则/数据生成侧处理。

## 坐标与视角

棋盘为 `7 x 9 = 63` 格，内部 square 编号按：

```text
square = rank * 7 + file
```

其中 `file` 为 `0..6`，`rank` 为 `0..8`。

网络输入采用 POV 相对视角。白方视角保持原 square，黑方视角做 180 度旋转：

```text
oriented_square = square                      # white POV
oriented_square = 62 - square                 # black POV
```

这样每个 POV 下，己方兽穴总是在 oriented rank 0，敌方兽穴总是在 oriented rank 8。

相对阵营定义为：

```text
relative_owner = 0  # POV 己方棋子
relative_owner = 1  # POV 敌方棋子
```

动物类型顺序固定为：

```text
0 ELEPHANT
1 LION
2 TIGER
3 PANTHER
4 WOLF
5 DOG
6 CAT
7 RAT
```

## `JunglePieceSquare`

路径：`model/modules/features/jungle_piece_square.py`

这是必选核心块，用于表达每枚棋子在 POV 相对棋盘上的位置。

- `FEATURE_NAME = "JunglePieceSquare"`
- `INPUT_FEATURE_NAME = "JunglePieceSquare"`
- 棋子平面：`2 x 8 = 16`
- 输入维度：`63 x 16 = 1008`
- `MAX_ACTIVE_FEATURES = 16`
- `NUM_REAL_FEATURES = 1008`

索引公式：

```text
index = oriented_square + 63 * (piece_type + 8 * relative_owner)
```

PSQT 初始化使用简化子力值：

```text
ELEPHANT = 800
LION     = 700
TIGER    = 600
PANTHER  = 500
WOLF     = 400
DOG      = 300
CAT      = 200
RAT      = 100
```

POV 己方棋子写入正值，POV 敌方棋子写入负值，并按 `1 / nnue2score` 缩放后填入 PSQT 输出列。

## `JunglePieceTerrain`

路径：`model/modules/features/jungle_piece_terrain.py`

这是第一阶段的增强块，用于把固定地形与“哪类棋子站在该地形上”绑定。纯地形常量不进入输入，因为斗兽棋棋盘地形固定；只有与棋子绑定后才提供局面信息。

- `FEATURE_NAME = "JunglePieceTerrain"`
- `INPUT_FEATURE_NAME = "JunglePieceTerrain"`
- 特征形态：`相对阵营 x 动物类型 x 地形标签`
- 地形标签数：`8`
- 输入维度：`2 x 8 x 8 = 128`
- 每枚棋子最多 3 个地形标签
- `MAX_ACTIVE_FEATURES = 16 x 3 = 48`
- `NUM_REAL_FEATURES = 128`

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

地形坐标均以 POV oriented square 表示：

```text
OWN_DEN          = (file=3, rank=0)
ENEMY_DEN        = (file=3, rank=8)

OWN_TRAPS        = (2,0), (4,0), (3,1)
ENEMY_TRAPS      = (2,8), (4,8), (3,7)

WATER            = files 1,2,4,5 x ranks 3,4,5
```

每个棋子的地形索引：

```text
plane = piece_type + 8 * relative_owner
index = 8 * plane + terrain_tag
```

`JunglePieceTerrain` 的 PSQT 输出列初始化为 0，只让训练学习地形对估值的修正。

## 当前组合维度

当前第一阶段组合：

```text
JunglePieceSquare+JunglePieceTerrain
```

组合后：

```text
NUM_INPUTS          = 1008 + 128 = 1136
MAX_ACTIVE_FEATURES = 16 + 48   = 64
NUM_REAL_FEATURES   = 1136
```

组合顺序很重要。数据编码器必须先输出 `JunglePieceSquare` 的索引块，再输出 `JunglePieceTerrain` 的索引块；第二个块的索引需要按组合特征的偏移量加上 `JunglePieceSquare.NUM_INPUTS`。

## 预留扩展

代码中预留：

```text
JUNGLE_RESERVED_FEATURES = ("JungleMobilityAndGoal", "JungleThreats")
```

未来推荐组合：

```text
JunglePieceSquare+JunglePieceTerrain+JungleMobilityAndGoal+JungleThreats
```

`JungleMobilityAndGoal` 可编码：

- 每枚棋子的合法步数 bucket。
- 是否下一步可进入敌方兽穴。
- 到敌方兽穴的距离 bucket。
- 是否守在己方兽穴/陷阱附近。
- 狮虎是否有可用跳河。
- 鼠是否在水中、是否阻挡跳河路径。

`JungleThreats` 可编码：

- 当前是否能吃敌子。
- 当前是否被敌子可吃。
- 是否被己方保护。
- 敌子是否在我方陷阱中。
- 我方棋子是否在敌方陷阱中。
- 鼠象特殊克制、水中鼠限制、狮虎跳吃等规则侧判定结果。