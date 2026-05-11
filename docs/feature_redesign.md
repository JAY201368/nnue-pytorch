好的，我们先按“静态局面特征”推进，并采用“两层设计”：必选核心块保证尽快跑通，增强块保持模块化、后续可逐块打开。
我建议第一阶段先定成这个方向：**斗兽棋版不做 king bucket，而以“相对视角的棋子-格子特征”为主干**，沿用原项目的 `InputFeature + ComposedFeatureTransformer` 组合方式。

## 核心特征块

必选块：`JunglePieceSquare`

- 棋盘：`7 x 9 = 63` 格。
- 视角：对黑方视角做 180 度旋转，使“己方兽穴、敌方兽穴、前进方向”在两个视角下语义一致。
- 棋子平面：`己方/敌方 x 8 种动物 = 16`。
- 维度：`63 x 16 = 1008`。
- 单视角最大激活数：最多 16 枚棋子，所以 `MAX_ACTIVE_FEATURES = 16`。
- 索引建议：

```text
index = oriented_square + 63 * (piece_type + 8 * relative_owner)
```

这里 `relative_owner = 0` 表示 POV 己方棋子，`1` 表示 POV 敌方棋子。

这个块类似原来的 `HalfKAv2_hm^` 主干，但斗兽棋没有王，所以不做 king bucket。兽穴、陷阱、水域的绝对/相对意义会通过 `oriented_square` 被棋子-格子权重学到。

## 增强特征块

建议预留三个可组合块，但第一版可以只打开第一个或前两个。

`JunglePieceTerrain`

这是对 `PieceSquare` 的粗粒度泛化，帮助模型更快学到水、陷阱、兽穴周边的共同规律。

- 类型可定义为：普通陆地、水、己方陷阱、敌方陷阱、己方兽穴邻近、敌方兽穴邻近等。
- 特征形态：`相对阵营 x 动物类型 x 地形类型`。
- 注意：纯“地形常量”没有意义，因为棋盘地形固定；必须绑定棋子或局面状态。

`JungleMobilityAndGoal`

编码斗兽棋非常重要的“能不能动、离兽穴多近、是否有闯穴威胁”。

可包含：

- 每枚棋子的合法步数 bucket。
- 是否下一步可进入敌方兽穴。
- 到敌方兽穴的曼哈顿距离 bucket。
- 是否守在己方兽穴/陷阱附近。
- 狮虎是否有可用跳河。
- 鼠是否在水中、是否阻挡跳河路径。

`JungleThreats`

对应原项目已有的 `Full_Threats` 思路，但按斗兽棋重写。

可包含：

- 当前是否能吃敌子。
- 当前是否被敌子可吃。
- 是否被己方保护。
- 敌子是否在我方陷阱中。
- 我方棋子是否在敌方陷阱中。
- 鼠象特殊克制、水中鼠限制、狮虎跳吃都应由规则侧判定后编码为威胁特征。

## 推荐第一版 Feature Set

我建议第一阶段目标定为：

```text
JunglePieceSquare+JunglePieceTerrain
```

然后把威胁与机动性作为第二批打开：

```text
JunglePieceSquare+JunglePieceTerrain+JungleMobilityAndGoal+JungleThreats
```

这样既能快速跑通训练管线，又不会把第一版特征设计压得太复杂。历史相关规则，比如 7-3、17-5、重复、无进展，先不进入网络输入，由规则/数据生成侧负责合法性和标签。