# Animal Chess (斗兽棋) Python Library

一个纯Python实现的斗兽棋库，支持走法生成、验证、棋盘表示和SVG可视化。

## 功能特性

- ✅ 完整的斗兽棋规则实现
- ✅ 走法生成与验证
- ✅ 特殊规则支持（鼠吃象、狮虎跳河、陷阱、兽穴等）
- ✅ FEN格式支持
- ✅ SVG棋盘可视化
- ✅ Unicode emoji显示

## 安装

该包已经内置在项目中：
```python
import animalchess
```

## 快速开始

### 创建棋盘

```python
import animalchess

# 创建初始棋盘
board = animalchess.Board()

# 打印棋盘（ASCII）
print(board)

# 打印棋盘（Unicode emoji）
print(board.unicode())
```

### 走法生成

```python
# 生成所有合法走法
for move in board.generate_pseudo_legal_moves():
    print(move.uci())  # 输出类似 "a1b1"

# 检查特定走法是否合法
move = animalchess.Move.from_uci("a1b1")
if board.is_legal(move):
    board.push(move)
```

### 游戏状态

```python
# 检查游戏是否结束
if board.is_game_over():
    print(f"游戏结束，结果: {board.result()}")

# 获取FEN表示
print(board.fen())
```

### SVG可视化

```python
import animalchess.svg

# 生成SVG
svg_data = animalchess.svg.board(board, size=400)

# 保存到文件
with open("board.svg", "w", encoding="utf-8") as f:
    f.write(svg_data)
```

## 棋子说明

### 棋子等级（从高到低）

| 等级 | 棋子 | 符号 | 中文 | Emoji |
|------|------|------|------|-------|
| 8 | Elephant | e/E | 象 | 🐘 |
| 7 | Lion | l/L | 狮 | 🦁 |
| 6 | Tiger | t/T | 虎 | 🐯 |
| 5 | Leopard | p/P | 豹 | 🐆 |
| 4 | Wolf | w/W | 狼 | 🐺 |
| 3 | Dog | d/D | 狗 | 🐕 |
| 2 | Cat | c/C | 猫 | 🐱 |
| 1 | Rat | r/R | 鼠 | 🐭 |

**注意**：大写表示红方，小写表示黑方

### 特殊规则

#### 1. 吃子规则

- **普通规则**：大吃小或同级
- **特殊规则**：
  - 鼠可以吃象（1吃8）
  - 象不能吃鼠

#### 2. 地形规则

**小河 (River) 🌊**
- 河流位置：棋盘中央3行，左右两侧各2列（中间D列是陆桥）
- 只有老鼠能进入河水中
- 狮和虎可以跳过整条河（横向或纵向）
- 如果河中有老鼠，狮虎不能跳过
- 其他动物可以走陆桥（D列）通过河流区域

**陷阱 (Trap) ⚠️**
- 位置：每方兽穴周围3个格子
- 敌方棋子进入陷阱后等级降为0，可被任何己方棋子吃掉

**兽穴 (Den) 🏠**
- 位置：棋盘底部和顶部中央
- 己方棋子不能进入己方兽穴
- 任何棋子进入对方兽穴即获胜

## 完整示例

```python
import animalchess
import animalchess.svg

# 创建游戏
board = animalchess.Board()

print("初始棋盘：")
print(board.unicode())
print()

# 进行几步棋
moves = [
    "a3b3",  # 红鼠向右
    "g7f7",  # 黑鼠向左
    "g1g2",  # 红虎上移
]

for move_uci in moves:
    move = animalchess.Move.from_uci(move_uci)
    if board.is_legal(move):
        piece = board.piece_at(move.from_square)
        print(f"红方走: {piece.chinese_name()} {move.uci()}")
        board.push(move)
        print(board.unicode())
        print()
    else:
        print(f"非法走法: {move_uci}")

# 检查游戏状态
print(f"当前轮到: {'红方' if board.turn == animalchess.RED else '黑方'}")
print(f"游戏是否结束: {board.is_game_over()}")
print(f"FEN: {board.fen()}")

# 生成SVG
svg = animalchess.svg.board(board, size=500)
with open("animalchess_game.svg", "w", encoding="utf-8") as f:
    f.write(svg)
print("棋盘已保存到 animalchess_game.svg")
```

## API参考

### Board类

主要方法：
- `Board(fen=None)` - 创建棋盘
- `reset()` - 重置到初始位置
- `piece_at(square)` - 获取指定位置的棋子
- `set_piece_at(square, piece)` - 设置棋子
- `generate_pseudo_legal_moves()` - 生成伪合法走法
- `is_legal(move)` - 检查走法是否合法
- `push(move)` - 执行走法
- `pop()` - 撤销上一步
- `is_game_over()` - 检查游戏是否结束
- `result()` - 获取游戏结果 ("1-0", "0-1", 或 "*")
- `fen()` - 获取FEN表示
- `set_fen(fen)` - 从FEN设置棋盘
- `unicode()` - 获取Unicode表示
- `copy()` - 复制棋盘

### Piece类

- `Piece(piece_type, color)` - 创建棋子
- `symbol()` - 获取符号（如 "R", "e"）
- `unicode_symbol()` - 获取emoji
- `chinese_name()` - 获取中文名（如 "红鼠"）

### Move类

- `Move(from_square, to_square)` - 创建走法
- `Move.from_uci(uci)` - 从UCI字符串创建
- `uci()` - 转为UCI字符串

### 常量

- `RED`, `BLACK` - 颜色
- `RAT`, `CAT`, `DOG`, `WOLF`, `LEOPARD`, `TIGER`, `LION`, `ELEPHANT` - 棋子类型
- `RIVER_SQUARES` - 河流方格
- `RED_TRAPS`, `BLACK_TRAPS` - 陷阱方格
- `RED_DEN`, `BLACK_DEN` - 兽穴方格

## 棋盘坐标系统

```
   a b c d e f g
9  T . C . W . L  (黑方底线)
8  . . . 穴 . . .
7  E . W . P . R
6  . ~ ~ . ~ ~ .  (河流，D列是陆桥)
5  . ~ ~ . ~ ~ .  (河流，D列是陆桥)
4  . ~ ~ . ~ ~ .  (河流，D列是陆桥)
3  R . P . W . E
2  . . . 穴 . . .
1  L . D . C . T  (红方底线)
```

- 文件（列）：a-g（从左到右）
- 等级（行）：1-9（从下到上）
- 方格名称：如 "d1"（红方兽穴）, "d9"（黑方兽穴）

## 开发与测试

```python
# 测试特殊规则
import animalchess

board = animalchess.Board()

# 测试鼠吃象
board.clear()
board.set_piece_at(animalchess.D5, animalchess.Piece(animalchess.RAT, animalchess.RED))
board.set_piece_at(animalchess.D6, animalchess.Piece(animalchess.ELEPHANT, animalchess.BLACK))
board.turn = animalchess.RED

move = animalchess.Move.from_uci("d5d6")
print(f"鼠吃象合法: {board.is_legal(move)}")  # True
if board.can_capture(animalchess.D5, animalchess.D6):
    board.push(move)
    print("成功！鼠吃掉了象")
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 作者

Animal Chess Library Project

---

**基于python-chess架构开发**

