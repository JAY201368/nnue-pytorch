# Animal Chess 快速开始指南

## 📦 导入库

```python
import animalchess
import animalchess.svg  # 如果需要SVG可视化
```

## 🎮 基础操作

### 1. 创建棋盘

```python
# 创建初始局面
board = animalchess.Board()

# 创建空棋盘
board = animalchess.Board(None)
board.clear()

# 从FEN创建
board = animalchess.Board("l5t/1d3c1/r1p1w1e/7/7/7/E1W1P1R/1C3D1/T5L r 0 1")
```

### 2. 显示棋盘

```python
# ASCII显示
print(board)

# Unicode emoji显示
print(board.unicode())

# 输出：
# 🐯 · ⚠️ 🏠 ⚠️ · 🦁
# · 🐱 · ⚠️ · 🐕 ·
# 🐘 · 🐺 · 🐆 · 🐭
# ... 等等
```

### 3. 获取棋子

```python
# 获取特定位置的棋子
piece = board.piece_at(animalchess.A1)
if piece:
    print(piece.chinese_name())  # 红狮
    print(piece.symbol())         # L
    print(piece.unicode_symbol()) # 🦁
```

### 4. 走法

```python
# 生成所有合法走法
for move in board.generate_pseudo_legal_moves():
    print(move.uci())  # 如 "a1b1"

# 创建走法
move = animalchess.Move.from_uci("a3b3")

# 检查合法性
if board.is_legal(move):
    board.push(move)  # 执行走法

# 撤销走法
last_move = board.pop()
```

### 5. 游戏状态

```python
# 检查游戏是否结束
if board.is_game_over():
    result = board.result()  # "1-0", "0-1", 或 "*"
    print(f"游戏结束: {result}")

# 当前轮到谁
if board.turn == animalchess.RED:
    print("红方走")
else:
    print("黑方走")
```

## 🎯 常用常量

### 颜色
```python
animalchess.RED    # True (红方)
animalchess.BLACK  # False (黑方)
```

### 棋子类型
```python
animalchess.RAT      # 1 - 鼠
animalchess.CAT      # 2 - 猫
animalchess.DOG      # 3 - 狗
animalchess.WOLF     # 4 - 狼
animalchess.LEOPARD  # 5 - 豹
animalchess.TIGER    # 6 - 虎
animalchess.LION     # 7 - 狮
animalchess.ELEPHANT # 8 - 象
```

### 方格
```python
# 方格名称: a1-g9
animalchess.A1, animalchess.D1  # 等等

# 特殊位置
animalchess.RED_DEN      # 红方兽穴 (d1)
animalchess.BLACK_DEN    # 黑方兽穴 (d9)
animalchess.RED_TRAPS    # 红方陷阱 [c1, e1, d2]
animalchess.BLACK_TRAPS  # 黑方陷阱 [c9, e9, d8]
animalchess.RIVER_SQUARES # 河流方格 (不包括D列陆桥)
# 河流：B4-C4, E4-F4, B5-C5, E5-F5, B6-C6, E6-F6
# 注意：D列(d4, d5, d6)是陆桥，所有动物都可以走
```

## 🔥 高级功能

### 检查特殊规则

```python
# 检查能否吃子
can_eat = board.can_capture(from_square, to_square)

# 检查狮虎能否跳河
can_jump = board.can_jump_river(from_square, to_square)

# 获取棋子有效等级（考虑陷阱）
power = board.get_piece_power(square)

# 检查是否在河中
is_in_river = board.is_river(square)

# 检查是否在陷阱中
is_in_trap = board.is_trap(square, for_color=animalchess.RED)
```

### FEN操作

```python
# 获取FEN
fen = board.fen()
print(fen)  # "l5t/1d3c1/r1p1w1e/7/7/7/E1W1P1R/1C3D1/T5L r 0 1"

# 从FEN设置
board.set_fen("l5t/1d3c1/r1p1w1e/7/7/7/E1W1P1R/1C3D1/T5L r 0 1")
```

### 复制棋盘

```python
# 复制当前棋盘状态
board_copy = board.copy()

# 在副本上尝试走法
board_copy.push(move)
if board_copy.is_game_over():
    print("这步棋会导致游戏结束")
```

### SVG可视化

```python
import animalchess.svg

# 生成SVG
svg_data = animalchess.svg.board(board, size=500, coordinates=True)

# 保存到文件
with open("game.svg", "w", encoding="utf-8") as f:
    f.write(svg_data)

# 生成单个棋子的SVG
piece = animalchess.Piece(animalchess.LION, animalchess.RED)
piece_svg = animalchess.svg.piece(piece, size=60)
```

## 💡 实用示例

### 简单AI（随机走法）

```python
import random

def simple_ai_move(board):
    """随机选择一个合法走法"""
    moves = list(board.generate_pseudo_legal_moves())
    if moves:
        return random.choice(moves)
    return None

# 使用
while not board.is_game_over():
    move = simple_ai_move(board)
    if move:
        board.push(move)
        print(board.unicode())
```

### 遍历游戏历史

```python
# 记录走法
moves = []

# 走几步棋
move1 = animalchess.Move.from_uci("a3b3")
board.push(move1)
moves.append(move1)

move2 = animalchess.Move.from_uci("g7f7")
board.push(move2)
moves.append(move2)

# 回退到初始状态
while board.move_stack:
    board.pop()

# 重新执行
for move in moves:
    board.push(move)
```

### 分析局面

```python
def analyze_position(board):
    """分析当前局面"""
    red_pieces = sum(1 for p in board.pieces_dict.values() if p.color == animalchess.RED)
    black_pieces = sum(1 for p in board.pieces_dict.values() if p.color == animalchess.BLACK)
    
    print(f"红方棋子数: {red_pieces}")
    print(f"黑方棋子数: {black_pieces}")
    
    # 统计各类棋子
    for color in [animalchess.RED, animalchess.BLACK]:
        color_name = "红方" if color == animalchess.RED else "黑方"
        pieces_count = {}
        
        for piece in board.pieces_dict.values():
            if piece.color == color:
                name = animalchess.piece_chinese(piece.piece_type)
                pieces_count[name] = pieces_count.get(name, 0) + 1
        
        print(f"\n{color_name}棋子: {pieces_count}")

analyze_position(board)
```

## 🐛 常见问题

### Q: 如何判断鼠吃象？
```python
rat_square = animalchess.A3
elephant_square = animalchess.B3

# 设置棋子
board.set_piece_at(rat_square, animalchess.Piece(animalchess.RAT, animalchess.RED))
board.set_piece_at(elephant_square, animalchess.Piece(animalchess.ELEPHANT, animalchess.BLACK))

# 检查
if board.can_capture(rat_square, elephant_square):
    print("鼠可以吃象！")
```

### Q: 如何检查狮虎跳河？
```python
# 狮/虎在河边
if board.can_jump_river(from_square, to_square):
    print("可以跳过河流")
```

### Q: 如何查看陷阱效果？
```python
square = animalchess.C9  # 黑方陷阱
piece = board.piece_at(square)

if piece:
    normal_power = piece.piece_type
    effective_power = board.get_piece_power(square)
    print(f"原始等级: {normal_power}, 有效等级: {effective_power}")
```

## 📚 更多资源

- 完整文档: `README.md`
- 示例代码: `example.py`
- 运行示例: `python -m animalchess.example`

---

祝你玩得开心！🎮

