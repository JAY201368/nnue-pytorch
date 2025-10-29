#!/usr/bin/env python3
"""
Animal Chess (斗兽棋) - 示例代码
演示如何使用animalchess库
"""

import animalchess
import animalchess.svg


def basic_example():
    """基础用法示例"""
    print("=" * 50)
    print("基础用法示例")
    print("=" * 50)
    
    # 创建棋盘
    board = animalchess.Board()
    
    # 显示初始棋盘
    print("\n初始棋盘（Unicode）：")
    print(board.unicode())
    
    print("\n初始棋盘（ASCII）：")
    print(board)
    
    print(f"\nFEN: {board.fen()}")
    print(f"当前轮到: {'白方' if board.turn == animalchess.WHITE else '黑方'}")


def move_example():
    """走法示例"""
    print("\n" + "=" * 50)
    print("走法示例")
    print("=" * 50)
    
    board = animalchess.Board()
    
    # 列出所有合法走法
    print("\n白方的前10个合法走法：")
    moves = list(board.generate_pseudo_legal_moves())
    for i, move in enumerate(moves[:10]):
        from_sq = move.from_square
        to_sq = move.to_square
        piece = board.piece_at(from_sq)
        print(f"{i+1}. {piece.chinese_name()} {animalchess.square_name(from_sq)} -> {animalchess.square_name(to_sq)} ({move.uci()})")
    
    print(f"\n共有 {len(moves)} 个合法走法")
    
    # 执行几步棋
    print("\n执行几步棋：")
    test_moves = ["a3b3", "g7f7", "b3c3"]
    
    for move_uci in test_moves:
        move = animalchess.Move.from_uci(move_uci)
        piece = board.piece_at(move.from_square)
        color_name = "白方" if board.turn == animalchess.WHITE else "黑方"
        
        if board.is_legal(move):
            print(f"\n{color_name}: {piece.chinese_name()} {move.uci()}")
            board.push(move)
            print(board.unicode())
        else:
            print(f"非法走法: {move_uci}")


def special_rules_example():
    """特殊规则示例"""
    print("\n" + "=" * 50)
    print("特殊规则示例")
    print("=" * 50)
    
    # 示例1：鼠吃象
    print("\n1. 鼠吃象规则")
    board = animalchess.Board()
    board.clear()
    board.set_piece_at(animalchess.D5, animalchess.Piece(animalchess.RAT, animalchess.WHITE))
    board.set_piece_at(animalchess.D6, animalchess.Piece(animalchess.ELEPHANT, animalchess.BLACK))
    board.turn = animalchess.WHITE
    
    print(board.unicode())
    
    move = animalchess.Move.from_uci("d5d6")
    print(f"\n白鼠 d5->d6 (吃黑象): {'合法' if board.is_legal(move) else '非法'}")
    
    if board.can_capture(animalchess.D5, animalchess.D6):
        print("✓ 鼠可以吃象！")
    
    # 示例2：象不能吃鼠
    print("\n2. 象不能吃鼠规则")
    board.clear()
    board.set_piece_at(animalchess.D5, animalchess.Piece(animalchess.ELEPHANT, animalchess.WHITE))
    board.set_piece_at(animalchess.D6, animalchess.Piece(animalchess.RAT, animalchess.BLACK))
    board.turn = animalchess.WHITE
    
    print(board.unicode())
    
    move = animalchess.Move.from_uci("d5d6")
    print(f"\n白象 d5->d6 (吃黑鼠): {'合法' if board.is_legal(move) else '非法'}")
    
    if not board.can_capture(animalchess.D5, animalchess.D6):
        print("✓ 象不能吃鼠！")
    
    # 示例3：陷阱
    print("\n3. 陷阱规则")
    board = animalchess.Board()
    board.clear()
    
    # 在黑方陷阱放一个黑象
    trap_square = animalchess.BLACK_TRAPS[0]  # c9
    board.set_piece_at(trap_square, animalchess.Piece(animalchess.ELEPHANT, animalchess.BLACK))
    # 白鼠在旁边
    board.set_piece_at(animalchess.C8, animalchess.Piece(animalchess.RAT, animalchess.WHITE))
    board.turn = animalchess.WHITE
    
    print(board.unicode())
    print(f"\n黑象在陷阱 {animalchess.square_name(trap_square)} 上")
    print(f"黑象的有效等级: {board.get_piece_power(trap_square)} (原始等级8，陷阱中变为0)")
    
    move = animalchess.Move.from_uci("c8c9")
    print(f"白鼠 c8->c9 (吃陷阱中的黑象): {'合法' if board.is_legal(move) else '非法'}")
    
    if board.can_capture(animalchess.C8, trap_square):
        print("✓ 鼠可以吃陷阱中的象！")


def game_simulation():
    """完整游戏模拟"""
    print("\n" + "=" * 50)
    print("完整游戏模拟")
    print("=" * 50)
    
    board = animalchess.Board()
    move_count = 0
    max_moves = 20
    
    print("\n开始游戏...")
    print(board.unicode())
    
    while not board.is_game_over() and move_count < max_moves:
        moves = list(board.generate_pseudo_legal_moves())
        if not moves:
            break
        
        # 选择第一个合法走法（简单AI）
        move = moves[0]
        piece = board.piece_at(move.from_square)
        color_name = "白方" if board.turn == animalchess.WHITE else "黑方"
        
        print(f"\n第{move_count + 1}步 - {color_name}: {piece.chinese_name()} {move.uci()}")
        board.push(move)
        print(board.unicode())
        
        move_count += 1
    
    if board.is_game_over():
        print(f"\n游戏结束！结果: {board.result()}")
        winner = "白方获胜" if board.result() == "1-0" else "黑方获胜" if board.result() == "0-1" else "平局"
        print(winner)
    else:
        print(f"\n模拟{max_moves}步后停止")


def svg_example():
    """SVG可视化示例"""
    print("\n" + "=" * 50)
    print("SVG可视化示例")
    print("=" * 50)
    
    board = animalchess.Board()
    
    # 走几步棋
    moves = ["a3b3", "g7f7", "c3c4"]
    for move_uci in moves:
        move = animalchess.Move.from_uci(move_uci)
        if board.is_legal(move):
            board.push(move)
    
    # 生成SVG
    svg_data = animalchess.svg.board(board, size=500)
    
    # 保存到文件
    filename = "animalchess_example.svg"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg_data)
    
    print(f"\n✓ SVG棋盘已保存到: {filename}")
    print("可以用浏览器打开查看")


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 10 + "斗兽棋 (Animal Chess)" + " " * 17 + "║")
    print("║" + " " * 15 + "示例程序" + " " * 24 + "║")
    print("╚" + "=" * 48 + "╝")
    
    # 运行各个示例
    basic_example()
    move_example()
    special_rules_example()
    game_simulation()
    svg_example()
    
    print("\n" + "=" * 50)
    print("示例运行完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

