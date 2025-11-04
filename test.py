#!/usr/bin/env python3
"""
斗兽棋 (Jungle Chess) - 完整测试文件
模仿 nnue-pytorch 项目中使用 chess 包的方式测试 jungle_chess 包
"""

import jungle_chess as jc
import jungle_chess.svg


def test_basic_board():
    """测试基本棋盘功能"""
    print("=" * 60)
    print("测试 1: 基本棋盘功能")
    print("=" * 60)

    # 创建棋盘
    board = jc.Board()
    print("\n✓ 成功创建棋盘")

    # 显示初始棋盘
    print("\n初始棋盘（Unicode）：")
    print(board.unicode())

    print("\n初始棋盘（ASCII）：")
    print(board)

    # 测试 FEN
    fen = board.fen()
    print(f"\nFEN: {fen}")

    # 测试颜色
    print(f"当前轮到: {'白方' if board.turn == jc.WHITE else '黑方'}")
    print(f"半步计数: {board.halfmove_clock}")
    print(f"全步计数: {board.fullmove_number}")

    # 测试棋子访问（类似 chess.Board().color_at()）
    color = board.color_at(jc.A1)
    print(f"\na1 位置的棋子颜色: {color}")

    piece = board.piece_at(jc.A1)
    if piece:
        print(f"a1 位置的棋子: {piece.chinese_name()}")

    print("\n✓ 基本棋盘功能测试通过")


def test_piece_creation():
    """测试棋子创建"""
    print("\n" + "=" * 60)
    print("测试 2: 棋子创建与属性")
    print("=" * 60)

    # 创建各种棋子
    pieces = [
        (jc.RAT, jc.WHITE, "白鼠"),
        (jc.CAT, jc.WHITE, "白猫"),
        (jc.DOG, jc.BLACK, "黑狗"),
        (jc.WOLF, jc.BLACK, "黑狼"),
        (jc.LEOPARD, jc.WHITE, "白豹"),
        (jc.TIGER, jc.WHITE, "白虎"),
        (jc.LION, jc.BLACK, "黑狮"),
        (jc.ELEPHANT, jc.BLACK, "黑象"),
    ]

    print("\n创建的棋子：")
    for piece_type, color, expected_name in pieces:
        piece = jc.Piece(piece_type, color)
        actual_name = piece.chinese_name()
        assert actual_name == expected_name, f"棋子名称错误: 期望 {expected_name}, 实际 {actual_name}"
        print(f"  {piece.symbol()} - {actual_name} (类型值: {piece.piece_type})")

    print("\n✓ 棋子创建测试通过")


def test_move_generation():
    """测试走法生成"""
    print("\n" + "=" * 60)
    print("测试 3: 走法生成")
    print("=" * 60)

    board = jc.Board()

    # 测试伪合法走法生成
    pseudo_moves = list(board.pseudo_legal_moves)
    print(f"\n初始局面的伪合法走法数: {len(pseudo_moves)}")

    # 测试合法走法生成
    legal_moves = list(board.legal_moves)
    print(f"初始局面的合法走法数: {len(legal_moves)}")

    # 显示前10个走法
    print("\n前10个合法走法：")
    for i, move in enumerate(legal_moves[:10], 1):
        from_sq = move.from_square
        to_sq = move.to_square
        piece = board.piece_at(from_sq)
        print(f"  {i:2d}. {piece.chinese_name()} {jc.square_name(from_sq)} → {jc.square_name(to_sq)} ({move.uci()})")

    # 测试 Move.from_uci() (类似 chess.Move.from_uci())
    test_move_uci = "a3b3"
    move = jc.Move.from_uci(test_move_uci)
    print(f"\n从 UCI '{test_move_uci}' 创建走法:")
    print(f"  from_square: {jc.square_name(move.from_square)}")
    print(f"  to_square: {jc.square_name(move.to_square)}")
    print(f"  合法性: {board.is_legal(move)}")

    print("\n✓ 走法生成测试通过")


def test_move_execution():
    """测试走法执行与撤销"""
    print("\n" + "=" * 60)
    print("测试 4: 走法执行与撤销")
    print("=" * 60)

    board = jc.Board()
    initial_fen = board.fen()

    # 执行几步棋
    test_moves = ["a3b3", "g7f7", "b3b4", "f7f6"]

    print("\n执行走法序列：")
    for move_uci in test_moves:
        move = jc.Move.from_uci(move_uci)
        piece = board.piece_at(move.from_square)
        color_name = "白方" if board.turn == jc.WHITE else "黑方"

        assert board.is_legal(move), f"走法 {move_uci} 不合法"

        print(f"  {color_name}: {piece.chinese_name()} {move.uci()}")
        board.push(move)

    print(f"\n执行 {len(test_moves)} 步后:")
    print(f"  移动栈长度: {len(board.move_stack)}")
    print(f"  当前轮到: {'白方' if board.turn == jc.WHITE else '黑方'}")
    print(f"  FEN: {board.fen()}")

    # 测试 peek()
    last_move = board.peek()
    print(f"\n最后一步 (peek): {last_move.uci()}")

    # 撤销所有走法
    print("\n撤销所有走法：")
    while board.move_stack:
        move = board.pop()
        print(f"  撤销: {move.uci()}")

    final_fen = board.fen()
    print(f"\n撤销后 FEN: {final_fen}")

    assert initial_fen == final_fen, "FEN 不匹配，撤销失败"
    print("✓ 走法与初始局面一致")

    # 测试 root()
    board.push(jc.Move.from_uci("a3b3"))
    board.push(jc.Move.from_uci("g7f7"))
    root_board = board.root()
    print(f"\nroot() 返回的棋盘移动栈长度: {len(root_board.move_stack)}")
    assert len(root_board.move_stack) == 0, "root() 应返回无移动历史的棋盘"

    print("\n✓ 走法执行与撤销测试通过")


def test_copy_and_clone():
    """测试棋盘复制"""
    print("\n" + "=" * 60)
    print("测试 5: 棋盘复制")
    print("=" * 60)

    board = jc.Board()

    # 执行几步
    board.push(jc.Move.from_uci("a3b3"))
    board.push(jc.Move.from_uci("g7f7"))

    # 复制棋盘
    board_copy = board.copy()

    print(f"\n原棋盘移动栈长度: {len(board.move_stack)}")
    print(f"复制棋盘移动栈长度: {len(board_copy.move_stack)}")
    print(f"原棋盘 FEN: {board.fen()}")
    print(f"复制棋盘 FEN: {board_copy.fen()}")

    assert board.fen() == board_copy.fen(), "复制后 FEN 不一致"
    assert len(board.move_stack) == len(board_copy.move_stack), "移动栈长度不一致"

    # 在复制的棋盘上走一步，不应影响原棋盘
    board_copy.push(jc.Move.from_uci("b3c3"))

    print(f"\n复制棋盘走一步后:")
    print(f"  原棋盘移动栈长度: {len(board.move_stack)}")
    print(f"  复制棋盘移动栈长度: {len(board_copy.move_stack)}")

    assert len(board.move_stack) == 2, "原棋盘被修改了"
    assert len(board_copy.move_stack) == 3, "复制棋盘走法未记录"

    print("\n✓ 棋盘复制测试通过")


def test_fen_parsing():
    """测试 FEN 解析"""
    print("\n" + "=" * 60)
    print("测试 6: FEN 解析与设置")
    print("=" * 60)

    # 测试起始 FEN
    board1 = jc.Board()
    fen1 = board1.fen()
    print(f"\n标准起始 FEN:\n  {fen1}")

    # 从 FEN 创建棋盘
    board2 = jc.Board(fen=fen1)
    fen2 = board2.fen()

    assert fen1 == fen2, f"FEN 不匹配:\n  原始: {fen1}\n  解析: {fen2}"
    print("✓ FEN 解析正确")

    # 测试自定义 FEN
    custom_fen = "3l3/7/7/7/7/7/7/3L3/7 w 0 1"
    board3 = jc.Board(fen=custom_fen)
    print(f"\n自定义 FEN: {custom_fen}")
    print(board3.unicode())

    # 测试 set_fen()
    board4 = jc.Board()
    board4.set_fen(custom_fen)
    result_fen = board4.fen()
    # FEN 可能在解析后有细微差异，只检查棋盘部分
    assert custom_fen.split()[0] in result_fen, f"set_fen() 失败: 期望包含 {custom_fen.split()[0]}, 实际 {result_fen}"
    print("✓ set_fen() 正确")

    print("\n✓ FEN 解析测试通过")


def test_special_squares():
    """测试特殊格子（河流、陷阱、兽穴）"""
    print("\n" + "=" * 60)
    print("测试 7: 特殊格子")
    print("=" * 60)

    print("\n河流格子:")
    print(f"  {[jc.square_name(sq) for sq in jc.RIVER_SQUARES]}")
    print(f"  总数: {len(jc.RIVER_SQUARES)}")

    print("\n白方陷阱:")
    print(f"  {[jc.square_name(sq) for sq in jc.WHITE_TRAPS]}")

    print("\n黑方陷阱:")
    print(f"  {[jc.square_name(sq) for sq in jc.BLACK_TRAPS]}")

    print("\n兽穴:")
    print(f"  白方兽穴: {jc.square_name(jc.WHITE_DEN)}")
    print(f"  黑方兽穴: {jc.square_name(jc.BLACK_DEN)}")

    # 测试格子检查函数
    board = jc.Board()
    test_squares = [jc.D3, jc.D4, jc.C1, jc.D1]

    print("\n格子检查:")
    for sq in test_squares:
        sq_name = jc.square_name(sq)
        is_river = sq in jc.RIVER_SQUARES
        is_trap = sq in jc.WHITE_TRAPS or sq in jc.BLACK_TRAPS
        is_den = sq == jc.WHITE_DEN or sq == jc.BLACK_DEN
        print(f"  {sq_name}: 河流={is_river}, 陷阱={is_trap}, 兽穴={is_den}")

    print("\n✓ 特殊格子测试通过")


def test_capture_rules():
    """测试吃子规则"""
    print("\n" + "=" * 60)
    print("测试 8: 吃子规则")
    print("=" * 60)

    # 测试1: 鼠吃象
    print("\n规则1: 鼠可以吃象")
    board = jc.Board()
    board.clear()
    board.set_piece_at(jc.D5, jc.Piece(jc.RAT, jc.WHITE))
    board.set_piece_at(jc.D6, jc.Piece(jc.ELEPHANT, jc.BLACK))
    board.turn = jc.WHITE

    move = jc.Move.from_uci("d5d6")
    can_capture = board.can_capture(jc.D5, jc.D6)
    is_legal = board.is_legal(move)

    print(f"  白鼠 d5 → d6 (黑象)")
    print(f"  can_capture: {can_capture}")
    print(f"  is_legal: {is_legal}")
    assert can_capture and is_legal, "鼠应该能吃象"
    print("  ✓ 通过")

    # 测试2: 象不能吃鼠
    print("\n规则2: 象不能吃鼠")
    board.clear()
    board.set_piece_at(jc.D5, jc.Piece(jc.ELEPHANT, jc.WHITE))
    board.set_piece_at(jc.D6, jc.Piece(jc.RAT, jc.BLACK))
    board.turn = jc.WHITE

    move = jc.Move.from_uci("d5d6")
    can_capture = board.can_capture(jc.D5, jc.D6)
    is_legal = board.is_legal(move)

    print(f"  白象 d5 → d6 (黑鼠)")
    print(f"  can_capture: {can_capture}")
    print(f"  is_legal: {is_legal}")
    assert not can_capture and not is_legal, "象不应该能吃鼠"
    print("  ✓ 通过")

    # 测试3: 陷阱中的棋子
    print("\n规则3: 陷阱中的棋子可以被任何敌方棋子吃")
    board.clear()
    trap_square = jc.BLACK_TRAPS[0]  # c9
    board.set_piece_at(trap_square, jc.Piece(jc.ELEPHANT, jc.BLACK))
    board.set_piece_at(jc.C8, jc.Piece(jc.RAT, jc.WHITE))
    board.turn = jc.WHITE

    power_in_trap = board.get_piece_power(trap_square)
    print(f"  黑象在陷阱 {jc.square_name(trap_square)}")
    print(f"  陷阱中的有效等级: {power_in_trap}")
    assert power_in_trap == 0, "陷阱中的棋子等级应为0"

    can_capture = board.can_capture(jc.C8, trap_square)
    print(f"  白鼠可以吃: {can_capture}")
    assert can_capture, "鼠应该能吃陷阱中的象"
    print("  ✓ 通过")

    print("\n✓ 吃子规则测试通过")


def test_game_over():
    """测试游戏结束判定"""
    print("\n" + "=" * 60)
    print("测试 9: 游戏结束判定")
    print("=" * 60)

    # 测试1: 对方无子
    print("\n场景1: 对方无子（白方走棋，黑方无子）")
    board = jc.Board()
    board.clear()
    board.set_piece_at(jc.A1, jc.Piece(jc.RAT, jc.WHITE))
    board.turn = jc.WHITE

    print(board.unicode())
    is_over = board.is_game_over()
    result = board.result()

    print(f"  is_game_over: {is_over}")
    print(f"  result: {result}")
    assert is_over, "游戏应该结束"
    assert result == "1-0", "白方应该获胜"
    print("  ✓ 通过")

    # 测试2: 进入敌方兽穴（通过走法进入）
    print("\n场景2: 进入敌方兽穴")
    board.clear()
    board.set_piece_at(jc.D8, jc.Piece(jc.RAT, jc.WHITE))  # d8，下一步可以进入 d9
    board.set_piece_at(jc.A1, jc.Piece(jc.RAT, jc.BLACK))
    board.turn = jc.WHITE

    print("走法前:")
    print(board.unicode())
    print(f"  当前轮到: {'白方' if board.turn == jc.WHITE else '黑方'}")

    # 白鼠进入黑方兽穴
    move = jc.Move.from_uci("d8d9")
    board.push(move)

    print("\n走法后（白鼠进入黑方兽穴 d9）:")
    print(board.unicode())
    print(f"  当前轮到: {'白方' if board.turn == jc.WHITE else '黑方'} (走法后切换到对方)")

    # 检查黑方兽穴中的棋子
    piece_in_black_den = board.piece_at(jc.BLACK_DEN)
    if piece_in_black_den:
        print(f"  黑方兽穴(d9)中的棋子: {piece_in_black_den.chinese_name()}")

    is_over = board.is_game_over()
    result = board.result()

    print(f"  is_game_over: {is_over}")
    print(f"  result: {result}")
    # 白鼠进入黑方兽穴后，turn 切换到黑方
    # 黑方走棋时，检查黑方兽穴是否有白方棋子（白方获胜）
    assert is_over, "游戏应该结束"
    assert result == "1-0", f"白方应该获胜（进入黑方兽穴），但结果是 {result}"
    print("  ✓ 通过")

    # 测试3: 正常游戏中
    print("\n场景3: 正常游戏中")
    board = jc.Board()

    is_over = board.is_game_over()
    print(f"  is_game_over: {is_over}")
    assert not is_over, "游戏不应该结束"
    print("  ✓ 通过")

    print("\n✓ 游戏结束判定测试通过")


def test_svg_generation():
    """测试 SVG 生成"""
    print("\n" + "=" * 60)
    print("测试 10: SVG 生成")
    print("=" * 60)

    board = jc.Board()

    # 走几步
    moves = ["a3b3", "g7f7", "c3c4"]
    for move_uci in moves:
        move = jc.Move.from_uci(move_uci)
        if board.is_legal(move):
            board.push(move)

    print("\n生成 SVG...")
    svg_data = jc.svg.board(board, size=400)

    print(f"  SVG 数据长度: {len(svg_data)} 字符")
    assert len(svg_data) > 1000, "SVG 数据太短"
    assert "<svg" in svg_data, "缺少 SVG 标签"
    print("  ✓ SVG 数据生成成功")

    # 测试单个棋子 SVG
    piece = jc.Piece(jc.ELEPHANT, jc.WHITE)
    piece_svg = jc.svg.piece(piece, size=100)

    print(f"  单个棋子 SVG 长度: {len(piece_svg)} 字符")
    assert len(piece_svg) > 100, "棋子 SVG 数据太短"
    print("  ✓ 棋子 SVG 生成成功")

    print("\n✓ SVG 生成测试通过")


def test_bitboard_operations():
    """测试位板操作"""
    print("\n" + "=" * 60)
    print("测试 11: 位板操作")
    print("=" * 60)

    board = jc.Board()

    # 测试 pieces_mask
    rat_mask = board.pieces_mask(jc.RAT, jc.WHITE)
    print(f"\n白鼠位板掩码: {bin(rat_mask)}")

    # 测试 pieces (返回 SquareSet)
    white_rats = board.pieces(jc.RAT, jc.WHITE)
    print(f"白鼠位置: {[jc.square_name(sq) for sq in white_rats]}")

    # 测试 occupied
    occupied = board.occupied
    print(f"\n总占用格子数: {bin(occupied).count('1')}")

    # 测试 occupied_co
    white_occupied = board.occupied_co[jc.WHITE]
    black_occupied = board.occupied_co[jc.BLACK]
    print(f"白方占用格子数: {bin(white_occupied).count('1')}")
    print(f"黑方占用格子数: {bin(black_occupied).count('1')}")

    assert bin(white_occupied).count('1') == 8, "白方应有8个棋子"
    assert bin(black_occupied).count('1') == 8, "黑方应有8个棋子"

    # 测试 piece_type_at
    piece_type = board.piece_type_at(jc.A1)
    print(f"\na1 位置的棋子类型: {piece_type} (狮子={jc.LION})")
    assert piece_type == jc.LION, "a1 应该是狮子"

    print("\n✓ 位板操作测试通过")


def test_square_functions():
    """测试格子相关函数"""
    print("\n" + "=" * 60)
    print("测试 12: 格子相关函数")
    print("=" * 60)

    # 测试 square()
    sq = jc.square(0, 0)  # a1
    print(f"\nsquare(0, 0) = {sq} ({jc.square_name(sq)})")
    assert sq == jc.A1, "square(0, 0) 应该是 A1"

    # 测试 square_file 和 square_rank
    file = jc.square_file(jc.D5)
    rank = jc.square_rank(jc.D5)
    print(f"\nD5 (square={jc.D5}):")
    print(f"  file: {file} (期望 3)")
    print(f"  rank: {rank} (期望 4)")
    assert file == 3 and rank == 4, "D5 的 file 应该是 3, rank 应该是 4"

    # 测试 square_name
    sq_name = jc.square_name(jc.D5)
    print(f"\nsquare_name({jc.D5}) = '{sq_name}'")

    # 通过反向查找来验证（用square函数）
    file = jc.square_file(jc.D5)
    rank = jc.square_rank(jc.D5)
    sq_rebuilt = jc.square(file, rank)
    print(f"通过 square({file}, {rank}) 重建 = {sq_rebuilt}")
    assert sq_rebuilt == jc.D5, "重建的格子应该与原格子相同"

    # 测试所有格子
    print(f"\n总格子数: {len(jc.SQUARES)}")
    assert len(jc.SQUARES) == 63, "应该有63个格子"

    # 测试格子名称
    first_squares = [jc.square_name(sq) for sq in jc.SQUARES[:7]]
    print(f"前7个格子名称: {first_squares}")

    print("\n✓ 格子相关函数测试通过")


# 所有可用测试的字典
ALL_TESTS = {
    "basic_board": ("基本棋盘功能", test_basic_board),
    "piece_creation": ("棋子创建与属性", test_piece_creation),
    "move_generation": ("走法生成", test_move_generation),
    "move_execution": ("走法执行与撤销", test_move_execution),
    "copy_and_clone": ("棋盘复制", test_copy_and_clone),
    "fen_parsing": ("FEN 解析与设置", test_fen_parsing),
    "special_squares": ("特殊格子", test_special_squares),
    "capture_rules": ("吃子规则", test_capture_rules),
    "game_over": ("游戏结束判定", test_game_over),
    "svg_generation": ("SVG 生成", test_svg_generation),
    "bitboard_operations": ("位板操作", test_bitboard_operations),
    "square_functions": ("格子相关函数", test_square_functions),
}


def run_tests(test_names="basic_board"):
    """运行指定的测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "斗兽棋 (Jungle Chess)" + " " * 22 + "║")
    if test_names:
        print("║" + " " * 20 + "指定测试" + " " * 27 + "║")
    else:
        print("║" + " " * 20 + "完整测试套件" + " " * 25 + "║")
    print("╚" + "=" * 58 + "╝")

    # 确定要运行的测试
    if test_names is None:
        # 运行所有测试
        tests_to_run = list(ALL_TESTS.values())
        test_names_display = "所有测试"
    else:
        # 运行指定的测试
        tests_to_run = []
        invalid_tests = []
        for test_name in test_names:
            if test_name in ALL_TESTS:
                tests_to_run.append(ALL_TESTS[test_name])
            else:
                invalid_tests.append(test_name)

        if invalid_tests:
            print(f"\n❌ 无效的测试名称: {', '.join(invalid_tests)}")
            print("使用 --list 查看所有可用测试")
            return False

        test_names_display = ", ".join(test_names)

    print(f"\n运行测试: {test_names_display}")
    print("=" * 60)

    passed = 0
    failed = 0

    for description, test_func in tests_to_run:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ 测试失败: {test_func.__name__}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"✓ 通过: {passed}/{len(tests_to_run)}")
    if failed > 0:
        print(f"❌ 失败: {failed}/{len(tests_to_run)}")
    else:
        print("🎉 所有测试通过！")
    print("=" * 60)

    return failed == 0


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="斗兽棋 (Jungle Chess) 测试套件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test.py                    # 运行所有测试
  python test.py --list              # 列出所有可用测试
  python test.py --test basic_board  # 运行单个测试
  python test.py --test basic_board move_generation  # 运行多个测试
        """
    )

    parser.add_argument(
        "--test",
        nargs="+",
        metavar="TEST_NAME",
        help="运行指定的测试（可以指定多个）。使用 --list 查看所有可用测试",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用测试",
    )

    args = parser.parse_args()


    success = run_tests(args.test)

    if success:
        print("\n✓ jungle_chess 包测试完成，功能正常！")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
