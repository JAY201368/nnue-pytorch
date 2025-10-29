# Animal Chess (斗兽棋) Library
# Based on python-chess architecture
# Copyright (C) 2024

"""
A pure Python Animal Chess (斗兽棋) library with move generation and validation.
"""

__author__ = "Animal Chess Project"
__version__ = "0.1.0"

import collections
import copy
import enum
import typing
from typing import ClassVar, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Type, TypeVar, Union

# 颜色定义
Color = bool
COLORS = [WHITE, BLACK] = [True, False]  # White = True (白方/底部), Black = False (黑方/顶部)
COLOR_NAMES = ["black", "white"]

# 棋子类型定义 (按等级从低到高)
PieceType = int
PIECE_TYPES = [RAT, CAT, DOG, WOLF, LEOPARD, TIGER, LION, ELEPHANT] = range(1, 9)
PIECE_SYMBOLS = [None, "r", "c", "d", "w", "l", "t", "L", "e"]
PIECE_NAMES = [None, "rat", "cat", "dog", "wolf", "leopard", "tiger", "lion", "elephant"]
PIECE_CHINESE = [None, "鼠", "猫", "狗", "狼", "豹", "虎", "狮", "象"]

def piece_symbol(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_SYMBOLS[piece_type])

def piece_name(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_NAMES[piece_type])

def piece_chinese(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_CHINESE[piece_type])

# Unicode symbols for display
UNICODE_PIECE_SYMBOLS = {
    "E": "🐘", "e": "🐘",  # Elephant
    "L": "🦁", "l": "🦁",  # Lion (狮)
    "T": "🐯", "t": "🐯",  # Tiger
    "P": "🐆", "p": "🐆",  # Leopard
    "W": "🐺", "w": "🐺",  # Wolf
    "D": "🐕", "d": "🐕",  # Dog
    "C": "🐱", "c": "🐱",  # Cat
    "R": "🐭", "r": "🐭",  # Rat
}

# 棋盘位置定义 (7列 x 9行)
FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g"]
RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

Square = int
SQUARES = [
    A1, B1, C1, D1, E1, F1, G1,
    A2, B2, C2, D2, E2, F2, G2,
    A3, B3, C3, D3, E3, F3, G3,
    A4, B4, C4, D4, E4, F4, G4,
    A5, B5, C5, D5, E5, F5, G5,
    A6, B6, C6, D6, E6, F6, G6,
    A7, B7, C7, D7, E7, F7, G7,
    A8, B8, C8, D8, E8, F8, G8,
    A9, B9, C9, D9, E9, F9, G9,
] = range(63)

SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]

# 起始FEN
STARTING_FEN = "l5t/1d3c1/r1p1w1e/7/7/7/E1W1P1R/1C3D1/T5L r 0 1"

def square(file_index: int, rank_index: int) -> Square:
    """Gets a square number by file and rank index."""
    return rank_index * 7 + file_index

def square_file(square: Square) -> int:
    """Gets the file index (0-6)."""
    return square % 7

def square_rank(square: Square) -> int:
    """Gets the rank index (0-8)."""
    return square // 7

def square_name(square: Square) -> str:
    """Gets the name like 'a1'."""
    return SQUARE_NAMES[square]

# 特殊地形定义
# 小河 (River) - 中间的水域，注意D列(中间列)是陆桥不是河
RIVER_SQUARES = [
    B4, C4, E4, F4,    # 第4行 (不包括D4)
    B5, C5, E5, F5,    # 第5行 (不包括D5)
    B6, C6, E6, F6,    # 第6行 (不包括D6)
]

# 陷阱 (Traps)
WHITE_TRAPS = [C1, E1, D2]    # 白方陷阱 (底部)
BLACK_TRAPS = [C9, E9, D8]    # 黑方陷阱 (顶部)

# 兽穴 (Den)
WHITE_DEN = D1    # 白方兽穴 (底部中央)
BLACK_DEN = D9    # 黑方兽穴 (顶部中央)


class Piece:
    """A piece with type and color."""

    def __init__(self, piece_type: PieceType, color: Color) -> None:
        self.piece_type = piece_type
        self.color = color

    def symbol(self) -> str:
        """Gets the symbol for the piece."""
        symbol = piece_symbol(self.piece_type)
        return symbol.upper() if self.color else symbol

    def unicode_symbol(self) -> str:
        """Gets the Unicode emoji for the piece."""
        return UNICODE_PIECE_SYMBOLS.get(self.symbol(), "?")

    def chinese_name(self) -> str:
        """Gets the Chinese name."""
        prefix = "白" if self.color == WHITE else "黑"
        return prefix + piece_chinese(self.piece_type)

    def __repr__(self) -> str:
        return f"Piece.from_symbol({self.symbol()!r})"

    def __str__(self) -> str:
        return self.symbol()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Piece):
            return (self.piece_type, self.color) == (other.piece_type, other.color)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.piece_type, self.color))

    @classmethod
    def from_symbol(cls, symbol: str) -> "Piece":
        """Creates a Piece from a symbol."""
        for i, s in enumerate(PIECE_SYMBOLS):
            if s and s.lower() == symbol.lower():
                return cls(i, symbol.isupper())
        raise ValueError(f"Invalid piece symbol: {symbol!r}")


class Move:
    """Represents a move from a square to a square."""

    def __init__(self, from_square: Square, to_square: Square) -> None:
        self.from_square = from_square
        self.to_square = to_square

    def uci(self) -> str:
        """Gets a UCI-like string for the move."""
        if self:
            return SQUARE_NAMES[self.from_square] + SQUARE_NAMES[self.to_square]
        return "0000"

    def __bool__(self) -> bool:
        return bool(self.from_square or self.to_square)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Move):
            return (self.from_square == other.from_square and 
                    self.to_square == other.to_square)
        return NotImplemented

    def __repr__(self) -> str:
        return f"Move.from_uci({self.uci()!r})"

    def __str__(self) -> str:
        return self.uci()

    def __hash__(self) -> int:
        return hash((self.from_square, self.to_square))

    @classmethod
    def from_uci(cls, uci: str) -> "Move":
        """Parses a UCI string."""
        if uci == "0000":
            return cls.null()
        if len(uci) == 4:
            from_square = SQUARE_NAMES.index(uci[0:2])
            to_square = SQUARE_NAMES.index(uci[2:4])
            return cls(from_square, to_square)
        raise ValueError(f"Invalid UCI: {uci!r}")

    @classmethod
    def null(cls) -> "Move":
        """Gets a null move."""
        return cls(0, 0)


BoardT = TypeVar("BoardT", bound="Board")


class Board:
    """
    Animal Chess board with move generation and validation.
    """

    starting_fen = STARTING_FEN

    def __init__(self, fen: Optional[str] = STARTING_FEN) -> None:
        self.pieces_dict: Dict[Square, Piece] = {}
        self.turn = WHITE
        self.move_stack: List[Move] = []
        self.halfmove_clock = 0
        self.fullmove_number = 1

        if fen is None:
            self.clear()
        elif fen == self.starting_fen:
            self.reset()
        else:
            self.set_fen(fen)

    def reset(self) -> None:
        """Restores the starting position."""
        self.clear()
        self.turn = WHITE
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        # 设置白方棋子 (底部，rank 0-2)
        self.set_piece_at(A1, Piece(LION, WHITE))
        self.set_piece_at(G1, Piece(TIGER, WHITE))
        self.set_piece_at(B2, Piece(DOG, WHITE))
        self.set_piece_at(F2, Piece(CAT, WHITE))
        self.set_piece_at(A3, Piece(RAT, WHITE))
        self.set_piece_at(C3, Piece(LEOPARD, WHITE))
        self.set_piece_at(E3, Piece(WOLF, WHITE))
        self.set_piece_at(G3, Piece(ELEPHANT, WHITE))
        
        # 设置黑方棋子 (顶部，rank 6-8)
        self.set_piece_at(A7, Piece(ELEPHANT, BLACK))
        self.set_piece_at(C7, Piece(WOLF, BLACK))
        self.set_piece_at(E7, Piece(LEOPARD, BLACK))
        self.set_piece_at(G7, Piece(RAT, BLACK))
        self.set_piece_at(B8, Piece(CAT, BLACK))
        self.set_piece_at(F8, Piece(DOG, BLACK))
        self.set_piece_at(A9, Piece(TIGER, BLACK))
        self.set_piece_at(G9, Piece(LION, BLACK))

    def clear(self) -> None:
        """Clears the board."""
        self.pieces_dict.clear()
        self.turn = WHITE
        self.move_stack.clear()

    def piece_at(self, square: Square) -> Optional[Piece]:
        """Gets the piece at the given square."""
        return self.pieces_dict.get(square)

    def set_piece_at(self, square: Square, piece: Optional[Piece]) -> None:
        """Sets a piece at the given square."""
        if piece is None:
            self.pieces_dict.pop(square, None)
        else:
            self.pieces_dict[square] = piece

    def remove_piece_at(self, square: Square) -> Optional[Piece]:
        """Removes and returns the piece at the given square."""
        return self.pieces_dict.pop(square, None)

    def is_river(self, square: Square) -> bool:
        """Checks if square is in river."""
        return square in RIVER_SQUARES

    def is_trap(self, square: Square, for_color: Color) -> bool:
        """Checks if square is a trap for the given color."""
        if for_color == WHITE:
            return square in BLACK_TRAPS  # 白方进入黑方陷阱
        else:
            return square in WHITE_TRAPS  # 黑方进入白方陷阱

    def is_den(self, square: Square) -> bool:
        """Checks if square is a den."""
        return square in [WHITE_DEN, BLACK_DEN]

    def get_piece_power(self, square: Square) -> int:
        """
        Gets the effective power of a piece at a square.
        Returns 0 if in enemy trap, otherwise returns piece type.
        """
        piece = self.piece_at(square)
        if piece is None:
            return 0
        
        # 在敌方陷阱中，等级降为0
        if self.is_trap(square, not piece.color):
            return 0
        
        return piece.piece_type

    def can_capture(self, from_square: Square, to_square: Square) -> bool:
        """
        Checks if piece at from_square can capture piece at to_square.
        """
        attacker = self.piece_at(from_square)
        defender = self.piece_at(to_square)
        
        if attacker is None or defender is None:
            return False
        
        if attacker.color == defender.color:
            return False
        
        attacker_power = self.get_piece_power(from_square)
        defender_power = self.get_piece_power(to_square)
        
        # 特殊规则：鼠吃象
        if attacker.piece_type == RAT and defender.piece_type == ELEPHANT:
            return True
        
        # 特殊规则：象不能吃鼠
        if attacker.piece_type == ELEPHANT and defender.piece_type == RAT:
            return False
        
        # 普通规则：大吃小或同级
        return attacker_power >= defender_power

    def can_jump_river(self, from_square: Square, to_square: Square) -> bool:
        """
        Checks if a lion or tiger can jump over river.
        """
        piece = self.piece_at(from_square)
        if piece is None or piece.piece_type not in [LION, TIGER]:
            return False
        
        from_file, from_rank = square_file(from_square), square_rank(from_square)
        to_file, to_rank = square_file(to_square), square_rank(to_square)
        
        # 必须是直线跳跃
        if from_file != to_file and from_rank != to_rank:
            return False
        
        # 检查是否跳过河流
        if from_file == to_file:  # 纵向跳跃
            if from_rank < 3 and to_rank > 5:  # 向上跳
                # 检查河中是否有老鼠
                for r in [3, 4, 5]:
                    check_square = square(from_file, r)
                    if check_square in RIVER_SQUARES:
                        rat_piece = self.piece_at(check_square)
                        if rat_piece and rat_piece.piece_type == RAT:
                            return False
                return True
            elif from_rank > 5 and to_rank < 3:  # 向下跳
                for r in [3, 4, 5]:
                    check_square = square(from_file, r)
                    if check_square in RIVER_SQUARES:
                        rat_piece = self.piece_at(check_square)
                        if rat_piece and rat_piece.piece_type == RAT:
                            return False
                return True
        
        if from_rank == to_rank and from_rank in [3, 4, 5]:  # 横向跳跃
            min_file, max_file = min(from_file, to_file), max(from_file, to_file)
            for f in range(min_file + 1, max_file):
                check_square = square(f, from_rank)
                if check_square in RIVER_SQUARES:
                    rat_piece = self.piece_at(check_square)
                    if rat_piece and rat_piece.piece_type == RAT:
                        return False
            return True
        
        return False

    def generate_pseudo_legal_moves(self) -> Iterator[Move]:
        """Generates all pseudo-legal moves."""
        for from_square in range(63):
            piece = self.piece_at(from_square)
            if piece is None or piece.color != self.turn:
                continue
            
            # 不能进入己方兽穴
            own_den = WHITE_DEN if self.turn == WHITE else BLACK_DEN
            
            from_file, from_rank = square_file(from_square), square_rank(from_square)
            
            # 普通移动：上下左右一格
            for df, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                to_file, to_rank = from_file + df, from_rank + dr
                
                if 0 <= to_file < 7 and 0 <= to_rank < 9:
                    to_square = square(to_file, to_rank)
                    
                    # 不能进己方兽穴
                    if to_square == own_den:
                        continue
                    
                    # 检查河流规则
                    if self.is_river(to_square):
                        # 只有老鼠能进河
                        if piece.piece_type != RAT:
                            continue
                    
                    # 检查是否有己方棋子
                    target_piece = self.piece_at(to_square)
                    if target_piece and target_piece.color == self.turn:
                        continue
                    
                    # 如果有敌方棋子，检查能否吃
                    if target_piece:
                        if self.can_capture(from_square, to_square):
                            yield Move(from_square, to_square)
                    else:
                        yield Move(from_square, to_square)
            
            # 狮虎跳河
            if piece.piece_type in [LION, TIGER]:
                # 尝试跳跃
                for df, dr in [(0, 3), (0, -3), (3, 0), (-3, 0), (4, 0), (-4, 0)]:
                    to_file, to_rank = from_file + df, from_rank + dr
                    
                    if 0 <= to_file < 7 and 0 <= to_rank < 9:
                        to_square = square(to_file, to_rank)
                        
                        if to_square == own_den:
                            continue
                        
                        if self.can_jump_river(from_square, to_square):
                            target_piece = self.piece_at(to_square)
                            if target_piece:
                                if target_piece.color != self.turn and self.can_capture(from_square, to_square):
                                    yield Move(from_square, to_square)
                            else:
                                yield Move(from_square, to_square)

    def is_legal(self, move: Move) -> bool:
        """Checks if a move is legal."""
        # 简化版：所有伪合法移动都是合法的
        return move in list(self.generate_pseudo_legal_moves())

    def push(self, move: Move) -> None:
        """Makes a move."""
        piece = self.piece_at(move.from_square)
        captured = self.piece_at(move.to_square)
        
        self.set_piece_at(move.to_square, piece)
        self.set_piece_at(move.from_square, None)
        
        self.move_stack.append(move)
        self.turn = not self.turn
        
        if captured:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        
        if self.turn == WHITE:
            self.fullmove_number += 1

    def pop(self) -> Move:
        """Unmakes the last move."""
        if not self.move_stack:
            raise IndexError("Move stack is empty")
        
        move = self.move_stack.pop()
        # 简化版：不恢复被吃的棋子
        piece = self.piece_at(move.to_square)
        self.set_piece_at(move.from_square, piece)
        self.set_piece_at(move.to_square, None)
        
        self.turn = not self.turn
        return move

    def is_game_over(self) -> bool:
        """Checks if the game is over."""
        # 检查是否有棋子进入对方兽穴
        enemy_den = BLACK_DEN if self.turn == WHITE else WHITE_DEN
        piece_in_den = self.piece_at(enemy_den)
        if piece_in_den and piece_in_den.color == self.turn:
            return True
        
        # 检查对方是否还有棋子
        enemy_color = not self.turn
        has_enemy_pieces = any(p.color == enemy_color for p in self.pieces_dict.values())
        if not has_enemy_pieces:
            return True
        
        return False

    def result(self) -> str:
        """Returns game result."""
        if not self.is_game_over():
            return "*"
        
        enemy_den = BLACK_DEN if self.turn == WHITE else WHITE_DEN
        piece_in_den = self.piece_at(enemy_den)
        if piece_in_den and piece_in_den.color == self.turn:
            return "1-0" if self.turn == WHITE else "0-1"
        
        enemy_color = not self.turn
        has_enemy_pieces = any(p.color == enemy_color for p in self.pieces_dict.values())
        if not has_enemy_pieces:
            return "1-0" if self.turn == WHITE else "0-1"
        
        return "*"

    def fen(self) -> str:
        """Gets FEN representation."""
        # 简化的FEN格式
        board_part = []
        for rank in range(8, -1, -1):
            empty = 0
            rank_str = []
            for file in range(7):
                sq = square(file, rank)
                piece = self.piece_at(sq)
                if piece:
                    if empty:
                        rank_str.append(str(empty))
                        empty = 0
                    rank_str.append(piece.symbol())
                else:
                    empty += 1
            if empty:
                rank_str.append(str(empty))
            board_part.append("".join(rank_str))
        
        board_fen = "/".join(board_part)
        turn_part = "r" if self.turn == WHITE else "b"
        
        return f"{board_fen} {turn_part} {self.halfmove_clock} {self.fullmove_number}"

    def set_fen(self, fen: str) -> None:
        """Sets position from FEN."""
        parts = fen.split()
        if len(parts) < 2:
            raise ValueError("Invalid FEN")
        
        self.clear()
        
        # Parse board
        ranks = parts[0].split("/")
        for rank_idx, rank_str in enumerate(ranks):
            file_idx = 0
            for char in rank_str:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    sq = square(file_idx, 8 - rank_idx)
                    self.set_piece_at(sq, Piece.from_symbol(char))
                    file_idx += 1
        
        # Parse turn
        self.turn = WHITE if parts[1] == "r" else BLACK
        
        # Parse counters
        if len(parts) > 2:
            self.halfmove_clock = int(parts[2])
        if len(parts) > 3:
            self.fullmove_number = int(parts[3])

    def __str__(self) -> str:
        """String representation of the board with SGF-style coordinates."""
        lines = []
        
        # 顶部列标签
        lines.append("  " + " ".join([chr(ord('a') + i) for i in range(7)]))
        
        # 从第9行到第1行（从a到i）
        for rank in range(8, -1, -1):
            row_label = chr(ord('a') + rank)
            line = [row_label]
            for file in range(7):  # 从左到右：0->6
                sq = square(file, rank)
                piece = self.piece_at(sq)
                if piece:
                    line.append(piece.symbol())
                else:
                    # 显示特殊地形
                    if sq in RIVER_SQUARES:
                        line.append("~")
                    elif sq == WHITE_DEN:
                        line.append("穴")
                    elif sq == BLACK_DEN:
                        line.append("穴")
                    elif sq in WHITE_TRAPS or sq in BLACK_TRAPS:
                        line.append("阱")
                    else:
                        line.append("·")
            lines.append(" ".join(line))
        return "\n".join(lines)

    def unicode(self) -> str:
        """Unicode representation with emojis and SGF-style coordinates."""
        lines = []
        
        # 顶部列标签
        lines.append("  " + " ".join([chr(ord('a') + i) for i in range(7)]))
        
        # 从第9行到第1行（从a到i）
        for rank in range(8, -1, -1):
            row_label = chr(ord('a') + rank)
            line = [row_label]
            for file in range(7):  # 从左到右：0->6
                sq = square(file, rank)
                piece = self.piece_at(sq)
                if piece:
                    line.append(piece.unicode_symbol())
                else:
                    if sq in RIVER_SQUARES:
                        line.append("🌊")
                    elif sq == WHITE_DEN or sq == BLACK_DEN:
                        line.append("🏠")
                    elif sq in WHITE_TRAPS or sq in BLACK_TRAPS:
                        line.append("⚠️")
                    else:
                        line.append("·")
            lines.append(" ".join(line))
        return "\n".join(lines)

    def copy(self) -> "Board":
        """Creates a copy of the board."""
        board = type(self)(None)
        board.pieces_dict = self.pieces_dict.copy()
        board.turn = self.turn
        board.move_stack = self.move_stack.copy()
        board.halfmove_clock = self.halfmove_clock
        board.fullmove_number = self.fullmove_number
        return board

    def __repr__(self) -> str:
        return f"Board({self.fen()!r})"

