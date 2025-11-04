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


# === 7x9 Bitboard 定义与掩码 ===
Bitboard = int
BB_EMPTY: Bitboard = 0
BB_ALL: Bitboard = (1 << 63) - 1  # 7*9 = 63 格

# 每格单比特掩码
BB_SQUARES: List[Bitboard] = [1 << sq for sq in SQUARES]

# 按列/按行掩码
BB_FILES: List[Bitboard] = [
    sum(BB_SQUARES[sq] for sq in SQUARES if square_file(sq) == f)
    for f in range(7)
]

BB_RANKS: List[Bitboard] = [
    sum(BB_SQUARES[sq] for sq in SQUARES if square_rank(sq) == r)
    for r in range(9)
]

# 特殊地形掩码
def bb_from_squares(squares: Iterable[Square]) -> Bitboard:
    mask: Bitboard = BB_EMPTY
    for sq in squares:
        mask |= BB_SQUARES[sq]
    return mask

BB_RIVER: Bitboard = bb_from_squares(RIVER_SQUARES)
BB_WHITE_TRAPS: Bitboard = bb_from_squares(WHITE_TRAPS)
BB_BLACK_TRAPS: Bitboard = bb_from_squares(BLACK_TRAPS)
BB_WHITE_DEN: Bitboard = BB_SQUARES[WHITE_DEN]
BB_BLACK_DEN: Bitboard = BB_SQUARES[BLACK_DEN]

# 基础位操作与遍历
def popcount(bb: Bitboard) -> int:
    return int(bb.bit_count())

def lsb(bb: Bitboard) -> int:
    return (bb & -bb).bit_length() - 1

def msb(bb: Bitboard) -> int:
    return bb.bit_length() - 1

def scan_forward(bb: Bitboard) -> Iterator[Square]:
    while bb:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r

def scan_reversed(bb: Bitboard) -> Iterator[Square]:
    while bb:
        r = bb.bit_length() - 1
        yield r
        bb ^= BB_SQUARES[r]


class SquareSet:
    """7x9 位板集合的轻量封装。"""

    def __init__(self, squares: Union[int, Iterable[Square]] = BB_EMPTY) -> None:
        try:
            self.mask: Bitboard = int(squares) & BB_ALL  # type: ignore[arg-type]
            return
        except Exception:
            self.mask = BB_EMPTY
        for sq in typing.cast(Iterable[Square], squares):  # type: ignore[arg-type]
            self.add(sq)

    def __contains__(self, square: Square) -> bool:
        return bool(BB_SQUARES[square] & self.mask)

    def __iter__(self) -> Iterator[Square]:
        return scan_forward(self.mask)

    def __len__(self) -> int:
        return popcount(self.mask)

    def add(self, square: Square) -> None:
        self.mask |= BB_SQUARES[square]

    def discard(self, square: Square) -> None:
        self.mask &= ~BB_SQUARES[square]

    def tolist(self) -> List[bool]:
        result = [False] * 63
        for sq in self:
            result[sq] = True
        return result

    def __int__(self) -> int:
        return self.mask

    def __repr__(self) -> str:
        return f"SquareSet({self.mask:#018x})"

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

    def _repr_svg_(self) -> str:
        import jungle_chess.svg
        return jungle_chess.svg.piece(self, size=45)

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


BaseBoardT = TypeVar("BaseBoardT", bound="BaseBoard")


class BaseBoard:
    """
    基础棋盘类，使用位板表示棋子位置。
    不包含移动生成等高级功能，仅提供棋盘状态的基本操作。
    """

    def __init__(self, fen: Optional[str] = STARTING_FEN) -> None:
        # 各类棋子的位板
        self.rats: Bitboard = BB_EMPTY
        self.cats: Bitboard = BB_EMPTY
        self.dogs: Bitboard = BB_EMPTY
        self.wolves: Bitboard = BB_EMPTY
        self.leopards: Bitboard = BB_EMPTY
        self.tigers: Bitboard = BB_EMPTY
        self.lions: Bitboard = BB_EMPTY
        self.elephants: Bitboard = BB_EMPTY

        # 颜色占用位板
        self.occupied_co: List[Bitboard] = [BB_EMPTY, BB_EMPTY]
        self.occupied: Bitboard = BB_EMPTY

        if fen is None:
            self._clear_board()
        elif fen == STARTING_FEN:
            self._reset_board()
        else:
            self._set_board_fen(fen)

    def _reset_board(self) -> None:
        """重置到初始局面。"""
        self._clear_board()
        
        # 白方棋子 (底部)
        self._set_piece_at(A1, LION, WHITE)
        self._set_piece_at(G1, TIGER, WHITE)
        self._set_piece_at(B2, DOG, WHITE)
        self._set_piece_at(F2, CAT, WHITE)
        self._set_piece_at(A3, RAT, WHITE)
        self._set_piece_at(C3, LEOPARD, WHITE)
        self._set_piece_at(E3, WOLF, WHITE)
        self._set_piece_at(G3, ELEPHANT, WHITE)
        
        # 黑方棋子 (顶部)
        self._set_piece_at(A7, ELEPHANT, BLACK)
        self._set_piece_at(C7, WOLF, BLACK)
        self._set_piece_at(E7, LEOPARD, BLACK)
        self._set_piece_at(G7, RAT, BLACK)
        self._set_piece_at(B8, CAT, BLACK)
        self._set_piece_at(F8, DOG, BLACK)
        self._set_piece_at(A9, TIGER, BLACK)
        self._set_piece_at(G9, LION, BLACK)

    def reset_board(self) -> None:
        """重置棋盘到初始位置。"""
        self._reset_board()

    def _clear_board(self) -> None:
        """清空棋盘。"""
        self.rats = BB_EMPTY
        self.cats = BB_EMPTY
        self.dogs = BB_EMPTY
        self.wolves = BB_EMPTY
        self.leopards = BB_EMPTY
        self.tigers = BB_EMPTY
        self.lions = BB_EMPTY
        self.elephants = BB_EMPTY

        self.occupied_co[WHITE] = BB_EMPTY
        self.occupied_co[BLACK] = BB_EMPTY
        self.occupied = BB_EMPTY

    def clear_board(self) -> None:
        """清空棋盘。"""
        self._clear_board()

    def pieces_mask(self, piece_type: PieceType, color: Color) -> Bitboard:
        """获取指定类型和颜色的棋子位板。"""
        if piece_type == RAT:
            bb = self.rats
        elif piece_type == CAT:
            bb = self.cats
        elif piece_type == DOG:
            bb = self.dogs
        elif piece_type == WOLF:
            bb = self.wolves
        elif piece_type == LEOPARD:
            bb = self.leopards
        elif piece_type == TIGER:
            bb = self.tigers
        elif piece_type == LION:
            bb = self.lions
        elif piece_type == ELEPHANT:
            bb = self.elephants
        else:
            bb = BB_EMPTY

        return bb & self.occupied_co[color]

    def pieces(self, piece_type: PieceType, color: Color) -> SquareSet:
        """获取指定类型和颜色的棋子集合。"""
        return SquareSet(self.pieces_mask(piece_type, color))

    def piece_at(self, square: Square) -> Optional[Piece]:
        """获取指定位置的棋子。"""
        piece_type = self.piece_type_at(square)
        if piece_type:
            mask = BB_SQUARES[square]
            color = bool(self.occupied_co[WHITE] & mask)
            return Piece(piece_type, color)
        return None

    def piece_type_at(self, square: Square) -> Optional[PieceType]:
        """获取指定位置的棋子类型。"""
        mask = BB_SQUARES[square]

        if not self.occupied & mask:
            return None
        elif self.rats & mask:
            return RAT
        elif self.cats & mask:
            return CAT
        elif self.dogs & mask:
            return DOG
        elif self.wolves & mask:
            return WOLF
        elif self.leopards & mask:
            return LEOPARD
        elif self.tigers & mask:
            return TIGER
        elif self.lions & mask:
            return LION
        elif self.elephants & mask:
            return ELEPHANT
        else:
            return None

    def color_at(self, square: Square) -> Optional[Color]:
        """获取指定位置棋子的颜色。"""
        mask = BB_SQUARES[square]
        if self.occupied_co[WHITE] & mask:
            return WHITE
        elif self.occupied_co[BLACK] & mask:
            return BLACK
        return None

    def _remove_piece_at(self, square: Square) -> Optional[PieceType]:
        """移除指定位置的棋子（内部方法）。"""
        piece_type = self.piece_type_at(square)
        mask = BB_SQUARES[square]

        if piece_type == RAT:
            self.rats ^= mask
        elif piece_type == CAT:
            self.cats ^= mask
        elif piece_type == DOG:
            self.dogs ^= mask
        elif piece_type == WOLF:
            self.wolves ^= mask
        elif piece_type == LEOPARD:
            self.leopards ^= mask
        elif piece_type == TIGER:
            self.tigers ^= mask
        elif piece_type == LION:
            self.lions ^= mask
        elif piece_type == ELEPHANT:
            self.elephants ^= mask
        else:
            return None

        self.occupied ^= mask
        self.occupied_co[WHITE] &= ~mask
        self.occupied_co[BLACK] &= ~mask

        return piece_type

    def remove_piece_at(self, square: Square) -> Optional[Piece]:
        """移除并返回指定位置的棋子。"""
        color = bool(self.occupied_co[WHITE] & BB_SQUARES[square])
        piece_type = self._remove_piece_at(square)
        return Piece(piece_type, color) if piece_type else None

    def _set_piece_at(self, square: Square, piece_type: PieceType, color: Color) -> None:
        """在指定位置放置棋子（内部方法）。"""
        self._remove_piece_at(square)

        mask = BB_SQUARES[square]

        if piece_type == RAT:
            self.rats |= mask
        elif piece_type == CAT:
            self.cats |= mask
        elif piece_type == DOG:
            self.dogs |= mask
        elif piece_type == WOLF:
            self.wolves |= mask
        elif piece_type == LEOPARD:
            self.leopards |= mask
        elif piece_type == TIGER:
            self.tigers |= mask
        elif piece_type == LION:
            self.lions |= mask
        elif piece_type == ELEPHANT:
            self.elephants |= mask
        else:
            return

        self.occupied ^= mask
        self.occupied_co[color] ^= mask

    def set_piece_at(self, square: Square, piece: Optional[Piece]) -> None:
        """在指定位置设置棋子。"""
        if piece is None:
            self._remove_piece_at(square)
        else:
            self._set_piece_at(square, piece.piece_type, piece.color)

    def _set_board_fen(self, fen: str) -> None:
        """从FEN字符串设置棋盘（内部方法）。"""
        parts = fen.split()
        board_part = parts[0] if parts else fen
        
        self._clear_board()
        
        ranks = board_part.split("/")
        for rank_idx, rank_str in enumerate(ranks):
            file_idx = 0
            for char in rank_str:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    sq = square(file_idx, 8 - rank_idx)
                    piece = Piece.from_symbol(char)
                    self._set_piece_at(sq, piece.piece_type, piece.color)
                    file_idx += 1

    def copy(self: BaseBoardT) -> BaseBoardT:
        """创建棋盘的副本。"""
        board = type(self)(None)

        board.rats = self.rats
        board.cats = self.cats
        board.dogs = self.dogs
        board.wolves = self.wolves
        board.leopards = self.leopards
        board.tigers = self.tigers
        board.lions = self.lions
        board.elephants = self.elephants

        board.occupied_co[WHITE] = self.occupied_co[WHITE]
        board.occupied_co[BLACK] = self.occupied_co[BLACK]
        board.occupied = self.occupied

        return board

    def __copy__(self: BaseBoardT) -> BaseBoardT:
        return self.copy()

    @classmethod
    def empty(cls: Type[BaseBoardT]) -> BaseBoardT:
        """创建空棋盘。"""
        return cls(None)


BoardT = TypeVar("BoardT", bound="Board")


class _BoardState:
    """保存棋盘状态的快照，用于push/pop操作。"""

    def __init__(self, board: "Board") -> None:
        # 保存所有棋子位板
        self.rats = board.rats
        self.cats = board.cats
        self.dogs = board.dogs
        self.wolves = board.wolves
        self.leopards = board.leopards
        self.tigers = board.tigers
        self.lions = board.lions
        self.elephants = board.elephants

        # 保存占用位板
        self.occupied_w = board.occupied_co[WHITE]
        self.occupied_b = board.occupied_co[BLACK]
        self.occupied = board.occupied

        # 保存游戏状态
        self.turn = board.turn
        self.halfmove_clock = board.halfmove_clock
        self.fullmove_number = board.fullmove_number

    def restore(self, board: "Board") -> None:
        """恢复棋盘状态。"""
        board.rats = self.rats
        board.cats = self.cats
        board.dogs = self.dogs
        board.wolves = self.wolves
        board.leopards = self.leopards
        board.tigers = self.tigers
        board.lions = self.lions
        board.elephants = self.elephants

        board.occupied_co[WHITE] = self.occupied_w
        board.occupied_co[BLACK] = self.occupied_b
        board.occupied = self.occupied

        board.turn = self.turn
        board.halfmove_clock = self.halfmove_clock
        board.fullmove_number = self.fullmove_number


class Board(BaseBoard):
    """
    Animal Chess board with move generation and validation.
    继承自 BaseBoard，增加移动生成、游戏规则等高级功能。
    """

    starting_fen = STARTING_FEN

    def __init__(self, fen: Optional[str] = STARTING_FEN) -> None:
        BaseBoard.__init__(self, None)
        
        self.turn = WHITE
        self.move_stack: List[Move] = []
        self._stack: List[_BoardState] = []
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
        self.turn = WHITE
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.reset_board()
        self.clear_stack()

    def clear(self) -> None:
        """Clears the board."""
        self.turn = WHITE
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.clear_board()
        self.clear_stack()

    def clear_stack(self) -> None:
        """清空移动栈。"""
        self.move_stack.clear()
        self._stack.clear()

    def root(self) -> "Board":
        """返回根局面的副本。"""
        if self._stack:
            board = type(self)(None)
            self._stack[0].restore(board)
            return board
        else:
            return self.copy(stack=False)

    def remove_piece_at(self, square: Square) -> Optional[Piece]:
        """移除棋子并清空移动栈。"""
        piece = super().remove_piece_at(square)
        self.clear_stack()
        return piece

    def set_piece_at(self, square: Square, piece: Optional[Piece]) -> None:
        """设置棋子并清空移动栈。"""
        super().set_piece_at(square, piece)
        self.clear_stack()

    def _board_state(self) -> _BoardState:
        """创建当前棋盘状态的快照。"""
        return _BoardState(self)

    def peek(self) -> Move:
        """
        获取最后一步移动（不弹出）。
        
        :raises: IndexError 如果移动栈为空。
        """
        return self.move_stack[-1]

    @property
    def pseudo_legal_moves(self) -> "PseudoLegalMoveGenerator":
        """返回伪合法移动生成器。"""
        return PseudoLegalMoveGenerator(self)

    @property
    def legal_moves(self) -> "LegalMoveGenerator":
        """返回合法移动生成器（斗兽棋中与伪合法相同）。"""
        return LegalMoveGenerator(self)

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
        """Makes a move and saves board state."""
        # 保存当前状态到栈
        board_state = self._board_state()
        self._stack.append(board_state)
        self.move_stack.append(move)
        
        # 执行移动（使用父类方法避免清空栈）
        piece = self.piece_at(move.from_square)
        captured = self.piece_at(move.to_square)
        
        # 直接调用 BaseBoard 的方法，不触发 clear_stack
        BaseBoard.set_piece_at(self, move.to_square, piece)
        BaseBoard.set_piece_at(self, move.from_square, None)
        
        # 更新游戏状态
        self.turn = not self.turn
        
        if captured:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        
        if self.turn == WHITE:
            self.fullmove_number += 1

    def pop(self) -> Move:
        """Unmakes the last move by restoring previous state."""
        if not self.move_stack:
            raise IndexError("Move stack is empty")
        
        move = self.move_stack.pop()
        self._stack.pop().restore(self)
        return move

    def is_game_over(self) -> bool:
        """Checks if the game is over."""
        # 检查是否有棋子进入己方兽穴（对方获胜）
        # 如果当前是白方走棋，检查白方兽穴是否有黑方棋子
        # 如果当前是黑方走棋，检查黑方兽穴是否有白方棋子
        own_den = WHITE_DEN if self.turn == WHITE else BLACK_DEN
        piece_in_den = self.piece_at(own_den)
        if piece_in_den and piece_in_den.color != self.turn:
            return True
        
        # 检查对方是否还有棋子（使用位板）
        # 如果当前是白方走棋，检查黑方是否还有棋子
        # 如果当前是黑方走棋，检查白方是否还有棋子
        enemy_color = not self.turn
        has_enemy_pieces = bool(self.occupied_co[enemy_color])
        if not has_enemy_pieces:
            return True
        
        return False

    def result(self) -> str:
        """Returns game result."""
        if not self.is_game_over():
            return "*"
        
        # 检查是否有棋子进入己方兽穴（对方获胜）
        own_den = WHITE_DEN if self.turn == WHITE else BLACK_DEN
        piece_in_den = self.piece_at(own_den)
        if piece_in_den and piece_in_den.color != self.turn:
            # 对方棋子进入己方兽穴，对方获胜
            return "1-0" if piece_in_den.color == WHITE else "0-1"
        
        # 检查对方是否还有棋子（使用位板）
        enemy_color = not self.turn
        has_enemy_pieces = bool(self.occupied_co[enemy_color])
        if not has_enemy_pieces:
            # 对方无子，当前走棋方获胜
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

    def copy(self, *, stack: bool = True) -> "Board":
        """Creates a copy of the board."""
        board = super().copy()
        
        board.turn = self.turn
        board.halfmove_clock = self.halfmove_clock
        board.fullmove_number = self.fullmove_number
        
        if stack:
            board.move_stack = self.move_stack.copy()
            board._stack = self._stack.copy()
        else:
            board.move_stack = []
            board._stack = []
        
        return board

    def __repr__(self) -> str:
        return f"Board({self.fen()!r})"


class PseudoLegalMoveGenerator:
    """伪合法移动生成器。"""

    def __init__(self, board: Board) -> None:
        self.board = board

    def __bool__(self) -> bool:
        """检查是否有任何伪合法移动。"""
        return any(self.board.generate_pseudo_legal_moves())

    def count(self) -> int:
        """统计伪合法移动数量。"""
        return len(list(self))

    def __iter__(self) -> Iterator[Move]:
        """迭代所有伪合法移动。"""
        return self.board.generate_pseudo_legal_moves()

    def __contains__(self, move: Move) -> bool:
        """检查移动是否伪合法。"""
        return self.board.is_legal(move)

    def __repr__(self) -> str:
        moves = ", ".join(move.uci() for move in self)
        return f"<PseudoLegalMoveGenerator at {id(self):#x} ({moves})>"


class LegalMoveGenerator:
    """合法移动生成器（斗兽棋中与伪合法相同）。"""

    def __init__(self, board: Board) -> None:
        self.board = board

    def __bool__(self) -> bool:
        """检查是否有任何合法移动。"""
        return any(self.board.generate_pseudo_legal_moves())

    def count(self) -> int:
        """统计合法移动数量。"""
        return len(list(self))

    def __iter__(self) -> Iterator[Move]:
        """迭代所有合法移动。"""
        return self.board.generate_pseudo_legal_moves()

    def __contains__(self, move: Move) -> bool:
        """检查移动是否合法。"""
        return self.board.is_legal(move)

    def __repr__(self) -> str:
        moves = ", ".join(move.uci() for move in self)
        return f"<LegalMoveGenerator at {id(self):#x} ({moves})>"
