"""
Animal Chess SFG (SGF-like) parsing utilities.
- Parses headers like: (;FF[4]GM[1]SZ[7:9]PB[...]PW[...]RU[...]RE[...]OP[...]C[...];B[cc]C[...];W[cd]...)
- Pairs two consecutive same-color coords into a complete move: from -> to
- Converts SGF coords (file a-g, rank a-i) to internal squares and to UCI
- Can build a Board from OP[...] initial position
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable
import re

import animalchess as ac

# --- Coordinate helpers -----------------------------------------------------

def sgf_coord_to_square(coord: str) -> ac.Square:
    if len(coord) != 2:
        raise ValueError(f"Invalid SGF coord: {coord!r}")
    file_idx = ord(coord[0]) - ord("a")  # a..g -> 0..6
    rank_idx = ord(coord[1]) - ord("a")  # a..i -> 0..8
    if not (0 <= file_idx < 7 and 0 <= rank_idx < 9):
        raise ValueError(f"SGF coord out of bounds: {coord!r}")
    return ac.square(file_idx, rank_idx)


def square_to_sgf(square: ac.Square) -> str:
    f = ac.square_file(square)
    r = ac.square_rank(square)
    return chr(ord("a") + f) + chr(ord("a") + r)


def uci_from_sgf_move(from_sgf: str, to_sgf: str) -> str:
    fsq = sgf_coord_to_square(from_sgf)
    tsq = sgf_coord_to_square(to_sgf)
    return ac.square_name(fsq) + ac.square_name(tsq)

# --- Position helpers -------------------------------------------------------

# Map SGF position piece letters to our FEN letters (keep as-is).
# SGF样例行使用: l5t, 1d3c1 等，直接进入我们的简化FEN即可。

def fen_from_sgf_position(sgf_pos: str, turn_token: str = "w") -> str:
    """
    Convert SGF position (rows delimited by '/') to our FEN-style string.
    turn_token: 'w' or 'b' in SGF; our FEN uses 'r' for WHITE (bottom) and 'b' for BLACK.
    """
    # Translate SGF piece letters to our symbols (leopard j/J -> p/P)
    trans = str.maketrans({"j": "p", "J": "P"})
    pos = sgf_pos.translate(trans)
    side = "r" if turn_token.lower() == "w" else "b"
    return f"{pos} {side} 0 1"

# --- SFG object model -------------------------------------------------------

@dataclass
class Headers:
    data: Dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> str:
        return self.data[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.data[key] = value

    def items(self):
        return self.data.items()


@dataclass
class MoveNode:
    color: str  # 'B' or 'W'
    coord: str  # SGF coord like 'cg'
    comment: str = ""


@dataclass
class PairedMove:
    color: str  # 'B' or 'W'
    from_sgf: str
    to_sgf: str
    comment: str = ""

    def uci(self) -> str:
        return uci_from_sgf_move(self.from_sgf, self.to_sgf)


@dataclass
class Game:
    headers: Headers
    nodes: List[MoveNode]
    moves: List[PairedMove]

    def to_board(self) -> ac.Board:
        """Build a Board from headers and play all paired moves."""
        # Initial position from OP[...] header, format like: l5t/... w
        op = self.headers.get("OP", "")
        fen: Optional[str] = None
        if op:
            # Expect something like: "l5t/... w" (position + space + turn)
            parts = op.split()
            pos = parts[0]
            side = parts[1] if len(parts) > 1 else "w"
            fen = fen_from_sgf_position(pos, side)
        board = ac.Board(fen if fen else ac.STARTING_FEN)

        # Play moves
        for pm in self.moves:
            uci = pm.uci()
            mv = ac.Move.from_uci(uci)
            if board.is_legal(mv):
                board.push(mv)
            else:
                # 仍然尝试执行（有些SGF来自外系统，严格合法性可能不同）
                board.push(mv)
        return board

# --- Parser -----------------------------------------------------------------

_HEADER_KV_RE = re.compile(r"([A-Z]{1,3})\[([^\]]*)\]")
_NODE_RE = re.compile(r";([BW])\[([^\]]+)\](?:C\[([^\]]*)\])?")


def parse_headers(sfg_text: str) -> Headers:
    # 拿到根节点括号内第一部分的 headers
    # 形式: (;FF[4]GM[1]...OP[...]C[...] ;B[...] ...)
    start = sfg_text.find("(")
    if start == -1:
        return Headers({})
    # 到第一个 ;B 或 ;W 之前
    m = re.search(r";[BW]\[", sfg_text)
    end = m.start() if m else sfg_text.find(")", start + 1)
    chunk = sfg_text[start:end if end != -1 else None]

    hdrs: Dict[str, str] = {}
    for k, v in _HEADER_KV_RE.findall(chunk):
        hdrs[k] = v.strip()
    return Headers(hdrs)


def parse_nodes(sfg_text: str) -> List[MoveNode]:
    nodes: List[MoveNode] = []
    for color, coord, comment in _NODE_RE.findall(sfg_text):
        nodes.append(MoveNode(color=color, coord=coord.strip(), comment=(comment or "").strip()))
    return nodes


def pair_moves(nodes: List[MoveNode]) -> List[PairedMove]:
    """
    将同一方连续两次坐标配对为一次完整走子:
    例如: B[cg]; B[dg] -> from=cg, to=dg
    注释优先使用第二个节点的注释, 否则使用第一个。
    """
    moves: List[PairedMove] = []
    pending: Dict[str, Optional[Tuple[str, str]]] = {"B": None, "W": None}  # color -> (from, comment)

    for n in nodes:
        pen = pending.get(n.color)
        if pen is None:
            pending[n.color] = (n.coord, n.comment)
        else:
            from_coord, first_comment = pen
            comment = n.comment or first_comment or ""
            moves.append(PairedMove(color=n.color, from_sgf=from_coord, to_sgf=n.coord, comment=comment))
            pending[n.color] = None
    return moves


def read_game_str(sfg_text: str) -> Game:
    headers = parse_headers(sfg_text)
    nodes = parse_nodes(sfg_text)
    moves = pair_moves(nodes)
    return Game(headers=headers, nodes=nodes, moves=moves)


# Convenience helpers ---------------------------------------------------------

def replay_sfg_text(sfg_text: str, verbose: bool = False) -> ac.Board:
    game = read_game_str(sfg_text)
    board = game.to_board()
    if verbose:
        print("Headers:")
        for k, v in game.headers.items():
            print(f"  {k}: {v}")
        print("\nMoves (UCI):")
        for i, m in enumerate(game.moves, 1):
            print(f"  {i}. {m.color} {m.from_sgf}->{m.to_sgf}  ({m.uci()})  {m.comment}")
        print("\nFinal board:")
        print(board.unicode())
        print("FEN:", board.fen())
    return board
