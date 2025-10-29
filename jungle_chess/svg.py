# SVG rendering for Animal Chess
# Simplified version for board visualization

import animalchess
import xml.etree.ElementTree as ET
from typing import Optional

SQUARE_SIZE = 60
MARGIN = 30

# 棋盘颜色
DEFAULT_COLORS = {
    "square light": "#f0d9b5",
    "square dark": "#b58863",
    "river": "#87CEEB",
    "trap": "#FF6B6B",
    "den": "#FFD700",
    "margin": "#212121",
}


class SvgWrapper(str):
    def _repr_svg_(self) -> "SvgWrapper":
        return self


def board(board: Optional[animalchess.Board] = None, 
          size: Optional[int] = None,
          coordinates: bool = True) -> str:
    """
    Renders the Animal Chess board as SVG.
    
    :param board: The board to render
    :param size: Size in pixels
    :param coordinates: Whether to show coordinates
    """
    margin = MARGIN if coordinates else 0
    width = 7 * SQUARE_SIZE + 2 * margin
    height = 9 * SQUARE_SIZE + 2 * margin
    
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "version": "1.1",
        "viewBox": f"0 0 {width} {height}",
    })
    
    if size is not None:
        svg.set("width", str(size))
        svg.set("height", str(int(size * height / width)))
    
    # 背景
    if coordinates:
        ET.SubElement(svg, "rect", {
            "x": "0",
            "y": "0",
            "width": str(width),
            "height": str(height),
            "fill": DEFAULT_COLORS["margin"],
        })
    
    # 绘制棋盘格子
    for rank in range(9):
        for file in range(7):
            x = file * SQUARE_SIZE + margin  # 从左到右：a-g
            y = (8 - rank) * SQUARE_SIZE + margin  # 从上到下：i-a
            
            sq = animalchess.square(file, rank)
            
            # 确定格子颜色
            if sq in animalchess.RIVER_SQUARES:
                fill_color = DEFAULT_COLORS["river"]
            elif sq in animalchess.WHITE_TRAPS or sq in animalchess.BLACK_TRAPS:
                fill_color = DEFAULT_COLORS["trap"]
            elif sq == animalchess.WHITE_DEN or sq == animalchess.BLACK_DEN:
                fill_color = DEFAULT_COLORS["den"]
            else:
                fill_color = DEFAULT_COLORS["square light"] if (file + rank) % 2 == 0 else DEFAULT_COLORS["square dark"]
            
            ET.SubElement(svg, "rect", {
                "x": str(x),
                "y": str(y),
                "width": str(SQUARE_SIZE),
                "height": str(SQUARE_SIZE),
                "fill": fill_color,
                "stroke": "#000",
                "stroke-width": "1",
            })
            
            # 绘制特殊标记
            if sq == animalchess.WHITE_DEN or sq == animalchess.BLACK_DEN:
                ET.SubElement(svg, "text", {
                    "x": str(x + SQUARE_SIZE / 2),
                    "y": str(y + SQUARE_SIZE / 2 + 8),
                    "text-anchor": "middle",
                    "font-size": "24",
                    "fill": "#000",
                }).text = "穴"
            elif sq in animalchess.WHITE_TRAPS or sq in animalchess.BLACK_TRAPS:
                ET.SubElement(svg, "text", {
                    "x": str(x + SQUARE_SIZE / 2),
                    "y": str(y + SQUARE_SIZE / 2 + 6),
                    "text-anchor": "middle",
                    "font-size": "16",
                    "fill": "#fff",
                }).text = "阱"
    
    # 绘制棋子
    if board is not None:
        for sq, piece in board.pieces_dict.items():
            file = animalchess.square_file(sq)
            rank = animalchess.square_rank(sq)
            
            x = file * SQUARE_SIZE + margin + SQUARE_SIZE / 2  # 从左到右：a-g
            y = (8 - rank) * SQUARE_SIZE + margin + SQUARE_SIZE / 2
            
            # 棋子圆形背景
            color_fill = "#DC143C" if piece.color == animalchess.WHITE else "#000"
            ET.SubElement(svg, "circle", {
                "cx": str(x),
                "cy": str(y),
                "r": str(SQUARE_SIZE * 0.35),
                "fill": color_fill,
                "stroke": "#fff",
                "stroke-width": "2",
            })
            
            # 棋子文字
            ET.SubElement(svg, "text", {
                "x": str(x),
                "y": str(y + 8),
                "text-anchor": "middle",
                "font-size": "20",
                "font-weight": "bold",
                "fill": "#fff",
            }).text = piece.chinese_name()[1:]  # 去掉颜色前缀
    
    # 坐标标记
    if coordinates:
        for file in range(7):
            x = file * SQUARE_SIZE + margin + SQUARE_SIZE / 2  # 从左到右：a-g
            # 底部
            ET.SubElement(svg, "text", {
                "x": str(x),
                "y": str(height - 5),
                "text-anchor": "middle",
                "font-size": "16",
                "fill": "#e5e5e5",
            }).text = chr(ord('a') + file)  # a, b, c, ..., g
            # 顶部
            ET.SubElement(svg, "text", {
                "x": str(x),
                "y": str(20),
                "text-anchor": "middle",
                "font-size": "16",
                "fill": "#e5e5e5",
            }).text = chr(ord('a') + file)  # a, b, c, ..., g
        
        for rank in range(9):
            y = (8 - rank) * SQUARE_SIZE + margin + SQUARE_SIZE / 2 + 5
            row_label = chr(ord('a') + rank)  # a, b, c, ..., i
            # 左侧
            ET.SubElement(svg, "text", {
                "x": str(15),
                "y": str(y),
                "text-anchor": "middle",
                "font-size": "16",
                "fill": "#e5e5e5",
            }).text = row_label
            # 右侧
            ET.SubElement(svg, "text", {
                "x": str(width - 15),
                "y": str(y),
                "text-anchor": "middle",
                "font-size": "16",
                "fill": "#e5e5e5",
            }).text = row_label
    
    return SvgWrapper(ET.tostring(svg, encoding="unicode"))


def piece(piece: animalchess.Piece, size: Optional[int] = None) -> str:
    """Renders a single piece as SVG."""
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "version": "1.1",
        "viewBox": f"0 0 {SQUARE_SIZE} {SQUARE_SIZE}",
    })
    
    if size is not None:
        svg.set("width", str(size))
        svg.set("height", str(size))
    
    color_fill = "#DC143C" if piece.color == animalchess.WHITE else "#000"
    
    ET.SubElement(svg, "circle", {
        "cx": str(SQUARE_SIZE / 2),
        "cy": str(SQUARE_SIZE / 2),
        "r": str(SQUARE_SIZE * 0.4),
        "fill": color_fill,
        "stroke": "#fff",
        "stroke-width": "2",
    })
    
    ET.SubElement(svg, "text", {
        "x": str(SQUARE_SIZE / 2),
        "y": str(SQUARE_SIZE / 2 + 8),
        "text-anchor": "middle",
        "font-size": "24",
        "font-weight": "bold",
        "fill": "#fff",
    }).text = piece.chinese_name()[1:]
    
    return SvgWrapper(ET.tostring(svg, encoding="unicode"))

