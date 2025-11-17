# Jungle Chess Engine Interface
# 简化版引擎通信接口，适配斗兽棋
# 基于 python-chess engine.py 简化而来

"""
斗兽棋引擎通信接口

提供与斗兽棋引擎的基本通信功能，支持：
- 简化的 UCI 风格协议
- 基本的走棋和分析功能
- 同步和异步接口
"""
"""
尚未实现大部分功能
"""

import asyncio
import logging
import subprocess
import threading
import typing
from typing import Any, Dict, List, Optional, Tuple, Union

import jungle_chess as jc

LOGGER = logging.getLogger(__name__)


class EngineError(RuntimeError):
    """引擎运行时错误"""
    pass


class EngineTerminatedError(EngineError):
    """引擎进程意外退出"""
    pass


class Limit:
    """搜索限制条件"""
    
    def __init__(self,
                 time: Optional[float] = None,
                 depth: Optional[int] = None,
                 nodes: Optional[int] = None):
        """
        初始化搜索限制
        
        :param time: 思考时间（秒）
        :param depth: 搜索深度（层数）
        :param nodes: 搜索节点数
        """
        self.time = time
        self.depth = depth
        self.nodes = nodes
    
    def __repr__(self) -> str:
        parts = []
        if self.time is not None:
            parts.append(f"time={self.time!r}")
        if self.depth is not None:
            parts.append(f"depth={self.depth!r}")
        if self.nodes is not None:
            parts.append(f"nodes={self.nodes!r}")
        return f"Limit({', '.join(parts)})"


class Score:
    """局面评分（厘子为单位，100 = 1子优势）"""
    
    def __init__(self, cp: int) -> None:
        self.cp = cp
    
    def __str__(self) -> str:
        return f"+{self.cp}" if self.cp > 0 else str(self.cp)
    
    def __repr__(self) -> str:
        return f"Score({self})"
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Score):
            return self.cp == other.cp
        return NotImplemented
    
    def __lt__(self, other: object) -> bool:
        if isinstance(other, Score):
            return self.cp < other.cp
        return NotImplemented


class InfoDict(typing.TypedDict, total=False):
    """引擎返回的信息字典"""
    score: Score          # 评分
    depth: int           # 搜索深度
    nodes: int           # 搜索节点数
    time: float          # 用时（秒）
    pv: List[jc.Move]    # 主要变着
    nps: int             # 每秒节点数


class PlayResult:
    """走棋结果"""
    
    def __init__(self,
                 move: Optional[jc.Move],
                 info: Optional[InfoDict] = None):
        self.move = move
        self.info: InfoDict = info or {}
    
    def __repr__(self) -> str:
        return f"<PlayResult move={self.move} info={self.info}>"


class SimpleEngine:
    """
    简化的斗兽棋引擎接口（同步）
    
    示例用法：
        engine = SimpleEngine.popen("./jungle_engine")
        result = engine.play(board, Limit(time=1.0))
        print(f"引擎走棋: {result.move.uci()}")
        engine.quit()
    """
    
    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.initialized = False
    
    def _send(self, command: str) -> None:
        """发送命令到引擎"""
        LOGGER.debug(f"<< {command}")
        if self.process.stdin:
            self.process.stdin.write(f"{command}\n".encode('utf-8'))
            self.process.stdin.flush()
    
    def _receive(self) -> str:
        """接收引擎的一行输出"""
        if self.process.stdout:
            line = self.process.stdout.readline().decode('utf-8').strip()
            LOGGER.debug(f">> {line}")
            return line
        return ""
    
    def initialize(self) -> None:
        """初始化引擎"""
        self._send("junglechess")  # 自定义协议标识
        self._send("isready")
        
        while True:
            line = self._receive()
            if line == "readyok":
                self.initialized = True
                break
            elif line.startswith("id "):
                # 引擎识别信息
                pass
    
    def play(self, 
             board: jc.Board, 
             limit: Limit,
             info: bool = False) -> PlayResult:
        """
        让引擎走棋
        
        :param board: 当前棋盘局面
        :param limit: 搜索限制
        :param info: 是否返回详细信息
        :return: PlayResult 包含最佳着法
        """
        if not self.initialized:
            raise EngineError("引擎未初始化")
        
        # 发送局面
        self._send(f"position fen {board.fen()}")
        
        # 发送搜索命令
        go_cmd = "go"
        if limit.time is not None:
            go_cmd += f" movetime {int(limit.time * 1000)}"
        if limit.depth is not None:
            go_cmd += f" depth {limit.depth}"
        if limit.nodes is not None:
            go_cmd += f" nodes {limit.nodes}"
        
        self._send(go_cmd)
        
        # 解析响应
        result_info: InfoDict = {}
        best_move = None
        
        while True:
            line = self._receive()
            
            if line.startswith("info "):
                # 解析信息行
                if info:
                    result_info = self._parse_info(line, board)
            
            elif line.startswith("bestmove "):
                # 解析最佳着法
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        best_move = jc.Move.from_uci(parts[1])
                    except ValueError:
                        pass
                break
        
        return PlayResult(best_move, result_info)
    
    def _parse_info(self, line: str, board: jc.Board) -> InfoDict:
        """解析 info 行"""
        info: InfoDict = {}
        tokens = line.split()[1:]  # 跳过 "info"
        
        i = 0
        while i < len(tokens):
            if tokens[i] == "depth" and i + 1 < len(tokens):
                info["depth"] = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "nodes" and i + 1 < len(tokens):
                info["nodes"] = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "time" and i + 1 < len(tokens):
                info["time"] = int(tokens[i + 1]) / 1000.0
                i += 2
            elif tokens[i] == "nps" and i + 1 < len(tokens):
                info["nps"] = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "score" and i + 2 < len(tokens):
                if tokens[i + 1] == "cp":
                    info["score"] = Score(int(tokens[i + 2]))
                i += 3
            elif tokens[i] == "pv":
                # 解析主要变着
                pv = []
                temp_board = board.copy()
                for j in range(i + 1, len(tokens)):
                    try:
                        move = jc.Move.from_uci(tokens[j])
                        if move in list(temp_board.generate_pseudo_legal_moves()):
                            temp_board.push(move)
                            pv.append(move)
                        else:
                            break
                    except ValueError:
                        break
                info["pv"] = pv
                break
            else:
                i += 1
        
        return info
    
    def quit(self) -> None:
        """退出引擎"""
        self._send("quit")
        self.process.wait(timeout=2.0)
    
    def close(self) -> None:
        """关闭引擎进程"""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
    
    @classmethod
    def popen(cls, command: Union[str, List[str]]) -> "SimpleEngine":
        """
        启动引擎进程
        
        :param command: 引擎可执行文件路径或命令列表
        :return: SimpleEngine 实例
        """
        if isinstance(command, str):
            command = [command]
        
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        engine = cls(process)
        engine.initialize()
        return engine
    
    def __enter__(self) -> "SimpleEngine":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def __repr__(self) -> str:
        return f"<SimpleEngine pid={self.process.pid}>"


# 便捷函数
def popen_engine(command: Union[str, List[str]]) -> SimpleEngine:
    """
    快捷方式：启动并初始化引擎
    
    示例：
        engine = popen_engine("./my_jungle_engine")
    """
    return SimpleEngine.popen(command)


if __name__ == "__main__":
    # 测试示例（需要有实际的引擎可执行文件）
    print("斗兽棋引擎接口 - 使用示例:")
    print()
    print("# 启动引擎")
    print("engine = SimpleEngine.popen('./jungle_engine')")
    print()
    print("# 让引擎走棋")
    print("board = jc.Board()")
    print("result = engine.play(board, Limit(time=1.0))")
    print("print(f'最佳着法: {result.move.uci()}')")
    print()
    print("# 关闭引擎")
    print("engine.quit()")

