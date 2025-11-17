import jungle_chess as jc

board = jc.Board("3L3/7/7/7/7/7/7/7/7 w 0 1")
print(board.unicode())
print(board.is_game_over())