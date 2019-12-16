import chess
from chess import pgn

def pgn_compare():

### Enter the directory of your pgn file below. If it contains FENs,
### those should be full (6 items) otherwise an error is raised
    pgn = open("C:\\cutechess\\results.pgn")
    offsets = chess.pgn.scan_offsets(pgn)
    pairs = []
    number = 0
    for offset in offsets:
        pgn.seek(offset)
        game = chess.pgn.read_game(pgn)
        main = game.main_line()
        board = game.board()
        variation = board.variation_san(main)
        pairs.append(variation)

    for i in range(0, len(pairs[:-1]), 2):
        game_odd, game_even = pairs[i], pairs[i+1]
        if game_odd != game_even:

### The output is the numbers of the games in a pair that eventually differs [Round: "#"]
### and the number of differing pairs.
            print(i+1, i+2)
            number += 1
    print("The number of differing pairs is " + str(number))

if __name__ == '__main__':
    pgn_compare()