import re
import csv
import itertools
import chess
from chess import Board
from chess import Move
from chess import pgn

def pgn_prepare():
    pgn = open("C:\\Rockstar\\neural-networks\\data\\TCEC_Season_9_-_Superfinal.pgn")
    tcec_offsets = chess.pgn.scan_offsets(pgn)
    all_posits = []
    all_evals = []
    for offset in tcec_offsets:
        moves = []
        fens = []
        pgn.seek(offset)
        game = chess.pgn.read_game(pgn)
        main = game.main_line()
        board = chess.Board()
        variation = board.variation_san(main)
        node = game
        while not node.is_end():
            next_node = node.variation(0)
            move = node.board().san(next_node.move)
            board = node.board()
            fens.append(board.fen())
            moves.append(move)
            node = next_node

        evals = re.findall(r'wv=(.*?),', str(game))
        start = len(moves) - len(evals)
        posits = fens[start:]
        all_evals.append(evals)
        all_posits.append(posits)

    all_evals = list(itertools.chain(*all_evals))
    all_posits = list(itertools.chain(*all_posits))
    print(len(all_posits))
    print(len(all_posits))
    return all_posits, all_evals
    pgn.close ()

if __name__ == '__main__':
    all_posits, all_evals = pgn_prepare()
    with open("tcec_fens.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in all_posits:
            writer.writerow([val])

    with open("tcec_evals.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in all_evals:
            writer.writerow([val])
