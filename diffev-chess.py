import timeit
import subprocess
import random
import numpy as np
import scipy as sp
import math
import re
import chess
from chess import uci
from chess import Board
from chess import Move
from chess import syzygy
from numpy import sqrt
from scipy.stats import norm
from scipy.optimize import differential_evolution

Engines = [
    {'file': 'C:\\msys2\\home\\lanto\\safechecks-tune\\stockfish.exe', 'name': 'test'},
    {'file': 'C:\\msys2\\home\\lanto\\safechecks-tune\\stockfish.exe', 'name': 'base'}
]

Draw = {'movenumber': 40, 'movecount': 8, 'score': 20}
Resign = {'movecount': 3, 'score': 400}
Openings = 'C:\\Cutechess\\2moves.epd'
Games = 25
Syzygy = 'C:\\Winboard\\Syzygy'
ParametersFile = 'safechecks.txt'
LogFile = 'tuning.txt'

Options = {'Clear Hash': True, 'Hash': 16, 'SyzygyPath': Syzygy,
           'SyzygyProbeDepth': 10, 'Syzygy50MoveRule': True, 'SyzygyProbeLimit': 5}

# Preparatory phase

# takes parameters from the engine

# takes parameters from file that is copied from engine output

def get_pars():
    params = []
    f = open(ParametersFile)
    lines = f.read().split('\n')
    if lines[-1] == '':
        lines.remove('')
    for p in lines:
        params.append(p.split(','))

    return params

Pars = get_pars()

# openings


def get_fens():
    fens = []
    lines = open(Openings).read().splitlines()
    for i in range(0, Games, 1):
        fen = random.choice(lines)
        fens.append(fen)
#  print(fens)
    return fens

def init_engines(pars):
    info_handlers = []
    uciEngines = []
    for e in Engines:
        uciEngines.append(uci.popen_engine(e['file']))

    for e, u in enumerate(uciEngines):
        u.uci()
        u.setoption(Options)
        u.setoption(pars[e])
        u.isready()

    return uciEngines


class DifferentialEvolution():

    def __init__(self):
        self.nameArray = [str(par[0]) for par in Pars]
        self.bounds = [(p[2], p[3]) for p in Pars]
        self.trial = [par[1] for par in Pars]

# Evaluation
    def evaluate(self, x):
        fens = get_fens()
        current = dict(zip(self.nameArray, np.rint(x)))
        trial = dict(zip(self.nameArray, self.trial))
        result = []
        with syzygy.open_tablebases(Syzygy) as tablebases:
            for fen in fens:
                result1 = self.trans_result(self.launchSf(
                    [current, trial], fen, tablebases,))
                result2 = self.trans_result(self.launchSf(
                    [trial, current], fen, tablebases,))
                result.append(result1 + result2)
        pentares = self.pentanomial(result)
        curr1 = pentares  # Markov process
        tri1 = pentares[::-1]
        curr0 = self.calc_los(curr1)
        tri0 = self.calc_los(tri1)
        print(1.0-curr0,x)
        return 1.0-curr0

    def trans_result(self, score):
        return {'1-0': 2, '1/2-1/2': 1, '0-1': 0}[score]

    def pentanomial(self, result):
        pentares = []
        for i in range(0, 5):
            pentares.append(result.count(i))
        return pentares

    def calc_los(self, pentares):
        sumi, sumi2 = 0, 0
        for e,i in zip(pentares,[0,0.5,1,1.5,2]):
#            res = 0.5 * i
            N = sum(pentares)
            sumi += e * i / N
            sumi2 += e * i * i / N
        sigma = math.sqrt(sumi2 - sumi * sumi)
        try:
          t0 = (sumi - 1) / sigma
        except ZeroDivisionError:
          t0 = (sumi - 1)
        los = norm.cdf(t0)
        return los 

# Game playing
    def launchSf(self, pars, fen, tablebases,):
        try:
            board = Board(fen, chess960=False)
        except BaseException:
            try:
                board.set_epd(fen)
            except BaseException:
                board = Board(chess960=False)
        wdl = None
        drawPlyCnt, resignPlyCnt = 0, 0
        whiteIdx = 1
        turnIdx = whiteIdx ^ (board.turn == chess.BLACK)
        uciEngines = init_engines(pars)
        info_handler = uci.InfoHandler()
        for u in uciEngines:
            u.info_handlers.append(info_handler)
            u.ucinewgame()

        while (not board.is_game_over(claim_draw=True)):

            if board.castling_rights == 0:

                try:
                    wdl = tablebases.probe_wdl(board)
                    if wdl is not None:
                        break
                except KeyError:
                    pass                           # < 1 ms

            uciEngines[turnIdx].position(board)
            bestmove, score = uciEngines[turnIdx].go(depth=7)
            score = info_handler.info["score"][1].cp

            if score is not None:
                    # Resign adjudication
                if abs(score) >= Resign['score']:
                    resignPlyCnt += 1
                    if resignPlyCnt >= 2 * Resign['movecount']:
                        break
                else:
                    resignPlyCnt = 0

                # Draw adjudication
                if abs(score) <= Draw['score'] and board.halfmove_clock > 0:
                    drawPlyCnt += 1
                    if drawPlyCnt >= 2 * Draw['movecount'] \
                            and board.fullmove_number >= Draw['movenumber']:
                        break
                else:
                    drawPlyCnt = 0
            else:
                # Disable adjudication over mate scores
                drawPlyCnt, resignPlyCnt = 0, 0

            board.push(bestmove)
            turnIdx ^= 1

        result = board.result(True)
        if result == '*':
            if resignPlyCnt >= 2 * Resign['movecount']:
                if score > 0:
                    result = '1-0' if board.turn == chess.WHITE else '0-1'
                else:
                    result = '0-1' if board.turn == chess.WHITE else '1-0'
            elif wdl is not None:
                if wdl <= -1:
                    result = '1-0' if board.turn == chess.WHITE else '0-1'
                elif wdl >= 1:
                    result = '0-1' if board.turn == chess.WHITE else '1-0'
                else:
                    result = '1/2-1/2'
            else:
                result = '1/2-1/2'

        for u in uciEngines:
            u.quit(0)
        return result
        exit(0)

if __name__ == '__main__':
    de = DifferentialEvolution()
    result = differential_evolution(de.evaluate,bounds=de.bounds,polish=False,workers=4)
    result.x, result.fun
