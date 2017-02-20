#!/usr/bin/python3
# Zinc, a chess engine testing tool. Copyright 2016 lucasart.
#
# Zinc is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# Zinc is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function  # Python 2.7 compatibility
from __future__ import division  # Python 2.7 compatibility
{'You need Python 2.7+ or Python 3.1+'}  # Syntax error in earlier Python version

import collections
import datetime
import math
import multiprocessing
import subprocess
import time
import random
import fileinput
import csv
from scipy.optimize import rosen, differential_evolution

import chess
import chess.polyglot
import chess.pgn
import chess.syzygy
import chess.uci

# Parameters
Engines = [
    {'file': 'stockfish.exe', 'name': 'base', 'debug': False},
    {'file': 'stockfish.exe', 'name': 'test', 'debug': False},
]
Options = [
    {'Hash': 1, 'Threads': 1},
    {'Hash': 1, 'Threads': 1}
]
TimeControls = [
    {'depth': 13, 'nodes': None, 'movetime': None, 'time': None, 'inc': None,
        'movestogo': None},
    {'depth': 13, 'nodes': None, 'movetime': None, 'time': None, 'inc': None,
        'movestogo': None}
]
Draw = {'movenumber': 40, 'movecount': 8, 'score': 20}
Resign = {'movecount': 3, 'score': 500}
Openings = '2moves.epd'
BookDepth = 8  # None
PgnOut = 'out.pgn'
Chess960 = False
Games = 2
Concurrency = 3
RatingInterval = 2
Tournament = 'tuning'  # 'gauntlet'
SyzygyPath = None  # None

class UCIEngine():
    def __init__(self, engine):
        self.process = subprocess.Popen(engine['file'], stdout=subprocess.PIPE,
           stdin=subprocess.PIPE, universal_newlines=True, bufsize=1)
        self.name = engine['name']
        self.debug = engine['debug']
        self.options = []
#        if engine['name'] is 'test':
#            self.options = names

    def readline(self):
        line = self.process.stdout.readline().rstrip()
        if self.debug:
            print('{}({}) > {}'.format(self.name, self.process.pid, line))
        return line

    def writeline(self, string):
        if self.debug:
            print('{}({}) < {}'.format(self.name, self.process.pid, string))
        self.process.stdin.write(string + '\n')

    def uci(self):
        self.writeline('uci')
        while True:
            line = self.readline()
            if line.startswith('option name '):
                tokens = line.split()
                name = tokens[2:tokens.index('type')]
                self.options.append(' '.join(name))
            elif line == 'uciok':
                break

    def setoptions(self, options):
        for name in options:
            value = options[name]
            if type(value) is bool:
                value = str(value).lower()
            self.writeline('setoption name {} value {}'.format(name, value))

    def isready(self):
        self.writeline('isready')
        while self.readline() != 'readyok':
            pass

    def newgame(self):
        self.writeline('ucinewgame')

    def go(self, args):
        tokens = ['go']
        for name in args:
            if args[name] is not None:
                tokens += [name, str(args[name])]
        self.writeline(' '.join(tokens))

        score = None
        while True:
            line = self.readline()
            if line.startswith('info'):
                i = line.find('score ')
                if i != -1:
                    tokens = line[(i + len('score ')):].split()
                    assert len(tokens) >= 2
                    if tokens[0] == 'cp':
                        if len(tokens) == 2 or not tokens[2].endswith('bound'):
                            score = int(tokens[1])
                    elif tokens[0] == 'mate':
                        score = math.copysign(Resign['score'], int(tokens[1]))

            elif line.startswith('bestmove'):
                return line.split()[1], score


class TimeoutError(Exception):  # Python 2.7 compatibility
    pass


class Clock():
    def __init__(self, timeControl):
        self.timeControl = timeControl
        self.time = timeControl['time']
        self.movestogo = timeControl['movestogo']

    def consume(self, seconds):
        if self.time is not None:
            self.time -= seconds
            if self.time < 0:
                raise TimeoutError
            if self.timeControl['inc']:
                self.time += self.timeControl['inc']

        if self.movestogo is not None:
            self.movestogo -= 1
            if self.movestogo <= 0:
                self.movestogo = self.timeControl['movestogo']
                if self.timeControl['time']:
                    self.time += self.timeControl['time']


def play_move(uciEngine, clocks, turnIdx, whiteIdx):
    def to_msec(seconds):
        return int(seconds * 1000) if seconds is not None else None

    startTime = time.time()

    bestmove, score = uciEngine.go({
        'depth': clocks[turnIdx].timeControl['depth'],
        'nodes': clocks[turnIdx].timeControl['nodes'],
        'movetime': clocks[turnIdx].timeControl['movetime'],
        'wtime': to_msec(clocks[whiteIdx].time),
        'btime': to_msec(clocks[whiteIdx ^ 1].time),
        'winc': to_msec(clocks[whiteIdx].timeControl['inc']),
        'binc': to_msec(clocks[whiteIdx ^ 1].timeControl['inc']),
        'movestogo': clocks[turnIdx].movestogo
    })

    elapsed = time.time() - startTime
    clocks[turnIdx].consume(elapsed)

    return bestmove, score


def play_game(uciEngines, fen, whiteIdx, timeControls, tablebases, returnPgn=False,
        pgnRound=None):
    scores = 0
    board = chess.Board(fen, Chess960)
    turnIdx = whiteIdx ^ (board.turn == chess.BLACK)
    clocks = [Clock(timeControls[0]), Clock(timeControls[1])]

    for e in uciEngines:
        e.newgame()

    drawPlyCnt, resignPlyCnt = 0, 0
    lostOnTime, wdl = None, None
    posCmd = ['position fen', fen]

    while len(posCmd) < 80:
        uciEngines[turnIdx].writeline(' '.join(posCmd))
        uciEngines[turnIdx].isready()

        if board.halfmove_clock == 0:
            wdl = tablebases.probe_wdl(board)
            if wdl is not None:
                break

        try:
            bestmove, score = play_move(uciEngines[turnIdx], clocks, turnIdx, whiteIdx)
            if uciEngines[1]:
                scores += score
        except TimeoutError:
            lostOnTime = turnIdx
            break

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

        if board.move_stack:
            posCmd.append(bestmove)
        else:
            posCmd += ['moves', bestmove]

        board.push_uci(bestmove)
        turnIdx ^= 1

    result, reason = board.result(True), 'chess rules'
    if result == '*':
        if lostOnTime is not None:
            result = '1-0' if lostOnTime == whiteIdx else '0-1'
            reason = 'lost on time'
        elif resignPlyCnt >= 2 * Resign['movecount']:
            reason = 'resign'
            if score > 0:
                result = '1-0' if board.turn == chess.WHITE else '0-1'
            else:
                result = '0-1' if board.turn == chess.WHITE else '1-0'
        elif wdl is not None:
            reason = 'tb adjudication'
            if wdl == -2:
                result = '1-0' if board.turn == chess.WHITE else '0-1'
            elif wdl == 2:
                result = '0-1' if board.turn == chess.WHITE else '1-0'
            else:
                result = '1/2-1/2'
        else:
            result = '1/2-1/2'
            reason = 'draw adjudication'

    if returnPgn:
        game = chess.pgn.Game.from_board(board)
        game.headers['White'] = uciEngines[whiteIdx].name
        game.headers['Black'] = uciEngines[whiteIdx ^ 1].name
        game.headers['Result'] = result
        game.headers['Date'] = datetime.date.today().isoformat()
        game.headers['Round'] = pgnRound
        game.headers['FEN'] = fen
        exporter = chess.pgn.StringExporter(variations=False, comments=False)
        pgnText = game.accept(exporter)
        pgnText += '\n{{{}}}'.format(reason)
    else:
        pgnText = None

    # Return numeric score, from engine #0 perspective
    scoreWhite = 1.0 if result == '1-0' else (0 if result == '0-1' else 0.5)
    return result, scores, pgnText


def print_score(engines, scores):
    N = scores
    if N >= 2:
        mean = sum(scores) / N
        variance = sum((x - mean)**2 for x in scores) / (N - 1)
        margin = 1.96 * math.sqrt(variance / N)
        print('score of {} vs. {} = {:.2f}% +/- {:.2f}%'.format(
            engines[0]['name'], engines[1]['name'], 100*(mean), 100*margin))
        with open('results1.csv', 'a') as f:
            print('{:}'.format(int(4*(mean-0.5))), file=f, end='\n')


def run_pool(engines, fens, tablebases, timeControls, concurrency, pgnOut, locpars):
    # I/O objects for the process pool
    jobQueue = multiprocessing.Queue()
    resultQueue = multiprocessing.Queue()

    # Prepare the processes
    processes = []
    assert len(engines) == 2  # Tournaments should be managed by the caller
    for i in range(concurrency):
        process = multiprocessing.Process(target=play_games,
            args=(engines, jobQueue, resultQueue, tablebases, pgnOut, locpars))
        processes.append(process)

    # Prepare the jobQueue
    for idx, fen in enumerate(fens):
        jobQueue.put(Job(round=idx+1, fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', white=idx % 2))

    # Insert 'None' padding values as a stopping buffer
    for i in range(concurrency):
        jobQueue.put(None)

    try:
        for p in processes:
            p.start()

        scores = 0
        for i in range(0, len(fens)):
            r = resultQueue.get()
#            print(r.display)
            scores += r.score

#            if (i+1) % RatingInterval == 0:
#                print_score(engines, scores)

            if pgnOut:
                with open(pgnOut, 'a') as f:
                    print(r.pgnText, file=f, end='\n\n')

        for p in processes:
            p.join()

        return scores

    except KeyboardInterrupt:
        print_score(engines, scores)


def init_engine(engine, options):
    uciEngine = UCIEngine(engine)
    uciEngine.uci()

    for name in options:
        if name not in uciEngine.options:
            print('warning: "{}" is not a valid UCI Option for engine "{}"'
                .format(name, uciEngine.name))

    uciEngine.setoptions(options)

    if Chess960:
        uciEngine.setoptions({'UCI_Chess960': True})

    uciEngine.isready()

    return uciEngine


def play_games(engines, jobQueue, resultQueue, tablebases, pgnOut, locpars):
    try:
        uciEngines = []
        Parameters = []
        
#        for i, e in enumerate(engines):
        Parameters = set_parameters(Pars)
        uciEngines.append(init_engine(Engines[0], Options[0]))
        uciEngines.append(init_engine(Engines[1], Parameters))

        while True:
            # HACK: We can't just test jobQueue.empty(), then run jobQueue.get(). Between
            # both operations, another process could steal a job from the queue. That's
            # why we insert some padding 'None' values at the end of the queue
            job = jobQueue.get()
            if job is None:
                return

            result, scores, pgnText = play_game(uciEngines, job.fen, job.white,
                TimeControls, tablebases, pgnOut, job.round)

            display = 'Game #{} ({} vs. {}): {}'.format(
                job.round, engines[job.white]['name'],
                engines[job.white ^ 1]['name'], result)

            resultQueue.put(Result(score=scores, display=display, pgnText=pgnText))

    except KeyboardInterrupt:
        pass

def get_pars():
  params = []
  f = open('result.csv')
  lines = f.read().split('\n')
  if lines[-1] == '':
    lines.remove('')

  for p in lines:
    params.append(p.split())
  
  return params

def get_parameters():
    with open('result.txt', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        params = map(tuple, csvreader)
        Parameters = dict(params)
    
    return Parameters

Pars = get_pars()

def Array2Pars(parsArray):
  locpars = Pars[:]
  for n, par in enumerate(locpars):
    locpars[n][1] = int(parsArray[n])
  return locpars

def Pars2Array(pars):
  parsArray = [int(par[1]) for par in pars]  
  return parsArray

def fitness(locpars):
    scores = run_pool([Engines[0], Engines[1]], fens, tablebases, TimeControls, Concurrency, PgnOut, locpars)
    print(scores)
    return scores
  
def getBounds():
  return [(-100, 100) for p in Pars]

def set_parameters(pars):
    with open('result.txt', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        params = map(tuple, csvreader)
        Parameters = dict(params)
    values = Pars2Array(pars)
    Parameters = dict(zip(Parameters.keys(), values))
    opts = tuple(Parameters.items()) + tuple(Options[1].items())
    Parameters = dict(opts)
    return Parameters

def statusMsg(xk, convergence = 1):
  newPars = Array2Pars(xk)
  print
  for p in newPars:
    print( p[0],p[1])
  print
  return False

def choose_fens():
    fens = []

    if Openings.endswith('.epd'):  # EPD
        lines = []
        with open(Openings, 'r') as f:
            lines = f.readlines()
        for i in range(0,Games,2):
            random_line_num = random.randrange(0,len(lines))
            fen = lines[random_line_num].rstrip().split(';')[0]
            if fen == '':
                f.seek(0)
            else:
                fens.append(fen)
                if i + 1 < Games:
                    fens.append(fen)
        return fens

    elif Openings.endswith('.pgn'):  # PGN
        games = []
        with open(Openings, 'r') as f:
            game = chess.pgn.read_game()
            games.append(game)
        for i in range(0,Games,2):
            random_game_num = random.randrange(0,len(games))
            fen = lines[random_line_num].rstrip().split(';')[0]
            if fen == '':
                f.seek(0)
            else:
                fens.append(fen)
                if i + 1 < Games:
                    fens.append(fen)
        return fens

    else:  # PolyGlot
        assert Openings.endswith('.bin')
        with chess.polyglot.open_reader(Openings) as book:
            for i in range(0, Games, 2):
                board = chess.Board(chess960=Chess960)
                while (BookDepth is None) or (board.fullmove_number <= BookDepth):
                    board.push(book.weighted_choice(board).move(Chess960))
                fen = board.fen()
                print(fen)
                fens.append(fen)
                if i + 1 < Games:
                    fens.append(fen)

        return fens

Job = collections.namedtuple('Job', 'round fen white')
Result = collections.namedtuple('Result', 'score display pgnText')

if __name__ == '__main__':
    fens = choose_fens()
    with chess.syzygy.open_tablebases(SyzygyPath) as tablebases:
        # Run the tournament
        assert len(Engines) >= 2
        if Tournament == 'gauntlet':
            for e in Engines[1:]:
                run_pool([Engines[0], e], fens, tablebases, TimeControls, Concurrency, PgnOut)
        elif Tournament == 'round-robin':
            for i in range(len(Engines) - 1):
                for e in Engines[i+1:]:
                    run_pool([Engines[i], e], fens, tablebases, TimeControls, Concurrency, PgnOut)
        else:
            assert len(Engines) == 2
            assert Tournament == 'tuning'
            f = fitness(Pars2Array(Pars))
            print( '\n' +  'Reference correlation: ' + str(-f))
            res = differential_evolution(fitness, getBounds(), disp=True, tol = 5,
                callback = statusMsg, popsize = 20, strategy = 'best1bin', init = 'random', polish = False)
            statusMsg(res.x)
            print( 'Search/eval correlation: ', -res.fun)
