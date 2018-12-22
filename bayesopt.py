import timeit
import subprocess
import random
import numpy as np
import scipy as sp
import math
import re
import chess
from bayes_opt import BayesianOptimization
from operator import itemgetter
from chess import uci
from chess import Board
from chess import Move
from chess import syzygy
from numpy import sqrt
from scipy.stats import chi2
from scipy.stats import norm
from statistics import median

Engines = [
    {'file': 'C:\\msys2\\home\\lanto\\safechecks\\tune.exe', 'name': 'test'},
    {'file': 'C:\\msys2\\home\\lanto\\safechecks\\tune.exe', 'name': 'base'}
    ]

Draw = {'movenumber': 40, 'movecount': 8, 'score': 20}
Resign = {'movecount': 3, 'score': 400}
population_size=40
iterations=200
dynamic_rate=5
Openings = 'C:\\Cutechess\\2moves.epd'
Games = 50
UseEngine = False
Syzygy = 'C:\\Winboard\\Syzygy'
ParametersFile = 'C:\\Rockstar\\safechecks.txt'
LogFile = 'tuning.txt'
DynamicConstraints = True

Options = {'Clear Hash': True, 'Hash': 16, 'SyzygyPath': Syzygy, \
          'SyzygyProbeDepth': 10, 'Syzygy50MoveRule': True, 'SyzygyProbeLimit': 5}

##  Preparatory phase

# takes parameters from the engine
def getPars():
  sf = subprocess.Popen(Engines[0]['file'],  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
  sf.stdin.write('isready' + '\n')
  pars = []
  outline = []
  while outline is not '':
    outline = sf.stdout.readline().rstrip()
    if not (outline.startswith('Stockfish ') or outline.startswith('Unknown ') or outline == ''):
      pars.append(outline.split(','))
  sf.terminate()
  sf.wait()
  return pars

# takes parameters from file that is copied from engine output
def get_pars():
  params = []
  f = open(ParametersFile)
  lines = f.read().split('\n')
  if lines[-1] == '':
    lines.remove('')
  for p in lines:
    params.append(p.split(','))
  
  return sorted(params)

if UseEngine:
  Pars = getPars()
else:
  Pars = get_pars()

# openings
def get_fens():
  fens = []
  lines = open(Openings).read().splitlines()
  for i in range(0, Games, 1):
    fen =random.choice(lines)
    fens.append(fen)
#  print(fens)
  return fens

def shuffled(x):
  y = x[:]
  random.shuffle(y)
  return y

def init_engines(pars):
  info_handlers = []
  uciEngines = []
  for e in Engines:
    uciEngines.append(uci.popen_engine(e['file']))

  for e,u in enumerate(uciEngines):
    u.uci()
    u.setoption(Options)
    u.setoption(pars[e])
    u.isready()

  return uciEngines

class DifferentialEvolution():

  def __init__(self):
    self.nameArray = [str(par[0]) for par in Pars]
    self.parsArray = [int(par[1]) for par in Pars]
    self.bounds = [(int(p[2]), int(p[3])) for p in Pars]
    self.n_parameters = len(self.nameArray)
    self.current = [int(par[1]) for par in Pars]
    self.trial = [int(par[1]) for par in Pars]
    self.variables = dict(zip(self.nameArray, self.current))
    self.pbounds = dict(zip(self.nameArray, self.bounds))

###  Evaluation
  def evaluate(self, variables):
    num = 0
    fens = get_fens()
#    current = dict(zip(self.nameArray, variables))
    trial = dict(zip(self.nameArray, self.trial))
    result = []
    with syzygy.open_tablebases(Syzygy) as tablebases:
      for fen in fens:
        result1 = self.trans_result(self.launchSf([variables, trial], fen, tablebases,))
        result2 = self.trans_result(self.launchSf([trial, variables], fen, tablebases,))
        result.append(result1 + result2)
    pentares = self.pentanomial(result)
    curr = float(self.calc_los(pentares))
    return curr

  def trans_result(self, score):
    return {'1-0': 2, '1/2-1/2': 1, '0-1': 0}[score]

  def pentanomial(self, result):
    pentares = []
    for i in range(0,5):
      pentares.append(result.count(i))
    return pentares

  def calc_los(self, pentares):
    sumi, sumi2 = 0, 0
    for i in range(0,5):
      res = 0.5 * i
      N = sum(pentares)
      sumi += pentares[i] * res / N
      sumi2 += pentares[i] * res * res / N
    sigma = math.sqrt(sumi2 - sumi * sumi)
    try:
      t = math.sqrt(N) * (sumi - 1) / sigma * 100
    except ZeroDivisionError:
      t = 0.0
#    los = norm.cdf(t) * 100
#    return '{0:.2f}'.format(round(t, 2))
    return t

###  Game playing
  def launchSf(self, pars, fen, tablebases,):
    try:
      board = Board(fen,chess960=False)
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

#          if len(re.findall(r"[rnbqkpRNBQKP]", board.board_fen())) < 6:
#            wdl = tablebases.probe_wdl(board)
#            if wdl is not None:
#              break                       # ~ 1.5 ms

        try:
          wdl = tablebases.probe_wdl(board)
          if wdl is not None:
            break
        except KeyError:
          pass                           # < 1 ms

      uciEngines[turnIdx].position(board)
      bestmove, score = uciEngines[turnIdx].go(depth=9)
      score = info_handler.info["score"][1].cp
#        print(score)

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
#            print('tb draw')
      else:
        result = '1/2-1/2'
#          print('draw')

#    print(board.fen())
#    print(re.findall(r"[rnbqkpRNBQKP]", board.board_fen()))
    for u in uciEngines:
      u.quit(0)
#    print(result)
    return result
    exit(0)

if __name__ == '__main__':
  de = DifferentialEvolution()
  variables = dict(zip(de.nameArray, de.current))
  def black_box_function(**variables):
    f = de.evaluate(variables)
    return f
 
  optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=de.pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=0,
  )

  optimizer.maximize(
    init_points=2,
    n_iter=30,
    acq='poi',
  )
  print(optimizer.max)
