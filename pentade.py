import sys
import subprocess
import random
import numpy as np
import scipy as sp
import math
import re
import scipy.stats
import chess
import collections
import operator
from chess import uci
from chess import Board
from chess import Move
from chess import syzygy
from numpy import sqrt
from scipy.stats import chi2
from scipy.stats import norm
from statistics import median
from decimal import *

Engines = [
    {'file': 'C:\\msys2\\home\\lanto\\material\\tune.exe', 'name': 'test'},
    {'file': 'C:\\msys2\\home\\lanto\\material\\tune.exe', 'name': 'base'}
    ]

Draw = {'movenumber': 40, 'movecount': 8, 'score': 20}
Resign = {'movecount': 3, 'score': 400}
population_size=40
iterations=200
dynamic_rate=5
Openings = 'C:\\Cutechess\\2moves.epd'
Games = 10
UseEngine = False
Syzygy = 'C:\\Winboard\\Syzygy'
ParametersFile = 'C:\\Rockstar\\quadratic.txt'

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
  
  return params

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
#    for u in self.engines:
#      if (not u.is_alive()):
#        self.engines = init_engines()

#    uciEngines = self.engines
  uciEngines = []
  for e in Engines:
    uciEngines.append(uci.popen_engine(e['file']))

  for u in uciEngines:
    u.uci()
    u.setoption(Options)
    u.setoption(pars[uciEngines.index(u)])
    u.isready()

  return uciEngines

class DifferentialEvolution():

  def __init__(self, F=0.5, CR=0.9, JR=None):
    self.params = Pars
    self.nameArray = [str(par[0]) for par in Pars]
    self.parsArray = [int(par[1]) for par in Pars]
    self.bounds = [(int(p[2]), int(p[3])) for p in Pars]
    self.lbounds = [int(l) for l, h in self.bounds]
    self.hbounds = [int(h) for l, h in self.bounds]
    self.n_parameters = len(self.nameArray)
    self.f = F
    self.cr = CR
    self.jr = JR
    self.current = self.initialize()
    self.population = [[00.00,[0,0,0,0,0],0,p] for p in self.current]
    self.training = (np.array([self.lbounds,]*population_size) + 
      np.array([self.hbounds,]*population_size) - np.array(self.current)).tolist()
    self.trial = [[00.00,[0,0,0,0,0],0,p] for p in self.training]
    self.history = self.population
    self.current_matrix = []
    self.diagonal = []
#    self.engines = init_engines()

  def getBounds():
    return [(int(p[2]), int(p[3])) for p in Pars]

  def initialize(self):
    initialized = []
    for i in range(population_size):
      randArray = [random.randint(b[0],b[1]) for b in self.bounds]
      initialized.append(randArray)
    return initialized

###  Evaluation
  def evaluate(self):
    num = 0
    population = []
    fens = get_fens()
    for curr, tri in zip(shuffled(self.population), shuffled(self.trial)):
      current = dict(zip(self.nameArray, curr[3]))
      trial = dict(zip(self.nameArray, tri[3]))
      result = []
      with chess.syzygy.open_tablebases(Syzygy) as tablebases:
        for fen in fens:
          result1 = self.trans_result(self.launchSf([current, trial], fen, tablebases,))
          result2 = self.trans_result(self.launchSf([trial, current], fen, tablebases,))
          result.append(result1 + result2)
      pentares = self.pentanomial(result)
      curr[1] = (np.array(curr[1]) + np.array(pentares)).tolist()
      tri[1] = (np.array(tri[1]) + np.array(pentares[::-1])).tolist()
      curr[0] = float(self.calc_los(curr[1]))
      tri[0] = float(self.calc_los(tri[1]))
      num += 1
      print('{0:3d}  {1:7.2f} | {2:7.2f}'.format(num, curr[0], tri[0]))
      curr[2] = sum(curr[1])
      tri[2] = sum(tri[1])

###  Selection
      if curr[0] < 0.0:
#      if random.uniform(0,1) < 0.5:
        population.append(tri)
      else:
        population.append(curr)
    self.population = sorted(population, key=operator.itemgetter(0))
    self.history = self.updateHistory()
#    print(self.population)
    for member in self.history[-5:][::-1]:
      print('{0:6.2f}, {1!s}, {2:3d}, {3!s}'.format(member[0],
        member[1], member[2], member[3]))
#    with open('tuning1.txt', 'a') as f:
#      f.write(str(self.population) + '\n' + str(self.history) + '\n')

###  History update
  def updateHistory(self):
    for popu in self.population:
      if str(popu[3]) not in str(self.history):
        self.history.append(popu)
      else:
        for hist in self.history:
          if str(popu[3]) in str(hist[3]):
            if popu[2] > hist[2]:
              hist = popu

#    self.history = [popu if (str(x[3]) in str(popu[3]) \
#      and x[2] < popu[2]) else x for x in self.history]

#    self.history = sorted(self.history, key=lambda games: games[2])
#    self.history = sorted(self.history, key=lambda fitness: fitness[0])[-population_size:]
    self.history = sorted(self.history, key=operator.itemgetter(0))[-population_size:]
    return self.history
    
  def trans_result(self, score):
    return {'1-0': 2, '1/2-1/2': 1, '0-1': 0}[score]

  def pentanomial(self, result):
    pentares = []
    for i in range(0,5):
      pentares.append(result.count(i))
#    print(pentares)
    return pentares

  def calc_los(self, pentares):
    sumi, sumi2 = 0, 0
    for i in range(0,5):
      res = 0.5 * i
      N = sum(pentares)
      sumi += pentares[i] * res / N
      sumi2 += pentares[i] * res * res / N
    sigma = math.sqrt(sumi2 - sumi * sumi)
#    t = math.sqrt(0.5 * N) * (sumi - 1) / sigma
#    los = norm.cdf(t)
#    print(sumi, sumi2, sigma, t)
    t = math.sqrt(2) * (sumi - 1) * 0.5 * 100 / sigma
    return '{0:.2f}'.format(t)

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

    try:
      while (not board.is_game_over(claim_draw=True)):

        if board.castling_rights == 0:
          if len(re.findall(r"[rnbqkpRNBQKP]", board.board_fen())) < 6:
            wdl = tablebases.probe_wdl(board)
            if wdl is not None:
              break

        uciEngines[turnIdx].position(board)
        bestmove, score = uciEngines[turnIdx].go(depth=4)
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
    except (MemoryError, SystemError, KeyboardInterrupt,
    OverflowError, OSError, ResourceWarning):
      for u in uciEngines:
        u.quit(1)
#    print(result)
    return result
    exit(0)

###  Mutation
  def mutate(self):
    self.trial = []
    best_individuum = self.population[-1]
    if self.f is None:
      use_f = random.uniform(0.5,1.5)
    else: 
      use_f = self.f
    for curr in shuffled(self.population):
      indices = random.sample(range(0,population_size), 2)
      r1 = best_individuum
      r2 = self.population[indices[0]]
      r3 = self.population[indices[1]]
      mutant = np.array(r1[3]) + use_f*(np.array(r2[3]) - np.array(r3[3]))

###  Crossover
      for j in range(0, self.n_parameters):
        if random.uniform(0,1) <= self.cr or j == random.randrange(0, self.n_parameters):
          mutant[j] = mutant[j]
        else:
          mutant[j] = curr[3][j]
        if mutant[j] < self.lbounds[j]:
          mutant[j] = 2*self.lbounds[j] - mutant[j]
        if mutant[j] > self.hbounds[j]:
           mutant[j] = 2*self.hbounds[j] - mutant[j]
#        print(mutant)
      self.trial.append([00.00,[0,0,0,0,0],0,mutant.astype(int).tolist()])

###  History injection
    for hist in self.history[-int(population_size / 5):]:
      if str(hist[3]) not in str(self.population) and str(hist[3]) not in str(self.trial):
        j_rand = random.randrange(0, population_size)
        self.trial[j_rand] = hist

###  Dynamic opposition
    self.current = [p[3] for p in self.population]
    self.current_matrix = np.append(self.current_matrix, self.current)
    if self.jr is not None and random.uniform(0,1) < self.jr:
      self.current_matrix = np.array(self.current)
      self.stats_analysis()
    elif self.jr is None and (g+1) % dynamic_rate == 0 and g != 0:
      self.current_matrix = self.current_matrix.reshape((dynamic_rate)*population_size, \
        self.n_parameters)
      self.stats_analysis()

###  Statistical analysis and output
  def stats_analysis(self):
    self.training = np.array([self.lbounds,]*population_size) + \
      np.array([self.hbounds,]*population_size) - np.array(self.current[:])
    self.trial = [[00.00,[0,0,0,0,0],0,p.tolist()] for p in self.training]
    covar = np.round(np.cov(self.current_matrix.T), 2)
    means = np.mean(self.current_matrix, axis=0).astype(int)
    medians = np.median(self.current_matrix, axis=0).astype(int)
    self.lbounds = np.percentile(self.current_matrix, 0, axis=0).astype(int)
    self.hbounds = np.percentile(self.current_matrix, 100, axis=0).astype(int)
    if self.jr is None:
      if self.n_parameters > 1:
        self.diagonal = covar.diagonal()
        sum_variations = sum(self.diagonal)
        if min(map(abs, means)) != 0:
          coeff_var = [sqrt(p) / abs(q) for p,q in zip(self.diagonal, means)]
        else:
          coeff_var = [sqrt(p) for p,q in zip(self.diagonal, means)]
      else:
        self.diagonal = np.round(np.var(self.current_matrix.T),2)
        sum_variations = self.diagonal
        if abs(means) != 0:
          coeff_var = sqrt(sum_variations)/means
        else:
          coeff_var = sqrt(sum_variations)
#    np.set_printoptions(threshold=np.inf)
#    np.savetxt('stats.txt', self.current_matrix.T, delimiter=',')
#    with open('stats.txt', 'a') as f:
#      f.write(str(self.current_matrix.T))
    self.print_save(medians, sum_variations, coeff_var)
    self.current_matrix = []

  ## Formatting, printing and saving
  def print_save(self, medians, sum_variations, coeff_var):
    print('{0:.2f}'.format(sum_variations))
    for i, name in enumerate(self.nameArray):
      print('{0:22} {1:5d} {2:5d} {3:5d} {4:9.2f} {5:8.2%}'.format(name,
        medians[i], self.lbounds[i], self.hbounds[i],
          self.diagonal[i], coeff_var[i]))
    with open('C:\\Rockstar\\tuning.txt', 'a') as f:
      f.write('{0:.2f}'.format(sum_variations) + '\n')
      for member in self.history[-5:][::-1]:
        f.write('{0:6.2f}, {1!s}, {2:3d}, {3!s}\n'.format(member[0],
          member[1], member[2], member[3]))
      for i, name in enumerate(self.nameArray):
        f.write('{0:22} {1:5d} {2:5d} {3:5d} {4:9.2f} {5:8.2%}'.format(name,
          medians[i], self.lbounds[i], self.hbounds[i],
            self.diagonal[i], coeff_var[i]) + '\n')
    with open(ParametersFile, 'w') as f:
      for i, name in enumerate(self.nameArray):
        f.write('{0},{1},{2},{3}'.format(name,
          medians[i], self.lbounds[i], self.hbounds[i]) + '\n')


#    ti = tolerance_interval(sum_variations)
#    ci = self.confidence_interval(sum_variations)
#    self.lbounds = (means - ci).tolist()
#    self.hbounds = (means + ci).tolist()
#    lbounds = np.amin(np.array(self.current_matrix), axis=0).astype(int)
#    hbounds = np.amax(np.array(self.current_matrix), axis=0).astype(int)
#    self.lbounds = list(map(max, lbounds, self.lbounds))
#    self.hbounds = list(map(min, hbounds, self.hbounds))
#    intervals = [tolerance_interval(x) for x in zip(*self.current_matrix)]
#    self.lbounds = (means - ci).tolist()
#    self.hbounds = (means + ci).tolist()
#    print(str(medians) + '\n' + str(self.lbounds) + '\n' + str(self.hbounds))
#    print(sum_variations)
#    self.diagonal = [float('{0:.2f}'.format(x)) for x in self.diagonal]
#    print(self.diagonal)
#    coeff = [float('{0:.2f}'.format(x)) for x in coeff_var]
#    print(coeff)
#    with open('tuning1.txt', 'a') as f:
#      f.write(str(covar) + '\n' + str(sum_variations) + \
#      '\n' + str(medians) + '\n' + str(self.lbounds) + '\n' + \
#      str(self.hbounds) + '\n' + str(self.diagonal) + '\n' + str(coeff) +'\n')


if __name__ == '__main__':
  de = DifferentialEvolution()

  g = 0
  while g < iterations or sum(de.diagonal) < 1.0:
    de.evaluate()
    de.mutate()
    g+=1

  exit(0)

'''
def tolerance_interval(sum_variations):
  n = population_size * dynamic_rate
  dof = n - 1
  # specify data coverage
  prop = 0.999
  prop_inv = (1.0 - prop) / 2.0
  gauss_critical = norm.isf(prop_inv)
#    print('Gaussian critical value: %.3f (coverage=%d%%)' % (gauss_critical, prop*100))
  # specify confidence
  prob = 0.999
  chi_critical = chi2.isf(q=prob, df=dof)
#    print('Chi-Squared critical value: %.3f (prob=%d%%, dof=%d)' % (chi_critical, prob*100, dof))
  # tolerance
  tolerance = sqrt((dof * (1 + (1/n)) * gauss_critical**2) / chi_critical)
  tolerance_interval = tolerance * math.sqrt(sum_variations)
  return tolerance_interval


  def confidence_interval(self, sum_variations):
    n = population_size * dynamic_rate
    if self.n_parameters > 1:
      confidence_interval = list(map(lambda x: (math.sqrt(x) / math.sqrt(n) * \
          sp.stats.t._ppf((1+0.9999)/2., n-1)).astype(int), self.diagonal))
    else:
      confidence_interval = (math.sqrt(sum_variations) / n) * \
          sp.stats.t._ppf((1+0.99)/2., n-1).astype(int)
    return confidence_interval
'''
