import random
import math
import numpy as np
from scipy.stats import norm
from bayes_opt import BayesianOptimization
from cutechess_batch import main

ParametersFile = 'nullmove.txt'

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

Pars = get_pars()

class DifferentialEvolution():

  def __init__(self):
    self.nameArray = [str(par[0]) for par in Pars]
    self.bounds = {str(p[0]):(p[2], p[3]) for p in Pars}
    self.current = [par[1] for par in Pars]
    self.trial = [par[1] for par in Pars]
    self.variables = dict(zip(self.nameArray, self.current))
    self.pbounds = dict(zip(self.nameArray, self.bounds))

###  Evaluation
  def evaluate(self, variables):
    for name in variables:
      variables[name] = round(variables[name],0)
    trial = dict(zip(self.nameArray, self.trial))
    result  = []
    result = main(variables, trial,)
    pentares_result = self.translate_result(result)
#    print(pentares_result)
#    alpha_hat = self.calc_ridits(pentares_result)
    los = self.calc_los(pentares_result)

    return los * 100

  def translate_result(self, result):
    pentares = []
    for i in range(0,len(result) - 1,2):
      current, next = result[i], result[i + 1]
      pentares.append(str(current)+str(next))
    category = [0,0,0,0,0]
    for score in pentares:
      if score == 'll':
        category[0] = category[0]+1
      if score == 'ld' or score == 'dl':
        category[1] = category[1]+1
      if score == 'dd' or score == 'wl' or score == 'lw':
        category[2] = category[2]+1
      if score == 'wd' or score == 'dw':
        category[3] = category[3]+1
      if score == 'ww':
        category[4] = category[4]+1
    return category


  def pentanomial(self, result):
    pentares = []
    for i in range(0,5):
      pentares.append(result.count(i))
      print(pentares)
    return pentares

  def calc_ridits(self, pentares):
    sump=sum(pentares)
    pentares1=np.array(pentares)/sump
    pentares2=np.array(pentares[::-1])/sump
    marginal=(pentares1+pentares2)/2
    ridits =[]
    ridits.append(0.5*marginal[0])
    for i in range(1,5):
      ridits.append(sum(marginal[:i])+0.5*marginal[i])
    ridits1=ridits*pentares1
    ridits2=ridits*pentares2
    A1bar=sum(ridits1)
    A2bar=sum(ridits2)
    alpha_hat=A1bar-A2bar+0.5
    return alpha_hat

  def calc_los(self, pentares):
    sumi, sumi2 = 0, 0
    for i,score in enumerate([0,0.5,1,1.5,2]):
      N = sum(pentares)
      sumi += pentares[i] * score / N
      sumi2 += pentares[i] * score * score / N
    sigma = math.sqrt(sumi2 - sumi * sumi)
    try:
      t0 = (sumi - 1) / sigma
    except ZeroDivisionError:
      t0 = (sumi - 1)
    los = norm.cdf(t0)
    return los 

if __name__ == '__main__':
  de = DifferentialEvolution()
  print(de.bounds)
  variables = de.variables

  def black_box_function(**variables):
    for name in variables:
      variables[name] = int(variables[name])
    f = de.evaluate(variables)
    return f

  optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=de.bounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#    random_state=0,
  )

  optimizer.maximize(
    n_iter = 100,
    acq='ei',
    alpha = 0.05,
#    n_restarts_optimizer=4,
#    kappa=2,
    xi = 1e-4,
  )
  print(optimizer.max)
