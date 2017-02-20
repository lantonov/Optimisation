import re
import numpy as np

def choose_fens():
    evals = np.genfromtxt('tcec_evals.csv',delimiter=',')
    saved = open('evals.csv', 'w')
    for item in evals:
        if float(item) == 0.00: item = '0,0,0,0,1,0,0,0,0,0'
        elif float(item) > 0.00 and float(item) < 0.21: item = '0,0,0,0,0,1,0,0,0,0'
        elif float(item) > 0.20 and float(item) < 0.51: item = '0,0,0,0,0,0,1,0,0,0'
        elif float(item) > 0.50 and float(item) < 1.01: item = '0,0,0,0,0,0,0,1,0,0'
        elif float(item) > 1.00 and float(item) < 3.01: item = '0,0,0,0,0,0,0,0,1,0'
        elif float(item) > 3.00: item = '0,0,0,0,0,0,0,0,0,1'
        elif float(item) < 0.00 and float(item) >= -0.20: item = '0,0,0,1,0,0,0,0,0,0'
        elif float(item) < -0.20 and float(item) >= -0.50: item = '0,0,1,0,0,0,0,0,0,0'
        elif float(item) < -0.50 and float(item) >= -1.50: item = '0,1,0,0,0,0,0,0,0,0'
        elif float(item) < -1.50: item = '1,0,0,0,0,0,0,0,0,0'
        elif re.findall(r'M[0-9]+', str(item)): item = '0,0,0,0,0,0,0,0,0,1'
        elif re.findall(r'-M[0-9]+', str(item)): item = '1,0,0,0,0,0,0,0,0,0'
        saved.write("%s\n" % item)
    return evals

if __name__ == '__main__':
    choose_fens()
