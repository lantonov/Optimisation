import re
import csv
from chess import uci
from chess import Board
from chess import Move
from chess import pgn
import network2

engine = 'stockfish.exe'
depth =  14
multipv = 1
Openings = 'C:\\Rockstar\\neural-networks\\src\\tcec_fens.csv'

def choose_fens():
    fens = []
    with open(Openings, 'r') as f:
        for i in range(0,50):
#            line = f.readline().rstrip().strip('[').strip(']').strip('F').strip('E').strip('N').strip(' ').strip('"')
            line = f.readline().rstrip()
#            f.readline()
#            f.readline()
#            f.readline()
#            f.readline()
            fens.append(line)
    #print(fens)
    #Assuming res is a flat list
    with open("input.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in fens:
            writer.writerow([val])

    return fens

def read_fens(filename='..\\data\\tcec_9_50.epd'):
#    inputs = tuple(open(filename,'r'))
    with open(filename, 'r') as myfile:
        inputs=myfile.read()
#    with open(filename, 'r') as csvfile:
#        fenreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#        for row in fenreader:
#            fens = inputs.append(','.join(row))
#    print(str(inputs))

    return inputs

def trans_fens():
    inputs = read_fens(filename='C:\\Rockstar\\neural-networks\\src\\tcec_fens.csv')

    # Single characters substitutions
    inputs = inputs.replace('\n','@')
#    inputs = inputs.translate(str.maketrans('12345678', 'ilmostuv'))
    inputs = inputs.replace('/','')

    # Empty squares
    inputs = inputs.replace('1','0,')
    inputs = inputs.replace('2','0,0,')
    inputs = inputs.replace('3','0,0,0,')
    inputs = inputs.replace('4','0,0,0,0,')
    inputs = inputs.replace('5','0,0,0,0,0,')
    inputs = inputs.replace('6','0,0,0,0,0,0,')
    inputs = inputs.replace('7','0,0,0,0,0,0,0,')
    inputs = inputs.replace('8','0,0,0,0,0,0,0,0,')

    # Castlings
    inputs = inputs.replace(' KQ ',' 2 ')
    inputs = inputs.replace(' KQkq ',' 0 ')
    inputs = inputs.replace(' kq ',' n ')
    inputs = inputs.replace(' Kkq ',' p ')
    inputs = inputs.replace(' Kq ',' 0 ')
    inputs = inputs.replace(' KQk ',' 1 ')
    inputs = inputs.replace(' KQq ',' 1 ')
    inputs = inputs.replace(' Kk ',' 0 ')
    inputs = inputs.replace(' Qq ',' 0 ')
    inputs = inputs.replace(' Qkq ',' p ')

    # Turn
    inputs = inputs.replace(' w ',' 1 ')
    inputs = inputs.replace(' b ',' p ')

    # Removal of ply counts
    inputs = re.sub(r' [0-9,]+ [0-9,]+@', '@', inputs)
    inputs = inputs.replace('-','0')

    # ep and spaces)
    inputs = re.sub(r' [a-z^-][0,]+@', ' 1@', inputs)

    # Pieces
    inputs = inputs.replace('P','1,')
    inputs = inputs.replace('N','2,')
    inputs = inputs.replace('B','3,')
    inputs = inputs.replace('R','4,')
    inputs = inputs.replace('Q','5,')
    inputs = inputs.replace('K','6,')
    inputs = inputs.replace('p','-1,')
    inputs = inputs.replace('n','-2,')
    inputs = inputs.replace('b','-3,')
    inputs = inputs.replace('r','-4,')
    inputs = inputs.replace('q','-5,')
    inputs = inputs.replace('k','-6,')

    inputs = inputs.replace(' ',',')
    inputs = inputs.replace(',,',',')
    inputs = inputs.split('@')
    del inputs[-1]

#    print(inputs)
    saved = open('tcec.csv', 'w')
    for item in inputs:
        saved.write("%s\n" % item)

def launchSf(pos):
  sf = uci.popen_engine(engine)
  info_handler = uci.InfoHandler()
  sf.info_handlers.append(info_handler)

  sf.setoption({'Clear Hash': True})
  sf.setoption({'Hash': 1})

  sf.uci()
  sf.isready()
  sf.ucinewgame()
  board = Board()
  fens = choose_fens()
  board.set_epd(fens[pos])
  print(board)
  sf.position(board)
  sf.go(depth=25)
  score = info_handler.info["score"][1].cp
  scores.append(score)
  print(scores)
  saved = open('tcec_eval.csv', 'w')
  for item in scores:
     saved.write("%s\n" % item)

if __name__ == '__main__':
#    fen = choose_fens()
#    scores = []
#    for pos in range(10,50):
#        launchSf(pos)
    trans_fens()
