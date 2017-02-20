import numpy as np
import network2

def open_data():
    fens = np.genfromtxt('tcec.csv',delimiter=',')
    evals = np.genfromtxt('evals.csv',delimiter=',')
    x = [np.reshape(x, (67, 1)) for x in fens]
    y = [np.reshape(y, (10, 1)) for y in evals]
    data = list(zip(x,y))
#    print(data)

    return data, x, y

if __name__ == '__main__':
    data, x, y = open_data()
#    net = network2.Network([67, 128, 64, 10], cost=network2.QuadraticCost)
    net = network2.load('net.txt')
    net.SGD(training_data=data, epochs=109, mini_batch_size=100, eta=0.5, lmbda = 0.5,
         evaluation_data=data, 
         monitor_evaluation_accuracy=True)
    net.save(filename='net.txt')
