import numpy as np
from numpy.lib import diff
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def read_data():
    """
    Reads the data of the file and return the result as a float
    """
    valores = read_csv("./src/ex2data1.csv", header=None).to_numpy()
    return valores.astype(float)

def data_exam_graph(X, Y):
    """
        Draw the first graph of the exercise. The results of both exams
        showing the admitted and Not admitted exams
    """
    # Results of admitted
    pos = np.where (Y == 1)
    # Result of not admitted
    neg = np.where (Y == 0)

    fig = plt.figure()
    plt.ylabel('Exam Score 2', c = 'k', size='15')
    plt.xlabel('Exam Score 1', c = 'k', size='15')
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k', label="Admitted")
    plt.scatter(X[neg, 0], X[neg, 1], c='#c6ce00', label="No Admitted")
    plt.legend(loc='lower left')

def sigmoide_fun(Z):
    G = 1 / (1 + (np.exp(-Z)))

    return G

def new_Theta(m, X, Y, Theta):
    """
        Calculate the new value of Theta with matrix
    """
    Z = np.matmul(X, Theta)
    S = sigmoide_fun(Z)
    Diff = S - Y

    X_t = np.transpose(X)
    NewTheta = (1 / m) * np.matmul(X_t, Diff)
    return NewTheta

def fun_J(m, X, Y, Theta):
    """
        Calculates the J function of the cost
        of the Logistic Regresion    
    """
    S = sigmoide_fun(np.dot(X, Theta))
    Sum1 = np.dot(Y, np.log(S))

    # This add is to dodge the log(0)
    Diff = (1 - S) + 0.00001
    Sum2 = np.dot((1 - Y), np.log(Diff))
    Sum = Sum1 + Sum2
    
    return (-1 / m) * Sum

def lineal_fun_graph(X, Y, Theta):
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                           np.linspace(x2_min, x2_max))

    h = sigmoide_fun(np.c_[
        np.ones((xx1.ravel().shape[0], 1)),
        xx1.ravel(),
        xx2.ravel()
    ].dot(Theta))

    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

def gradient():
    valores = read_data()
    # Add all the rows and the col(len - 1)
    X = valores[:, :-1]
    # The -1 value add the col(len - 1)
    Y = valores[:, -1]
    Y = 1 - Y
    # Row X
    m = np.shape(X)[0]
    XX = np.hstack([np.ones([m, 1]), X])
    # Row Y
    n = np.shape(XX)[1]
    
    Theta = np.zeros(n)

    Theta = new_Theta(m, XX, Y, Theta)
    J = fun_J(m, XX, Y, Theta)
    
    # Graph section
    data_exam_graph(X, Y)
    lineal_fun_graph(X, Y, Theta)
    return 1

# main
gradient()
plt.show()