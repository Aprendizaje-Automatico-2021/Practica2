import numpy as np
from numpy.lib import diff
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import scipy


def read_data():
    """
    Reads the data of the file and return the result as a float
    """
    valores = read_csv("./src/ex2data2.csv", header=None).to_numpy()
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

    plt.ylabel('Microchip 2', c = 'k', size='15')
    plt.xlabel('Microchip 1', c = 'k', size='15')
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k', label="y = 1")
    plt.scatter(X[neg, 0], X[neg, 1], c='#c6ce00', label="y = 0")
    plt.legend(loc='lower left')

def sigmoide_fun(Z):
    G = 1 / (1 + (np.exp(-Z)))

    return G

def fun_J(Theta, m, X, Y, lamb):
    """
        Calculates the J function of the cost
        of the Logistic Regresion    
    """
    S = sigmoide_fun(np.dot(X, Theta))
    Sum1 = np.dot(Y, np.log(S))

    # This add is to dodge the log(0)
    Diff = (1 - S) + 0.00001
    Sum2 = np.dot((1 - Y), np.log(Diff))
    # First part
    Sum = Sum1 + Sum2
    Sum = (-1 / m) * Sum
    # Lambda part
    Sum3 = np.sum(np.power(Theta, 2))
    Sum += (lamb / (2 * m)) * Sum3

    return Sum 

def new_theta(Theta, m, X, Y, lamb):
    """
        Calculate the new value of Theta with matrix
    """
    Z = np.matmul(X, Theta)
    S = sigmoide_fun(Z)
    Diff = S - Y

    X_t = np.transpose(X)
    NewTheta = (1 / m) * np.matmul(X_t, Diff) + (lamb/m) * Theta
    NewTheta[0] -= (lamb/m) * Theta[0]

    return NewTheta

def lineal_fun_graph(X, Y, Theta, poly):

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                           np.linspace(x2_min, x2_max))

    poly = poly.fit_transform(np.c_[xx1.ravel(),
                        xx2.ravel()])
    h = sigmoide_fun(np.dot(poly, Theta))

    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    return 0

def solution(lamb = 1):
    valores = read_data()
    
    # Add all the rows and the col(len - 1)
    X = valores[:, :-1]
    print("Shape X: ", np.shape(X))
    poly = PolynomialFeatures(6)
    XX = poly.fit_transform(X)

    # The -1 value add the col(len - 1)
    Y = valores[:, -1]

    print("Shape XX: ", np.shape(XX))
    m = np.shape(XX)[0]
    n = np.shape(XX)[1]
    Theta = np.zeros(n)

    # lambda
    J = fun_J(Theta, m, XX, Y, lamb)
    print("J: ", J)
    T = new_theta(Theta, m, XX, Y, lamb)

    theta_op = scipy.optimize.fmin_tnc(fun_J, Theta, new_theta, args=(m, XX, Y, lamb))

    # Graph section
    plt.title(r'$\lambda$ = ' + str(lamb))
    data_exam_graph(X, Y)
    lineal_fun_graph(X, Y, theta_op[0], poly)

    return 0

# main
plt.figure()
solution(-0.00005)
plt.show()