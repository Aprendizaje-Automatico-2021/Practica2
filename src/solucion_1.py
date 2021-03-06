import numpy as np
from numpy.lib import diff
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy . optimize as opt

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
    G = (1 / (1 + (np.exp(-Z))))

    return G

def new_Theta(Theta, X, Y):
    """
        Calculate the new value of Theta with matrix
    """
    Z = np.matmul(X, Theta)
    S = sigmoide_fun(Z)
    Diff = S - Y

    NewTheta = (1 / len(Y)) * np.matmul(np.transpose(X), Diff)
    return NewTheta

def fun_J(Theta, X, Y):
    """
        Calculates the J function of the cost
        of the Logistic Regresion    
    """
    S = sigmoide_fun(np.matmul(X, Theta))
    Sum1 = np.matmul(Y, np.log(S + 0.00001))

    # This add is to dodge the log(0)
    Diff = (1 - S) + 0.00001
    Sum2 = np.matmul((1 - Y), np.log(Diff))
    
    Sum = Sum1 + Sum2
    
    return (-1 / len(X)) * Sum

def lineal_fun_graph(X, Y, Theta):
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                           np.linspace(x2_min, x2_max))

    h = sigmoide_fun(np.c_[np.ones(( xx1.ravel().shape[0], 1)),
                                     xx1.ravel(),
                                     xx2.ravel()].dot(Theta))

    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

def percentage(Theta, X, Y):
    S = sigmoide_fun(np.matmul(X, Theta))
    i = 0
    success = 0
    for i in range(len(S)):
        if S[i] >= 0.5:
            S[i] = 1
        else:
            S[i] = 0
        if (S[i] == Y[i]):
            success += 1
    
    return (success / len(S)) * 100

def gradient():
    valores = read_data()
    # Add all the rows and the col(len - 1)
    # Matrix: (m, n)
    X = valores[:, :-1]
    XX = np.hstack([np.ones([np.shape(X)[0], 1]), X]) # [100, 3]
    # The -1 value add the col(len - 1)
    #(m,)
    Y = valores[:, -1]
    
    # (m, n + 1)
    # Row Y
    n = np.shape(XX)[1]
    
    Theta = np.zeros(n)

    J = fun_J(Theta, XX, Y)
    Theta = new_Theta(Theta, XX, Y)
    print("Funci??n de coste: ", J)
    print("New Thetas: ", Theta)


    # Graph section
    data_exam_graph(X, Y)
    result = opt.fmin_tnc(func = fun_J, x0 = Theta, fprime = new_Theta, args = (XX, Y))
    theta_opt = result[0]
    print("Precentage of succes: {} %".format(percentage(theta_opt, XX, Y)))
    
    lineal_fun_graph(XX, Y, theta_opt)
    return 1

# main
gradient()
plt.show()