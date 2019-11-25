import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

def getPolynomialDataMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1,degree+1):
        X = np.column_stack((X, x ** i))
    return X

def getWeightsForPolynomialFit(x, y, degree):
    X = getPolynomialDataMatrix(x, degree)
    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(y))
    return w

def pol_regression(features_train, y_train, degree): 
    #code
    plt.figure()
    plt.plot(features_train, y_train, 'bo')
    
    w = getWeightsForPolynomialFit(features_train, y_train, degree)
    Xtest = getPolynomialDataMatrix(x_test, degree)
    ytest = Xtest.dot(w)
    plt.plot(x_test, ytest, 'r')
    #parameters = degree
    #return parameters

data = pd.read_csv('data_pol_regression.csv')
x_input = data['x'].as_matrix()
y_input = data['y'].as_matrix()



##Create Variables X
#X = np.split(x_input, 4) ##Splits 20 into 4x5. Use last set of 5 as training
#x_train = np.array #init new array
#x_train = np.append (x_train, [X[0], X[1], X[2]]) #Adds 3/4 of input data to train
#x_train = np.matrix(x_train)
#x_test = np.array(X[3]) #sets 1/4 of input data to test
#
##Create Variables Y
#Y = np.split(y_input, 4)
#y_train = np.array
#y_train = np.append(y_train, [Y[0], Y[1], Y[2]])
#y_train = np.matrix(y_train)
#y_test = np.array(Y[3])


plt.clf()
#plt.plot(x_input, y_input, 'bo')
plt.plot(x_train, y_train, 'bo')
#plt.plot(x_test, y_test, 'g')
#plt.savefig('trainingdata.png')
plt.show()

#pol_regression(x_train, y_train, 0)

#Cross Validation 20points, 4 sets of 5. 4 training 1 test

#plot_data(data)
