import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

def get_poly_data_matrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1, degree+1):
        X = np.column_stack((X, x**i))
#    print(X)
    return X



def get_weights(x,y,degree):
    X = get_poly_data_matrix(x,degree)
    XX  = X.transpose().dot(X)
    weight = np.linalg.solve(XX, X.transpose().dot(y))
#    print(weight)
    return weight



def pol_regression(x_train, y_train, degree):
    #Code
    parameters = 0
#    print(coeffs)
#    print(coeffs[0])
    
    w1 = get_weights(x_train, y_train, degree)
    Xtest1 = get_poly_data_matrix(x_train, degree)
    ytest1 = Xtest1.dot(w1)
    
    x_test, y_test = sort_data(x_train, ytest1)
    plt.plot(x_test, y_test)
    
    return parameters




#Get Data
def get_data():
    data = pd.read_csv('data_pol_regression.csv')
#    print(data)
    data = data.sample(frac=1)
#    print(data)
    return data

def sort_data(x_in, y_in):
##Sort Data
        df = pd.DataFrame([x_in, y_in])
        df.sort_values(0, axis=1, ascending = True, inplace = True)
        x = []
        y = []
        for point in df:
            x = np.append(x, df[point][0])
            y = np.append(y, df[point][1])
        return x, y

def get_training_data(data, split):
    x_input = data['x'].as_matrix()
    y_input = data['y'].as_matrix() 
#    print(len(x_input)*split)
    test_size = round((len(x_input)*split))
#    print("Training Data:")
    x_train = []
    y_train = []
    for i in range(test_size):
#        print("X:{0} : Y:{1}".format(x_input[i],y_input[i]))
        x_train = np.append(x_train, x_input[i])
        y_train = np.append(y_train, y_input[i])
#    print("Test Data:")
    x_test = []
    y_test = []
    if split != 1:    
        
        for i in range(test_size,len(data)):
    #        print("X:{0} : Y:{1}".format(x_input[i],y_input[i]))
            x_test = np.append(x_test, x_input[i])
            y_test = np.append(y_test, y_input[i])
    x_test, y_test = sort_data(x_test, y_test)
    
    
    return x_train, y_train, x_test, y_test

####    Main Loop    #####
data = get_data()
x_train, y_train, x_test, y_test = get_training_data(data, 1)
####    Plotting Data    ####
plt.xlim(-5,5)
plt.plot(x_train, y_train, 'bo')

#pol_regression(x_train, )
pol_regression(x_train, y_train, 1)
pol_regression(x_train, y_train, 2)
pol_regression(x_train, y_train, 3)
pol_regression(x_train, y_train, 5)
pol_regression(x_train, y_train, 10)
plt.legend(["Data",1,2,3,5,10])
#
#w1 = get_weights(x_train, y_train, i)
#Xtest1 = get_poly_data_matrix(x_train, i)
#ytest1 = Xtest1.dot(w1)
#
#x_test, y_test = sort_data(x_train, ytest1)
#plt.plot(x_test, y_test, 'r')