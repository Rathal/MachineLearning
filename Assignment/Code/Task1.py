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



def pol_regression(features_train, y_train, degree):
    #Code
    parameters = 0
#    print(coeffs)
#    print(coeffs[0])
    
    return parameters




#Get Data
def get_data():
    data = pd.read_csv('data_pol_regression.csv')
#    print(data)
    data = data.sample(frac=1)
#    print(data)
    return data




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
        
        ##Sort Data
        df_Test = pd.DataFrame([x_test, y_test])
    #    print(df_Test)
        df_Test.sort_values(0, axis=1, ascending = True, inplace = True)
    #    print(df_Test)
        
        x_test = []
        y_test = []
        for point in df_Test:
    #        print(df_Test[point][0])
    #        x_test.append(df_Test[point][0])
    #        y_test.append(df_Test[point][1])
            x_test = np.append(x_test, df_Test[point][0])
            y_test = np.append(y_test, df_Test[point][1])
    
    return x_train, y_train, x_test, y_test





####    Main Loop    #####
data = get_data()
x_train, y_train, x_test, y_test = get_training_data(data, 1)

#for i in range(5):
#    print(get_weights(x_train,y_train,i))

#get_poly_data_matrix(x_train,1)
##  Printing Data ##
#print("Training Data:")
#for i in range(len(x_train)):
#    print("X: {0} | Y: {1}".format(x_train[i], y_train[i]))
#print("Testing Data:")
#for i in range(len(x_test)):
#    print("X: {0} | Y: {1}".format(x_test[i], y_test[i]))


####    Plotting Data    ####
plt.xlim(-5,5)
plt.plot(x_train, y_train, 'bo')
#plt.plot(x_test, y_test, 'g')


w1 = get_weights(x_train, y_train, 2)
print(w1)
Xtest1 = get_poly_data_matrix(x_train, 2)
print(Xtest1)
ytest1 = Xtest1.dot(w1)
print(ytest1)
plt.plot(x_train, ytest1, 'r')
#print(get_poly_data_matrix(x_train, 1))