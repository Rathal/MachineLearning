import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

def pol_regression(features_train, y_train, degree):
    #Code
    parameters = 0
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
    x_training = []
    y_training = []
    for i in range(test_size):
#        print("X:{0} : Y:{1}".format(x_input[i],y_input[i]))
        training_x.append(x_input[i])
        training_y.append(y_input[i])
#    print("Test Data:")
    
    x_test = []
    y_test = []
    for i in range(test_size,len(data)):
#        print("X:{0} : Y:{1}".format(x_input[i],y_input[i]))
        x_test.append(x_input[i])
        y_test.append(y_input[i])
        


####    Main Loop    #####
data = get_data()
get_training_data(data, 0.7)

    