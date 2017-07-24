'''
mlalgorithm.py
This file contins the apply_algorithm and analyze_data functions for the OpenAPS machine learning predictions.


Main Function:
        analyze_data(pred_plot_bool, clarke_plot_bool, actual_bg_test_array, bg_prediction)
Input:
        pred_plot_bool                  Boolean to show the plots of prediction and actual bg values.
        clarke_plot_bool                Boolean to show the Clarke Error Grid plot.
        actual_bg_test_array            The array of actual test bg values.
        bg_prediction                   The array of prediction bg values.

USAGE:
        analyze_data(True, True, actual_bg_test_array, bg_prediction)



Main Function:
        apply_algorithm(ml_algorithm, transform_bool, train_data_matrix, test_data_matrix, actual_bg_train_array)

Input:
        ml_algorithm                    The algorithm function that you want to use on the data.
        transform_bool                  Boolean for preprocessing the data by the StandardScaler transform.
        train_data_matrix               The data matrix of the training data.
        test_data_matrix                The data matrix of the testing data.
        actual_bg_train_array           Array of actual bg values for the training data.

Output:
        bg_prediction                   The array of bg predictions based on the machine learning algorithm.

USAGE:
    bg_prediction = apply_algorithm(algorithm, True, train_data_matrix, test_data_matrix, actual_bg_train_array)

Trevor Tsue
2017
'''


import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
import ClarkeErrorGrid



#This functions analyzes the data from actual_bg_test_array and the bg_prediction array
def analyze_data(pred_plot_bool, clarke_plot_bool, actual_bg_test_array, bg_prediction):
    plot, zone = ClarkeErrorGrid.clarke_error_grid(actual_bg_test_array, bg_prediction)
    print "                Zones are A:{}, B:{}, C:{}, D:{}, E:{}\n".format(zone[0],zone[1],zone[2],zone[3],zone[4])
    if clarke_plot_bool: plot.show()

    countarray = np.linspace(0, len(actual_bg_test_array) - 1, len(actual_bg_test_array))
    plt.clf()
    plt.plot(countarray, bg_prediction, label="BG Prediction")
    plt.plot(countarray, actual_bg_test_array, label="Actual BG")
    plt.title("Regression BG Analysis")
    plt.ylabel("Blood Glucose Level (mg/dl)")
    plt.xlabel("Time (minutes)")
    plt.legend(loc='upper left')

    # CHOOSE IF YOU WANT TO SHOW PLOT OR SAVE PLOT
    if pred_plot_bool: plt.show()


#Applies the machine learning algorithm
def apply_algorithm(ml_algorithm, transform_bool, train_data_matrix, test_data_matrix, actual_bg_train_array):
    if transform_bool:
        #Preprocess data with StandardScaler
        reg_scaler = prep.StandardScaler().fit(train_data_matrix)
        train_data_input = reg_scaler.transform(train_data_matrix)
        test_data_input = reg_scaler.transform(test_data_matrix)
    else:
        #Untransformed
        train_data_input = train_data_matrix
        test_data_input = test_data_matrix

    #Apply the linear support vector regression
    reg_model = ml_algorithm
    reg_model = reg_model.fit(train_data_input, actual_bg_train_array)
    bg_prediction = reg_model.predict(test_data_input)

    return bg_prediction
