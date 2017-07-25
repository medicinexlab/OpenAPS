'''
mlalgorithm.py
This file contins the apply_algorithm and analyze_data functions for the OpenAPS machine learning predictions.


Main Function:
        analyze_data(actual_bg_test_array, bg_prediction, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_string, algorithm_string, minutes_string):
Input:
        actual_bg_test_array            The array of actual test bg values.
        bg_prediction                   The array of prediction bg values.
        show_pred_plot                  Boolean to show the plots of prediction and actual bg values.
        save_pred_plot                  Boolean to save the plot of prediction and actual bg values
        show_clarke_plot                Boolean to show the Clarke Error Grid plot.
        save_clarke_plot                Boolean to save the Clarke Error Grid plot.
        id_string                       String of the ID number.
        algorithm_string                String of the algorithm name.
        minutes_string                  String with the data minutes and prediction minutes.

USAGE:
        analyze_data(actual_bg_test_array, bg_prediction, True, False, True, False, "00897741", "Linear Regression", "Data120Pred30")



Main Function:
        apply_algorithm(ml_algorithm, transform_bool, train_data_matrix, test_data_matrix, actual_bg_train_array, actual_bg_test_array)

Input:
        ml_algorithm                    The algorithm function that you want to use on the data.
        transform_bool                  Boolean for preprocessing the data by the StandardScaler transform.
        train_data_matrix               The data matrix of the training data.
        test_data_matrix                The data matrix of the testing data.
        actual_bg_train_array           Array of actual bg values for the training data.
        actual_bg_test_array            Array of actual bg values for the testing data.

Output:
        bg_prediction                   The array of bg predictions based on the machine learning algorithm.

USAGE:
    bg_prediction = apply_algorithm(linear_model.LinearRegression(normalize=True), True, train_data_matrix, test_data_matrix, actual_bg_train_array, actual_bg_test_array)

Trevor Tsue
2017-7-24
'''


import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
from sklearn.metrics import mean_squared_error
import math
import ClarkeErrorGrid



#This functions analyzes the data from actual_bg_test_array and the bg_prediction array
def analyze_data(actual_bg_test_array, bg_prediction, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_string, algorithm_string, minutes_string):
    #Root mean squared error
    rms = math.sqrt(mean_squared_error(actual_bg_test_array, bg_prediction))
    print "                Root Mean Squared Error: " + str(rms)

    plot, zone = ClarkeErrorGrid.clarke_error_grid(actual_bg_test_array, bg_prediction, id_string + " " + algorithm_string)
    print "                Zones are A:{}, B:{}, C:{}, D:{}, E:{}\n".format(zone[0],zone[1],zone[2],zone[3],zone[4])
    if save_clarke_plot: plt.savefig(id_string + algorithm_string.replace(" ", "") + minutes_string + "clarke.png")
    if show_clarke_plot: plot.show()

    countarray = np.linspace(0, len(actual_bg_test_array) - 1, len(actual_bg_test_array))
    plt.clf()
    plt.plot(countarray, bg_prediction, label="BG Prediction")
    plt.plot(countarray, actual_bg_test_array, label="Actual BG")
    plt.title(id_string + " " + algorithm_string + " BG Analysis")
    plt.ylabel("Blood Glucose Level (mg/dl)")
    plt.xlabel("Time (minutes)")
    plt.legend(loc='upper left')

    # CHOOSE IF YOU WANT TO SHOW PLOT OR SAVE PLOT
    if save_pred_plot: plt.savefig(id_string + algorithm_string.replace(" ","") + minutes_string + "plot.png")
    if show_pred_plot: plt.show()


#Applies the machine learning algorithm
def apply_algorithm(ml_algorithm, transform_bool, train_data_matrix, test_data_matrix, actual_bg_train_array, actual_bg_test_array):
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
    print "                R^2 Value: " + str(reg_model.score(test_data_matrix, actual_bg_test_array))

    return bg_prediction
