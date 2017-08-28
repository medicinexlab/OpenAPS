"""
mlalgorithm.py
This file contins the apply_algorithm and analyze_data functions for the OpenAPS machine learning predictions.


Main Functions:      analyze_ml_data(actual_bg_array, bg_prediction, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str, algorithm_str, minutes_str)



Main Function:


MedicineX OpenAPS
2017-7-24
"""


import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
from sklearn import metrics
import math
import ClarkeErrorGrid


#This functions analyzes the data from actual_bg_array and the bg_prediction array
def analyze_ml_data(actual_bg_array, bg_prediction, bg_time_array, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str, algorithm_str, minutes_str):
    """
    Function to analyze and plot the machine learning data. It takes in the actual_bg_array and the bg_prediction and compares
    the two with various analysis methods, such as root mean squared error, mean absolute error,
    R^2 coefficient of determination, and clarke error grid analysis.

    Input:      actual_bg_array                 The array of actual bg values
                bg_prediction                   The array of prediction bg values
                bg_time_array                   The array of times corresponding to bg_prediction
                show_pred_plot                  Boolean to show the prediction plot
                save_pred_plot                  Boolean to save the prediction plot
                show_clarke_plot                Boolean to show the clarke error grid
                save_clarke_plot                Boolean to save the clarke error grid
                id_str                          String of the ID
                algorithm_str                   String of the algorithm name
                minutes_str                     String of the number of minutes (both prediction and data minutes)
.
    Output:     None
    Usage:      analyze_ml_data(actual_bg_test_array, test_prediction, True, False, True, False, "00000001", "Linear Regression", "Pred30Data5")
    """

    #Root mean squared error
    rms = math.sqrt(metrics.mean_squared_error(actual_bg_array, bg_prediction))
    print "                Root Mean Squared Error: " + str(rms)
    print "                Mean Absolute Error: " + str(metrics.mean_absolute_error(actual_bg_array, bg_prediction))
    print "                R^2 Coefficient of Determination: " + str(metrics.r2_score(actual_bg_array, bg_prediction))

    plot, zone = ClarkeErrorGrid.clarke_error_grid(actual_bg_array, bg_prediction, id_str + " " + algorithm_str)
    print "                Percent A:{}".format(float(zone[0]) / (zone[0] + zone[1] + zone[2] + zone[3] + zone[4]))
    print "                Percent C, D, E:{}".format(float(zone[2] + zone[3] + zone[4])/ (zone[0] + zone[1] + zone[2] + zone[3] + zone[4]))
    print "                Zones are A:{}, B:{}, C:{}, D:{}, E:{}\n".format(zone[0],zone[1],zone[2],zone[3],zone[4])
    if save_clarke_plot: plt.savefig(id_str + algorithm_str.replace(" ", "") + minutes_str + "clarke.png")
    if show_clarke_plot: plot.show()

    plt.clf()
    plt.plot(bg_time_array, actual_bg_array, label="Actual BG", color='black', linestyle='-')
    plt.plot(bg_time_array, bg_prediction, label="BG Prediction", color='black', linestyle=':')
    plt.title(id_str + " " + algorithm_str + " BG Analysis")
    plt.ylabel("Blood Glucose Level (mg/dl)")
    plt.xlabel("Time (minutes)")
    plt.legend(loc='upper left')

    # SHOW/SAVE PLOT DEPENDING ON THE BOOLEAN PARAMETER
    if save_pred_plot: plt.savefig(id_str + algorithm_str.replace(" ","") + minutes_str + "plot.png")
    if show_pred_plot: plt.show()


#Preprocesses the data by the standard scaler relative to the train_data_matrix
def preprocess_data(train_data_matrix, valid_data_matrix, test_data_matrix):
    """
    Function to preprocess the data with the standard scaler from sci-kit learn.
    It takes in the training, validation, and testing matrices and returns the
    standardized versions of them.

    Input:      train_data_matrix               The data matrix with the training set data
                valid_data_matrix               The data matrix with the validation set data
                test_data_matrix                The data matrix with the testing set data
.
    Output:     transform_train_data_matrix     The data matrix with the standardized training set data
                transform_valid_data_matrix     The data matrix with the standardized validation set data
                transform_test_data_matrix      The data matrix with the standardized testing set data
    Usage:      analyze_ml_data(actual_bg_test_array, test_prediction, True, False, True, False, "00000001", "Linear Regression", "Pred30Data5")
    """

    reg_scaler = prep.StandardScaler().fit(train_data_matrix)
    transform_train_data_matrix = reg_scaler.transform(train_data_matrix)
    transform_valid_data_matrix = reg_scaler.transform(valid_data_matrix)
    transform_test_data_matrix = reg_scaler.transform(test_data_matrix)

    return transform_train_data_matrix, transform_valid_data_matrix, transform_test_data_matrix
