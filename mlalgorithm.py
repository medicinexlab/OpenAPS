'''
mlalgorithm.py
This file contins the apply_algorithm and analyze_data functions for the OpenAPS machine learning predictions.


Main Function:
        analyze_ml_data(actual_bg_test_array, bg_prediction, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str, algorithm_str, minutes_str):
Input:
        actual_bg_test_array            The array of actual test bg values.
        bg_prediction                   The array of prediction bg values.
        show_pred_plot                  Boolean to show the plots of prediction and actual bg values.
        save_pred_plot                  Boolean to save the plot of prediction and actual bg values
        show_clarke_plot                Boolean to show the Clarke Error Grid plot.
        save_clarke_plot                Boolean to save the Clarke Error Grid plot.
        id_str                          String of the ID number.
        algorithm_str                   String of the algorithm name.
        minutes_str                     String with the data minutes and prediction minutes.

USAGE:
        analyze_ml_data(actual_bg_test_array, bg_prediction, True, False, True, False, "00000001", "Linear Regression", "Pred30Data120")



Main Function:


MedicineX OpenAPS
2017-7-24
'''


import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
from sklearn import metrics
import math
import ClarkeErrorGrid


#This functions analyzes the data from actual_bg_test_array and the bg_prediction array
def analyze_ml_data(actual_bg_test_array, bg_prediction, show_pred_plot, save_pred_plot, show_clarke_plot, save_clarke_plot, id_str, algorithm_str, minutes_str):
    #Root mean squared error
    rms = math.sqrt(metrics.mean_squared_error(actual_bg_test_array, bg_prediction))
    print "                Root Mean Squared Error: " + str(rms)
    print "                Mean Absolute Error: " + str(metrics.mean_absolute_error(actual_bg_test_array, bg_prediction))
    print "                R^2 Coefficient of Determination: " + str(metrics.r2_score(actual_bg_test_array, bg_prediction))

    plot, zone = ClarkeErrorGrid.clarke_error_grid(actual_bg_test_array, bg_prediction, id_str + " " + algorithm_str)
    print "                Zones are A:{}, B:{}, C:{}, D:{}, E:{}\n".format(zone[0],zone[1],zone[2],zone[3],zone[4])
    if save_clarke_plot: plt.savefig(id_str + algorithm_str.replace(" ", "") + minutes_str + "clarke.png")
    if show_clarke_plot: plot.show()

    countarray = np.linspace(0, len(actual_bg_test_array) - 1, len(actual_bg_test_array))
    plt.clf()
    plt.plot(countarray, actual_bg_test_array, label="Actual BG")
    plt.plot(countarray, bg_prediction, label="BG Prediction")
    plt.title(id_str + " " + algorithm_str + " BG Analysis")
    plt.ylabel("Blood Glucose Level (mg/dl)")
    plt.xlabel("Time (minutes)")
    plt.legend(loc='upper left')

    # SHOW/SAVE PLOT DEPENDING ON THE BOOLEAN PARAMETER
    if save_pred_plot: plt.savefig(id_str + algorithm_str.replace(" ","") + minutes_str + "plot.png")
    if show_pred_plot: plt.show()


#Preprocesses the data by the standard scaler relative to the train_data_matrix
def preprocess_data(train_data_matrix, valid_data_matrix, test_data_matrix):
    reg_scaler = prep.StandardScaler().fit(train_data_matrix)
    transform_train_data_matrix = reg_scaler.transform(train_data_matrix)
    transform_valid_data_matrix = reg_scaler.transform(valid_data_matrix)
    transform_test_data_matrix = reg_scaler.transform(test_data_matrix)

    return transform_train_data_matrix, transform_valid_data_matrix, transform_test_data_matrix
