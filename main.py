'''
mainpred.py
This file is the main code for running the prediction algorithms for OpenAPS.

The OpenAPS data must be in a subdirectory called "data" with a subdirectory of
the ID number that contains the devicestatus.json file. For example:
        ./data/01234567/devicestatus.json
where . represents the current directory with the code.

The code requires the following files:
        bgdataframe.py
        bgarray.py
        datamatrix.py
        mlalgorithm.py
        ClarkeErrorGrid.py

This code also requires the following libraries:
        pandas
        numpy
        gatspy
        sklearn

Trevor Tsue
2017-7-24
'''

from collections import namedtuple
from bgdataframe import get_bg_data
from bgarray import get_bg_array
from datamatrix import make_data_matrix
from oldpred import get_old_pred
from oldpred import analyze_old_pred_data
from mlalgorithm import *
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from sklearn import neural_network



#DATA CONSTANTS MODIFY THESE TO RUN DATA

#Array of the ID to use. Put ID Number as a string (e.g. "00000001")
ID_ARRAY = np.array(["00000001", "00000003", "00000004", "00000007", "00000010", "00000011"])
#Array of the data minutes that will be tested. (e.g. [1,15,30,45,60,75,90,105,120])
DATA_MINUTES_ARRAY = np.array([120])
#Array of the minutes in the future that the predictions will be made for. (e.g. [1,15,30])
PRED_MINUTES_ARRAY = np.array([30])
#Choose whether to run 'eventualBG', 'iob', 'cob', 'acob'. (e.g. ['iob', 'acob'])
#Leave empty to run none
OLD_PRED_ALGORITHM_ARRAY = np.array([])
#Array of the algorithms that will be tested. (e.g. ["Linear Regression", "Ridge Regression"])
ALGORITHM_ARRAY = np.array(["Lasso Regression"])




#PLOTTING CONSTANTS

#List of lomb-scargle plots to print
#Leave empty to print none. Otherwise, use 'bg','iob','cob'. e.g. ['bg','cob']
PLOT_LOMB_ARRAY = np.array([])

#Boolean to show the prediction plot versus the actual bg
SHOW_PRED_PLOT = False
#Boolean to save the prediction plot
SAVE_PRED_PLOT = False
#Boolean to show the Clarke Error Grid plot
SHOW_CLARKE_PLOT = False
#Boolean to save the Clarke Error Grid plot
SAVE_CLARKE_PLOT = False



#ALGORITHM CONTANTS
#Values to be tested for the parameters
PARAMETER_VALUE_ARRAY = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12])
#Values for MLP parameters
MLP_PARAMETER_VALUE_ARRAY = np.array([1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 10, 100, 1000, 10000])
MLP_LAYER_ARRAY = np.array([2,4,8,16,32,64,128,256,512,1024])

def linear_regression_model(parameter_index_keeper):
    return linear_model.LinearRegression(normalize=True)

def ridge_regression_model(parameter_index_keeper):
    parameter_str = str(parameter_index_keeper)
    alpha_value = PARAMETER_VALUE_ARRAY[int(parameter_str[len(parameter_str) - 1])] #alpha value index is first digit
    return linear_model.Ridge(normalize=True, alpha=alpha_value)

def lasso_regression_model(parameter_index_keeper):
    parameter_str = str(parameter_index_keeper)
    alpha_value = PARAMETER_VALUE_ARRAY[int(parameter_str[len(parameter_str) - 1])] #alpha value index is first digit
    return linear_model.Lasso(normalize=True, alpha=alpha_value)

def svm_linear_regression(parameter_index_keeper):
    parameter_str = str(parameter_index_keeper)
    c_value = PARAMETER_VALUE_ARRAY[int(parameter_str[len(parameter_str) - 1])] #c value index is first digit
    #Ignore epsilon because it make very little difference
    # epsilon_value = PARAMETER_VALUE_ARRAY[int(parameter_str[len(parameter_str) - 2])] #epsilon value index is second digit
    return svm.SVR(kernel='linear', C=c_value)

def mlp_regression(parameter_index_keeper):
    parameter_str = str(parameter_index_keeper)
    alpha_value = MLP_PARAMETER_VALUE_ARRAY[int(parameter_str[len(parameter_str) - 1])] #alpha value index is first digit
    # layer_value = MLP_LAYER_ARRAY[int(parameter_str[len(parameter_str) - 1])]
    return neural_network.MLPRegressor(solver='sgd', alpha=alpha_value, hidden_layer_sizes=(100,10))

#Dictionary with the name of the algorithm as the key and the function as the value
ALGORITHM_DICT = {"Linear Regression":linear_regression_model,
                    "Ridge Regression":ridge_regression_model,
                    "Lasso Regression":lasso_regression_model,
                    "SVM Linear Regression":svm_linear_regression,
                    "MLP Regression":mlp_regression
                    }

ALGORITHM_NUM_PARAMETERS = {"Linear Regression":0,
                        "Ridge Regression":1,
                        "Lasso Regression":1,
                        "SVM Linear Regression":1,
                        "MLP Regression":1
                        }

#Dictionary with the name of the algorithm as the key and boolean to apply the StandardScaler transformation as the value
ALGORITHM_TRANSFORM = {"Linear Regression":False,
                        "Ridge Regression":False,
                        "Lasso Regression":False,
                        "SVM Linear Regression":True,
                        "MLP Regression":True
                        }

#Prediction CONSTANTS
MINIMUM_BG = 40
MAXIMUM_BG = 400

#ID CONSTANTS
ID_LIST = ["00000001", "00000003", "00000004", "00000007", "00000010", "00000011"]
#They work, but they don't have enough days to make accurate data charts
WORK_BUT_SMALL_ID_LIST = ["00000000", "00000005", "00000008", "00000009", "00000012"]
#THESE DON'T WORK!!!!! (e.g. they have loop algorithm only)
BAD_ID_LIST = ["00000002", "00000006"]

#Start and end training set dates
START_TRAIN_DATE_DICT = {
    "00000000": "2017-02-03", "00000001": "2016-10-09", "00000002": "2017-02-03",
    "00000003": "2017-01-06", "00000004": "2017-01-06", "00000005": "2017-02-09",
    "00000006": "2017-01-10", "00000007": "2017-01-03", "00000008": "2017-01-25",
    "00000009": "2017-02-10", "00000010": "2017-01-02", "00000011": "2016-12-07",
    "00000012": "2017-01-08"}

END_TRAIN_DATE_DICT = {
    "00000000": "2017-02-07", "00000001": "2016-10-18", "00000002": "2017-02-07",
    "00000003": "2017-01-15", "00000004": "2017-01-15", "00000005": "2017-02-13",
    "00000006": "2017-01-21", "00000007": "2017-01-16", "00000008": "2017-01-25",
    "00000009": "2017-02-15", "00000010": "2017-01-15", "00000011": "2016-12-20",
    "00000012": "2017-01-24"}

#Start and end validation set dates
START_VALID_DATE_DICT = {
    "00000000": "2017-02-08", "00000001": "2016-10-19", "00000002": "2017-02-03",
    "00000003": "2017-01-16", "00000004": "2017-01-16", "00000005": "2017-02-09",
    "00000006": "2017-01-10", "00000007": "2017-01-17", "00000008": "2017-01-25",
    "00000009": "2017-02-10", "00000010": "2017-01-16", "00000011": "2016-12-21",
    "00000012": "2017-01-08"}

END_VALID_DATE_DICT = {
    "00000000": "2017-02-09", "00000001": "2016-10-22", "00000002": "2017-02-07",
    "00000003": "2017-01-19", "00000004": "2017-01-19", "00000005": "2017-02-13",
    "00000006": "2017-01-21", "00000007": "2017-01-20", "00000008": "2017-01-25",
    "00000009": "2017-02-15", "00000010": "2017-01-19", "00000011": "2016-12-24",
    "00000012": "2017-01-24"}

#Start and end test set dates
START_TEST_DATE_DICT = {
    "00000000": "2017-02-10", "00000001": "2016-10-23", "00000002": "2017-02-08",
    "00000003": "2017-01-20", "00000004": "2017-01-20", "00000005": "2017-02-14",
    "00000006": "2017-01-22", "00000007": "2017-01-21", "00000008": "2017-01-26",
    "00000009": "2017-02-16", "00000010": "2017-01-20", "00000011": "2016-12-25",
    "00000012": "2017-01-25"}

END_TEST_DATE_DICT = {
    "00000000": "2017-02-12", "00000001": "2016-10-29", "00000002": "2017-02-09",
    "00000003": "2017-01-26", "00000004": "2017-01-26", "00000005": "2017-02-15",
    "00000006": "2017-01-11", "00000007": "2017-01-27", "00000008": "2017-01-26",
    "00000009": "2017-02-17", "00000010": "2017-01-26", "00000011": "2016-12-31",
    "00000012": "2017-01-26"}


#Function to return the training, validaiton, and testing set dates given the string of the ID number
def _get_id_dates(id_str):
    return START_TRAIN_DATE_DICT[id_str], END_TRAIN_DATE_DICT[id_str], START_VALID_DATE_DICT[id_str], END_VALID_DATE_DICT[id_str], START_TEST_DATE_DICT[id_str], END_TEST_DATE_DICT[id_str]


#Main function
def main():
    for id_str in ID_ARRAY:
        print "\nID Number: " + id_str
        start_train_str, end_train_str, start_valid_str, end_valid_str, start_test_str, end_test_str = _get_id_dates(id_str)
        bg_df, start_train_index, end_train_index, start_valid_index, end_valid_index, start_test_index, end_test_index = get_bg_data(id_str, start_train_str, end_train_str,
                                                                                                                                        start_valid_str, end_valid_str, start_test_str, end_test_str)
        train_lomb_data, valid_lomb_data, test_lomb_data = get_bg_array(bg_df, start_train_index, end_train_index, start_valid_index, end_valid_index, start_test_index, end_test_index, PLOT_LOMB_ARRAY)

        for pred_minutes in PRED_MINUTES_ARRAY:
            print "    Prediction Minutes: " + str(pred_minutes)

            #Analyze old pred methods
            if len(OLD_PRED_ALGORITHM_ARRAY) != 0:
                if pred_minutes % 5 != 0: raise Exception("The prediction minutes is not a multiple of 5.")
                eventual_pred_data, iob_pred_data, cob_pred_data, acob_pred_data = get_old_pred(bg_df, start_test_index, end_test_index, pred_minutes)
                if 'eventualBG' in OLD_PRED_ALGORITHM_ARRAY:
                    print("        eventualBG")
                    analyze_old_pred_data(eventual_pred_data, SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str, "eventualBG", "Pred" + str(pred_minutes))
                if 'iob' in OLD_PRED_ALGORITHM_ARRAY:
                    print("        iob")
                    analyze_old_pred_data(iob_pred_data, SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str, "IOB", "Pred" + str(pred_minutes))
                if 'cob' in OLD_PRED_ALGORITHM_ARRAY:
                    print("        cob")
                    analyze_old_pred_data(cob_pred_data, SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str, "COB", "Pred" + str(pred_minutes))
                if 'acob' in OLD_PRED_ALGORITHM_ARRAY:
                    print("        acob")
                    analyze_old_pred_data(acob_pred_data, SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str, "aCOB", "Pred" + str(pred_minutes))

            for data_minutes in DATA_MINUTES_ARRAY:
                print "        Data Minutes: " + str(data_minutes)

                train_data_matrix, actual_bg_train_array = make_data_matrix(bg_df, train_lomb_data, start_train_index, end_train_index, data_minutes, pred_minutes)
                valid_data_matrix, actual_bg_valid_array = make_data_matrix(bg_df, valid_lomb_data, start_valid_index, end_valid_index, data_minutes, pred_minutes)
                test_data_matrix, actual_bg_test_array = make_data_matrix(bg_df, test_lomb_data, start_test_index, end_test_index, data_minutes, pred_minutes)

                #Analyze ml algorithms
                for algorithm_str in ALGORITHM_ARRAY:
                    print "            Algorithm: " + algorithm_str

                    if ALGORITHM_TRANSFORM[algorithm_str]:
                        input_train_data_matrix, input_valid_data_matrix, input_test_data_matrix = preprocess_data(train_data_matrix, valid_data_matrix, test_data_matrix)
                    else:
                        input_train_data_matrix, input_valid_data_matrix, input_test_data_matrix = train_data_matrix, valid_data_matrix, test_data_matrix

                    #Since we have 10 parameter values for each parameter, when we validate, we have a power of 10 models to run.
                    #The parameter value represents the index of the PARAMETER_VALUE_ARRAY, so a parameter index of 0 will return 0.01 for the paramter.
                    #A parameter index of 1 will return 0.02, 2 will return 0.04, and so on.
                    #Each digit represents a different parameter value. If there is one parameter, we have 0-9
                    #If there are two parameters, the first parameter is controlled by X0 - X9, and the second is 0X - 9X
                    #With three parameters, the third is controlled by 0XY - 9XY, and this trend is continued
                    for parameter_index_keeper in range(10**ALGORITHM_NUM_PARAMETERS[algorithm_str]):
                        reg_model = ALGORITHM_DICT[algorithm_str](parameter_index_keeper)
                        reg_model = reg_model.fit(input_train_data_matrix, actual_bg_train_array) #Fit model to training data

                        valid_prediction = reg_model.predict(input_valid_data_matrix) #Predict new data
                        valid_prediction[valid_prediction < 40] = 40 #Set minimum bg level
                        valid_prediction[valid_prediction > 400] = 400 #Set maximum bg level

                        #The cost function is the mean squared error between the prediction and the real value.
                        #We want to use the model that has parameters that lead to the smallest cost function.
                        error_value = mean_squared_error(actual_bg_valid_array, valid_prediction)

                        #Keep the best model
                        if parameter_index_keeper == 0 or error_value < best_error_value:
                            best_error_value = error_value
                            best_reg_model = reg_model
                            best_parameter_index_keeper = parameter_index_keeper
                        print parameter_index_keeper, math.sqrt(error_value)

                    print "                Best Validation RMSE: " + str(math.sqrt(best_error_value))
                    print "                Best Validation Parameter Value: " + str(best_parameter_index_keeper)
                    print "                Best Validation Parameters: " + str(best_reg_model.get_params())
                    test_prediction = best_reg_model.predict(input_test_data_matrix)
                    test_prediction[test_prediction < MINIMUM_BG] = MINIMUM_BG #Set minimum bg level
                    test_prediction[test_prediction > MAXIMUM_BG] = MAXIMUM_BG #Set maximum bg level

                    analyze_ml_data(actual_bg_test_array, test_prediction, SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str, algorithm_str,
                                    "Pred" + str(pred_minutes) + "Data" + str(data_minutes))


#Run the main function
if __name__ ==  "__main__":
    main()
