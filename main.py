"""
mainpred.py
This file is the main code for running the prediction algorithms for OpenAPS.

The OpenAPS data must be in a subdirectory called "data" with a subdirectory of
the ID number that contains the devicestatus.json file. For example:
        ./data/01234567/devicestatus.json
where . represents the current directory with the code.

The code requires the following files:
        bgdata.py
        bgarray.py
        datamatrix.py
        mlalgorithm.py
        ClarkeErrorGrid.py
        oldpred.py

This code also requires the following libraries:
        pandas
        numpy
        gatspy
        sklearn

MedicineX OpenAPS
2017-7-24
"""

from collections import namedtuple
from bgdata import get_bg_dataframe
from bgdata import get_bg_index
from bgdata import get_new_df_entries_every_5_minutes
from bglomb import get_lomb_data
from datamatrix import make_data_matrix
from oldpred import analyze_old_pred_data
from mlalgorithm import *
from savedata import *
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from sklearn import neural_network
from sklearn import metrics
from itertools import product


"""
MODIFY THE VARIABLES BELOW TO RUN DATA
"""
#Array of the ID to use. Put ID Number as a string (e.g. "00000001")
#["00000001", "00000003", "00000004", "00000007", "00000010", "00000011",
# "00000013", "00000015", "00000016", "00000017", "00000020", "00000021",
# "00000023", "00000024"]
ID_ARRAY = np.array(["00000001"])

#Array of the minutes in the future that the predictions will be made for, AKA the prediction horizon. (e.g. [1,15,30])
#1 refers to the prediction 1 minute after the current time (does not include the current time)
PRED_MINUTES_ARRAY = np.array([30, 45, 60, 90, 120])

#Array of the data minutes that will be tested, AKA data horizon. (e.g. [1,15,30,45,60,75,90,105,120])
#Includes the current minute, so 1 is only the current minute, and 5 includes the current minute and 4 minutes before
DATA_MINUTES_ARRAY = np.array([5])

#Choose whether to run 'eventualBG', 'iob', 'cob', 'acob'. (e.g. ['iob', 'acob']). Leave empty to run none
#['acob', 'cob', 'eventualBG', 'iob']
OLD_PRED_ALGORITHM_ARRAY = np.array(['acob', 'cob', 'eventualBG', 'iob'])

#Array of the algorithms that will be tested
#["Linear Regression", "Ridge Regression", "Lasso Regression", "SVM Linear Regression", "MLP Regression"]
ALGORITHM_ARRAY = np.array(["Linear Regression", "Ridge Regression", "Lasso Regression", "SVM Linear Regression", "MLP Regression"])

#Prints every parameter for the grid search.
PRINT_PARAMTERS = False
"""
END
"""


"""
Modify to save/load the data
"""
SAVE_LOMB_DATA = False
LOAD_LOMB_DATA = False
SAVE_PRED_DATA = False
LOAD_PRED_DATA = False
SAVE_ALOGORITHM = False




"""
Modify the plotting variable below if needed
"""
#List of lomb-scargle plots to print
#Leave empty to print none. Otherwise, use something like ['bg','cob']
#['bg','iob','cob']
PLOT_LOMB_ARRAY = np.array([])

#Boolean to show the prediction plot versus the actual bg
SHOW_PRED_PLOT = False
#Boolean to save the prediction plot
SAVE_PRED_PLOT = True
#Boolean to show the Clarke Error Grid plot
SHOW_CLARKE_PLOT = False
#Boolean to save the Clarke Error Grid plot
SAVE_CLARKE_PLOT = True



"""
Constants below
"""
#ALGORITHM CONTANTS
#Values to be tested for the parameters
RIDGE_PARAMETER_ARRAY = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12])
LASSO_PARAMETER_ARRAY = np.array([0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28])
SVM_LINEAR_PARAMETER_ARRAY = np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56])
SVM_LINEAR_EPSILON_ARRAY = np.array([0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28])
#Values for MLP parameters
MLP_LEARNING_ARRAY = np.array([1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1])
MLP_LAYER_ARRAY = np.array([2, 4, 8, 16, 32, 64, 128, 256])
# MLP_FUNCTION_ARRAY = np.array(['identity', 'logistic', 'tanh', 'relu'])
# MLP_OPTIMIZER_ARRAY = np.array(['lbfgs', 'sgd', 'adam'])

ALGORITHM_LIST = ["Linear Regression", "Ridge Regression", "Lasso Regression", "SVM Linear Regression", "MLP Regression"]

#Returns the linear regression model
def linear_regression_model(parameter_array):
    return linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)

#Returns the ridge regression model
def ridge_regression_model(parameter_array):
    alpha_value = parameter_array[0]
    # ridge_solver = parameter_array[0]
    return linear_model.Ridge(alpha=alpha_value, fit_intercept=True, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)

#Returns the lasso regression model
def lasso_regression_model(parameter_array):
    alpha_value = parameter_array[0] #alpha value index is first index
    return linear_model.Lasso(alpha=alpha_value, fit_intercept=True, normalize=True, precompute=False, copy_X=True,
                                max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')

#Returns the svm linear regression model
def svm_linear_regression(parameter_array):
    c_value = parameter_array[0]
    # epsilon_value = parameter_array[1]
    return svm.SVR(kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=c_value, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

#Returns the mlp regression model
def mlp_regression(parameter_array):
    layer_value = parameter_array[0]
    second_layer_value = parameter_array[1]
    learning_rate = parameter_array[2]
    return neural_network.MLPRegressor(hidden_layer_sizes=(layer_value,second_layer_value), activation='identity', solver='adam', alpha=1,
                                        batch_size='auto', learning_rate='constant', learning_rate_init=learning_rate, power_t=0.5,
                                        max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
                                        momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#Dictionary with the name of the algorithm as the key and the function as the value
ALGORITHM_DICT = {
    "Linear Regression":linear_regression_model,
    "Ridge Regression":ridge_regression_model,
    "Lasso Regression":lasso_regression_model,
    "SVM Linear Regression":svm_linear_regression,
    "MLP Regression":mlp_regression}

ALGORITHM_PARAMETERS = {
    "Linear Regression":np.array([np.array([1])]),
    "Ridge Regression":np.array([RIDGE_PARAMETER_ARRAY]),
    "Lasso Regression":np.array([LASSO_PARAMETER_ARRAY]),
    "SVM Linear Regression":np.array([SVM_LINEAR_PARAMETER_ARRAY]),
    "MLP Regression":np.array([MLP_LAYER_ARRAY,MLP_LAYER_ARRAY,MLP_LEARNING_ARRAY])}

#Dictionary with the name of the algorithm as the key and boolean to apply the StandardScaler transformation as the value
ALGORITHM_TRANSFORM = {
    "Linear Regression":False,
    "Ridge Regression":False,
    "Lasso Regression":False,
    "SVM Linear Regression":True,
    "MLP Regression":True}

#Prediction CONSTANTS
MINIMUM_BG = 40
MAXIMUM_BG = 400

#ID CONSTANTS
ID_LIST = [
    "00000001", "00000003", "00000004", "00000007", "00000010", "00000011",
    "00000013", "00000015", "00000016", "00000017", "00000020", "00000021",
    "00000023", "00000024"]
#They work, but they don't have enough days to make accurate data charts
WORK_BUT_SMALL_ID_LIST = ["00000000", "00000005", "00000008", "00000009", "00000012", "00000018", "00000019", "00000025"]
#THESE DON'T WORK!!!!! (e.g. they have loop algorithm only, no data at all)
BAD_ID_LIST = ["00000002", "00000006", "00000014", "00000022"]

#Start and end training set dates
START_TRAIN_DATE_DICT = {
    "00000000": "2017-02-03", "00000001": "2016-10-09", "00000002": "2017-02-03",
    "00000003": "2017-01-06", "00000004": "2017-01-06", "00000005": "2017-02-09",
    "00000006": "2017-01-10", "00000007": "2017-01-07", "00000008": "2017-01-25",
    "00000009": "2017-02-10", "00000010": "2017-01-06", "00000011": "2016-12-11",
    "00000012": "2017-01-08", "00000013": "2017-05-16", "00000015": "2017-04-02",
    "00000016": "2017-06-22", "00000017": "2017-05-19", "00000018": "2016-07-07",
    "00000019": "2017-05-27", "00000020": "2017-04-01", "00000021": "2017-06-13",
    "00000023": "2016-11-07", "00000024": "2017-04-20", "00000025": "2017-04-20"}

END_TRAIN_DATE_DICT = {
    "00000000": "2017-02-07", "00000001": "2016-10-18", "00000002": "2017-02-07",
    "00000003": "2017-01-15", "00000004": "2017-01-15", "00000005": "2017-02-13",
    "00000006": "2017-01-21", "00000007": "2017-01-16", "00000008": "2017-01-25",
    "00000009": "2017-02-15", "00000010": "2017-01-15", "00000011": "2016-12-20",
    "00000012": "2017-01-24", "00000013": "2017-05-25", "00000015": "2017-04-11",
    "00000016": "2017-07-01", "00000017": "2017-05-28", "00000018": "2016-07-16",
    "00000019": "2017-06-02", "00000020": "2017-04-10", "00000021": "2017-06-22",
    "00000023": "2016-11-16", "00000024": "2017-04-29", "00000025": "2017-04-27"}

#Start and end validation set dates
START_VALID_DATE_DICT = {
    "00000000": "2017-02-08", "00000001": "2016-10-19", "00000002": "2017-02-03",
    "00000003": "2017-01-16", "00000004": "2017-01-16", "00000005": "2017-02-09",
    "00000006": "2017-01-10", "00000007": "2017-01-17", "00000008": "2017-01-25",
    "00000009": "2017-02-10", "00000010": "2017-01-16", "00000011": "2016-12-21",
    "00000012": "2017-01-08", "00000013": "2017-05-26", "00000015": "2017-04-12",
    "00000016": "2017-07-02", "00000017": "2017-05-29", "00000018": "2016-07-17",
    "00000019": "2017-06-03", "00000020": "2017-04-11", "00000021": "2017-06-23",
    "00000023": "2016-11-17", "00000024": "2017-04-30", "00000025": "2017-04-28"}

END_VALID_DATE_DICT = {
    "00000000": "2017-02-09", "00000001": "2016-10-22", "00000002": "2017-02-07",
    "00000003": "2017-01-19", "00000004": "2017-01-19", "00000005": "2017-02-13",
    "00000006": "2017-01-21", "00000007": "2017-01-20", "00000008": "2017-01-25",
    "00000009": "2017-02-15", "00000010": "2017-01-19", "00000011": "2016-12-24",
    "00000012": "2017-01-24", "00000013": "2017-05-29", "00000015": "2017-04-15",
    "00000016": "2017-07-05", "00000017": "2017-06-01", "00000018": "2016-07-20",
    "00000019": "2017-06-04", "00000020": "2017-04-14", "00000021": "2017-06-26",
    "00000023": "2016-11-20", "00000024": "2017-05-03", "00000025": "2017-04-30"}

#Start and end test set dates
START_TEST_DATE_DICT = {
    "00000000": "2017-02-10", "00000001": "2016-10-23", "00000002": "2017-02-08",
    "00000003": "2017-01-20", "00000004": "2017-01-20", "00000005": "2017-02-14",
    "00000006": "2017-01-22", "00000007": "2017-01-21", "00000008": "2017-01-26",
    "00000009": "2017-02-16", "00000010": "2017-01-20", "00000011": "2016-12-25",
    "00000012": "2017-01-25", "00000013": "2017-05-30", "00000015": "2017-04-16",
    "00000016": "2017-07-06", "00000017": "2017-06-02", "00000018": "2016-07-21",
    "00000019": "2017-06-05", "00000020": "2017-04-15", "00000021": "2017-06-27",
    "00000023": "2016-11-21", "00000024": "2017-05-04", "00000025": "2017-04-31"}

END_TEST_DATE_DICT = {
    "00000000": "2017-02-12", "00000001": "2016-10-29", "00000002": "2017-02-09",
    "00000003": "2017-01-26", "00000004": "2017-01-26", "00000005": "2017-02-15",
    "00000006": "2017-01-11", "00000007": "2017-01-27", "00000008": "2017-01-26",
    "00000009": "2017-02-17", "00000010": "2017-01-26", "00000011": "2016-12-31",
    "00000012": "2017-01-26", "00000013": "2017-06-05", "00000015": "2017-04-22",
    "00000016": "2017-07-12", "00000017": "2017-06-08", "00000018": "2016-07-27",
    "00000019": "2017-06-07", "00000020": "2017-04-21", "00000021": "2017-07-03",
    "00000023": "2016-11-27", "00000024": "2017-05-10", "00000025": "2017-05-04"}


#Function to return the training, validaiton, and testing set dates given the string of the ID number
def _get_id_dates(id_str):
    return START_TRAIN_DATE_DICT[id_str], END_TRAIN_DATE_DICT[id_str], START_VALID_DATE_DICT[id_str], END_VALID_DATE_DICT[id_str], START_TEST_DATE_DICT[id_str], END_TEST_DATE_DICT[id_str]


#Main function
def main():
    for id_str in ID_ARRAY:
        print "\nID Number: " + id_str
        start_train_str, end_train_str, start_valid_str, end_valid_str, start_test_str, end_test_str = _get_id_dates(id_str)
        bg_df = get_bg_dataframe(id_str) #imports json file as a pandas dataframe

        #get start and stop indices
        bg_df, start_train_index, end_train_index = get_bg_index(bg_df, start_train_str, end_train_str, "Training", True)
        bg_df, start_valid_index, end_valid_index = get_bg_index(bg_df, start_valid_str, end_valid_str, "Validation", False)
        bg_df, start_test_index, end_test_index = get_bg_index(bg_df, start_test_str, end_test_str, "Testing", False)

        train_bg_df, start_train_index, end_train_index = get_new_df_entries_every_5_minutes(bg_df, start_train_index, end_train_index, "New Training")
        valid_bg_df, start_valid_index, end_valid_index = get_new_df_entries_every_5_minutes(bg_df, start_valid_index, end_valid_index, "New Validation")
        test_bg_df, start_test_index, end_test_index = get_new_df_entries_every_5_minutes(bg_df, start_test_index, end_test_index, "New Testing")

        if LOAD_LOMB_DATA:
            train_lomb_data = load_array(id_str + "_train_lomb_data") #If you have already saved data, then load it
            valid_lomb_data = load_array(id_str + "_valid_lomb_data")
            test_lomb_data = load_array(id_str + "_test_lomb_data")
        else:
            #get the lomb-scargle data
            train_lomb_data = get_lomb_data(train_bg_df, start_train_index, end_train_index, PLOT_LOMB_ARRAY)
            valid_lomb_data = get_lomb_data(valid_bg_df, start_valid_index, end_valid_index, PLOT_LOMB_ARRAY)
            test_lomb_data = get_lomb_data(test_bg_df, start_test_index, end_test_index, PLOT_LOMB_ARRAY)

        if SAVE_LOMB_DATA:
            save_data(train_lomb_data, id_str + "_train_lomb_data") #Save data if you want to
            save_data(valid_lomb_data, id_str + "_valid_lomb_data")
            save_data(test_lomb_data, id_str + "_test_lomb_data")

        for pred_minutes in PRED_MINUTES_ARRAY:
            print "    Prediction Minutes: " + str(pred_minutes)

            #Analyze old pred methods
            if len(OLD_PRED_ALGORITHM_ARRAY) != 0:
                iob_pred, iob_time = analyze_old_pred_data(test_bg_df, OLD_PRED_ALGORITHM_ARRAY, start_test_index, end_test_index, pred_minutes,
                                        SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str)

            for data_minutes in DATA_MINUTES_ARRAY:
                print "        Data Minutes: " + str(data_minutes)

                #make data matrix inputs and the bg outputs
                train_data_matrix, actual_bg_train_array, time_bg_train_array = make_data_matrix(train_bg_df, train_lomb_data, start_train_index, end_train_index, data_minutes, pred_minutes)
                valid_data_matrix, actual_bg_valid_array, time_bg_valid_array = make_data_matrix(valid_bg_df, valid_lomb_data, start_valid_index, end_valid_index, data_minutes, pred_minutes)
                test_data_matrix, actual_bg_test_array, time_bg_test_array = make_data_matrix(test_bg_df, test_lomb_data, start_test_index, end_test_index, data_minutes, pred_minutes)

                #Analyze ml algorithms
                for algorithm_str in ALGORITHM_ARRAY:
                    print "            Algorithm: " + algorithm_str

                    if ALGORITHM_TRANSFORM[algorithm_str]:
                        input_train_data_matrix, input_valid_data_matrix, input_test_data_matrix = preprocess_data(train_data_matrix, valid_data_matrix, test_data_matrix) #Transform data
                    else:
                        input_train_data_matrix, input_valid_data_matrix, input_test_data_matrix = train_data_matrix, valid_data_matrix, test_data_matrix #don't transform data

                    new_parameter_bool = True
                    #Iterate over every possible combination of parameters
                    for parameter_array in product(*ALGORITHM_PARAMETERS[algorithm_str]):
                        reg_model = ALGORITHM_DICT[algorithm_str](parameter_array)
                        reg_model = reg_model.fit(input_train_data_matrix, actual_bg_train_array) #Fit model to training data

                        valid_prediction = reg_model.predict(input_valid_data_matrix) #Predict new data
                        valid_prediction[valid_prediction < MINIMUM_BG] = MINIMUM_BG #Set minimum bg level
                        valid_prediction[valid_prediction > MAXIMUM_BG] = MAXIMUM_BG #Set maximum bg level

                        #The cost function is the root mean squared error between the prediction and the real value.
                        #We want to use the model that has parameters that lead to the smallest cost function.
                        error_value = math.sqrt(metrics.mean_squared_error(actual_bg_valid_array, valid_prediction))

                        #Keep the best model
                        if new_parameter_bool or error_value < best_error_value:
                            new_parameter_bool = False
                            best_error_value = error_value
                            best_reg_model = reg_model
                            best_parameter_array = parameter_array

                        if PRINT_PARAMTERS: print parameter_array, error_value

                    print "                Best Validation RMSE: " + str(best_error_value)
                    print "                Best Validation Parameter Value: " + str(best_parameter_array)
                    print "                Best Validation Parameters: " + str(best_reg_model.get_params())
                    test_prediction = best_reg_model.predict(input_test_data_matrix)
                    test_prediction[test_prediction < MINIMUM_BG] = MINIMUM_BG #Set minimum bg level
                    test_prediction[test_prediction > MAXIMUM_BG] = MAXIMUM_BG #Set maximum bg level

                    analyze_ml_data(actual_bg_test_array, test_prediction, time_bg_test_array, iob_pred, iob_time, SHOW_PRED_PLOT, SAVE_PRED_PLOT, SHOW_CLARKE_PLOT, SAVE_CLARKE_PLOT, id_str, algorithm_str,
                                    "Pred" + str(pred_minutes) + "Data" + str(data_minutes)) #Analyze data

                    if SAVE_PRED_DATA:
                        save_data(actual_bg_test_array, "{}pred{}data{}{}_actual_bg_array".format(id_str, str(pred_minutes), str(data_minutes), algorithm_str))
                        save_data(test_prediction, "{}pred{}data{}{}_pred_bg_array".format(id_str, str(pred_minutes), str(data_minutes), algorithm_str))

                    if SAVE_ALOGORITHM:
                        save_data(best_reg_model, "{}pred{}data{}{}_object".format(id_str, str(pred_minutes), str(data_minutes), algorithm_str))


#Run the main function
if __name__ ==  "__main__":
    main()
