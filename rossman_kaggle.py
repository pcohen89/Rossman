__author__ = 'loaner'


################################ Imports ###################################
import pandas


################################ Globals ###################################
RAW = "/Users/loaner/Desktop/repos/Data/Raw/"

############################### Functions ###################################
def loadappend_data(PATH, train="train.csv", test="test.csv"):
    """
    :param PATH: path to data
    :param train: name of training file
    :param test: name of test file
    :return: training, validation, test
    """
    # append all data
    non_test = pd.read_csv(PATH + train)
    test = pd.read_csv(PATH + test)
    non_test['is_test'] = 0
    test['is_test'] = 1
    all_data = non_test.append(test, ignore_index=True)
    return all_data







############################## Executions ###################################

