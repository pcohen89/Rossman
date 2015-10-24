__author__ = 'loaner'


################################ Imports ###################################
import pandas as pd
from sklearn.preprocessing import LabelEncoder



################################ Globals ###################################
RAW = "/Users/loaner/Desktop/repos2/Rossman/Data/Raw/"

############################### Functions ###################################
def loadappend_data(PATH, train="train.csv", test="test.csv"):
    """
    :param PATH: path to data
    :param train: name of training file
    :param test: name of test file
    :return: training, validation, test
    """
    non_test = pd.read_csv(PATH + train)
    test = pd.read_csv(PATH + test)
    # create train test indicator
    non_test['is_test'] = 0
    test['is_test'] = 1
    all_data = non_test.append(test, ignore_index=True)
    return all_data

def clean_rossman(PATH):
    df = loadappend_data(RAW)
    # clean stores data
    stores = pd.read_csv(RAW + 'store.csv')
    enc = LabelEncoder()
    stores['StoreType'] = enc.fit_transform(stores.StoreType)
    stores['Assortment'] = enc.fit_transform(stores.Assortment)

    df = df.merge(stores, on='Store')
    return df

clean_rossman(RAW)

def create_feat_list(df, non_features):
    feats = list(df.columns.values)
    for var in non_features:
        feats.remove(var)
    return feats


def subm_correl(subm1, subm2, id, target):
    """
    Measures correlation between to Kaggle submissions
    """
    subm1 = pd.read_csv(SUBM_PATH + subm1)
    subm2 = pd.read_csv(SUBM_PATH + subm2)
    subm2 = subm2.rename(columns={target: 'target2'})
    merged_df = subm1.merge(subm2, on=id)
    return merged_df.corr()

def merge_subms(subm_dict, path, name, target):
    """
    :param subm_dict: Dictionary of dfs to merge, where key is csv name and
    value is weight (values must sum to 1 to keep outcome in original range
    :param path: path to submission folder
    :param name: name of new file
    :param target: outcome variable of submission
    :return:
    """
    subm = pd.read_csv(path+'template.csv')
    for csv, weight in subm_dict.iteritems():
        # Read in a new csv
        score = pd.read_csv(path+csv)
        # rename target to avoid merge issues
        score = score.rename(columns={target: 'target2'})
        # Merge files to be averaged
        subm = subm.merge(score, on='id')
        subm[target] += weight * subm['target2']
        subm = subm.drop('target2', 1)
    subm.to_csv(path+name, index=False)

def check_weight_and_merge(dict, name):
    """
    :param dict: file, weight pairs
    :param name: name of resulting blended file
    :return: blended file saved to server
    """
    total_weight = 0
    for key, val in dict.iteritems():
        total_weight += val
    print "The total weight should be 1.0, it is: %s" % (total_weight)
    merge_subms(dict, SUBM_PATH, name, 'cost')









############################## Executions ###################################

