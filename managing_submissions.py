__author__ = 'loaner'


################################ Imports ###################################
import pandas as pd
from pandas import to_datetime as to_dt
import numpy as np
import xgboost as xgb
import time
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor as Forest


################################ Globals ###################################
SUBM = "/Users/loaner/Desktop/repos2/Rossman/Data/Submission/"

################################ Functions ###################################

def subm_correl(subm1, subm2, id, target):
    """
    Measures correlation between to Kaggle submissions
    """
    subm1 = pd.read_csv(subm1)
    subm2 = pd.read_csv(subm2)
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
        subm = subm.merge(score, on='Id')
        subm[target] += weight * subm['target2']
        subm = subm.drop('target2', 1)
    subm.to_csv(path+name, index=False)

def check_weight_and_merge(dict, name, path):
    """
    :param dict: file, weight pairs
    :param name: name of resulting blended file
    :return: blended file saved to server
    """
    total_weight = 0
    for key, val in dict.iteritems():
        total_weight += val
    print "The total weight should be 1.0, it is: %s" % (total_weight)
    merge_subms(dict, path, name, 'Sales')

############################### Execute ##################################

subm_correl(SUBM + r"011 validation sample.csv",
            SUBM + r"added year lagged.csv",
            r'Id', r'Sales')

merge_weights = {r"new vars xgb frst and medians fixed frst.csv": .5,
                 "improving feature sets and forest take 2.csv": .5}

check_weight_and_merge(merge_weights, 'first blended submission', SUBM)