__author__ = 'loaner'


################################ Imports ###################################
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


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
    train = pd.read_csv(PATH + train)
    test = pd.read_csv(PATH + test)
    # create train test indicator
    train['is_test'] = 0
    test['is_test'] = 1
    all_data = train.append(test, ignore_index=True)
    return all_data



def clean_rossman(PATH):
    df = loadappend_data(PATH)
    # encode date and strs
    df['py_date'] = df.Date.apply(lambda x: pd.to_datetime(x).toordinal())
    df['state_holiday'] = LabelEncoder().fit_transform(df.StateHoliday)
    # load data about indvidual stores
    stores = pd.read_csv(RAW + 'store.csv')
    # encode strings
    enc = LabelEncoder()
    cols_to_encode = ['StoreType', 'Assortment', 'PromoInterval']
    for col in cols_to_encode:
        stores[col] = enc.fit_transform(stores[col])
    # rename cols
    new_names = [w.replace('Competition', '').replace('Since', '')
                 for w in stores.columns]
    stores.columns = new_names
    # create continuous date variable
    stores['open_since'] = stores.OpenYear * 100 + stores.OpenMonth
    stores['promo_since'] = stores.Promo2Year * 100 + stores.Promo2Week
    df = df.merge(stores, on='Store')
    # tag validtion
    df = tag_validation(df)
    # fill missings
    df.fillna(-1, inplace=True)
    return df

def tag_validation(df, val_pct=.2):
    df.sort('Date', inplace=True)
    # determine val obs
    trn_obs = len(df.ix[df.is_test == 0, 1])
    val_obs = int(trn_obs * (1 - val_pct))
    # create var
    df['is_val'] = 0
    df['is_val'][val_obs:trn_obs] = 1
    return df


def rmsle(actual, predicted):
    """
    #### __author__ = 'benhamner'
    Computes the root mean squared log error.
    This function computes the root mean squared log error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The root mean squared log error between actual and predicted
    """
    sle_val = (np.power(np.log(np.array(actual)+1) -
               np.log(np.array(predicted)+1), 2))
    msle_val = np.mean(sle_val)
    return np.sqrt(msle_val)


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

def feat_importances(frst, feats):
    outputs = pd.DataFrame({'feats': feats,
                            'weight': frst.feature_importances_})
    outputs = outputs.sort(columns='weight', ascending=False)
    return outputs


# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return rmspe


def write_xgb_preds(df, xgb_data, mod, pred_nm, is_test=0):
    """
    This writes predictions from an XGBOOST model into the data
    Parameters
    --------------
    df: pandas dataframe to predict into
    xgb_data: XGB dataframe (built from same data as df,
             with features used by mod)
    mod: XGB model used for predictions
    pred_nm: prediction naming convention
    scale: (float) this is the power to raise predictions to pred^scale
    Output
    --------------
    data frame with predictions
    """
    # Create name for predictions column
    nm = 'preds'+str(pred_nm)
    # Predict and rescale (rescales to e^pred - 1)
    df[nm] = mod.predict(xgb_data)
    df[nm] = df[nm].apply(lambda x: math.exp(x)-1)
    return df

def create_grpd_stats(df, var):
    """

    :param df: dataframe
    :param var: name of variable to group by
    :return: df with new features
    """
    grouped_sales = df[(df.is_val == 0) & (df.is_test == 0)].groupby(var).Sales
    # Create medians
    median_grpd = grouped_sales.median().reset_index()
    median_grpd.columns = [var, var + '_median']
    # Create means
    mean_grpd = grouped_sales.mean().reset_index()
    mean_grpd.columns = [var, var + '_mean']
    # Create stds
    std_grpd = grouped_sales.std().reset_index()
    std_grpd.columns = [var, var + '_std']
    # Do merges
    for new_stats in [median_grpd, mean_grpd, std_grpd]:
        df = df.merge(new_stats, on=var, how='left')
    return df


############################## Executions ###################################
# Prep data
all_df = clean_rossman(RAW)

# create sample helpers
is_val = (all_df.is_val == 1) & (all_df.is_test == 0)

is_test = (is_val == 0) & (is_trn == 0)

# Create median features
for feature in ['StoreType', 'Store', 'DayOfWeek']: #
    all_df = create_grpd_stats(all_df, feature)


# Create list of features
non_feat = ['Id', 'is_test', 'is_val', 'Sales', 'Date', 'Customers',
            'StateHoliday']
Xfeats = create_feat_list(all_df, non_feat)

all_df.Sales[all_df.Sales < 0] = 0

# Separate samples
val = all_df[(all_df.is_val == 1) & (all_df.is_test == 0)]
trn = all_df[(all_df.is_val == 0) & (all_df.is_test == 0)]
test = all_df[(all_df.is_test == 1)]

# Test features
frst = RandomForestRegressor(n_estimators=200, n_jobs=4)
frst.fit(trn[Xfeats], trn.Sales.values)
results = pd.DataFrame(data=zip(Xfeats, frst.feature_importances_))
print results.sort(columns=1, ascending=False)

# Run model
param = {'max_depth': 6, 'eta': .08,  'silent': 2, 'subsample': .75,
         'colsample_bytree': .85, 'gamma': .00025}
# Gradient boosting
xgb_trn = xgb.DMatrix(np.array(trn[Xfeats]),
                      label=trn.Sales.map(lambda x: np.log(x+1)))
xgb_val = xgb.DMatrix(np.array(val[Xfeats]))
for eta in [.12,]:
    t = time.time()
    param['eta'] = eta
    xboost = xgb.train(param.items(), xgb_trn, 1500)

    preds = write_xgb_preds(val, xgb_val, xboost, '_xgb')

    # Score excludes zero sales days
    all_df_nozero = all_df[all_df.Sales > 0]
    preds = preds[preds.Sales > 0]

    val = rmspe(preds.preds_xgb.values, val.Sales.values)
    print "For eta %s score is: %s" % (eta, val)
    print "Loop took %s minutes" % ((time.time() - t)/60)
