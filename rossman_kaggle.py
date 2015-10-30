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
    df['month'] = df.Date.apply(lambda x: pd.to_datetime(x).month)
    df['year'] = df.Date.apply(lambda x: pd.to_datetime(x).year)
    df['quarter'] = ((df.month - 1)//3 + 1)
    # previous day closed
    df['prev_day_closed'] = df.Open.shift(1) == 0
    df['next_day_closed'] = df.Open.shift(-1) == 0
    # load data about indvidual stores
    stores = pd.read_csv(RAW + 'store.csv')
    # rename cols
    stores.columns = [w.replace('Competition', '').replace('Since', '')
                      for w in stores.columns]
    # Create logged distance
    stores['logged_dist'] = np.log(stores.Distance)
    # create continuous date variable
    stores['open_since'] = stores.OpenYear * 100 + stores.OpenMonth
    stores['promo_since'] = stores.Promo2Year * 100 + stores.Promo2Week
    # merge stores to main
    df = df.merge(stores, on='Store')
    # encode strings
    enc = LabelEncoder()
    cols_to_enc = ['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday']
    for col in cols_to_enc:
        df[col] = enc.fit_transform(df[col])
    # tag validation
    df = tag_validation(df)
    # fill missings
    df.Open.fillna(1, inplace=True)
    df.fillna(-1, inplace=True)
    return df


def tag_validation(df, val_pct=.03):
    df.sort('Date', inplace=True)
    # determine val obs
    trn_obs = len(df.ix[df.is_test == 0, 1])
    val_obs = int(trn_obs * (1 - val_pct))
    # create var
    df['is_val'] = 0
    df['is_val'][val_obs:trn_obs] = 1
    return df


def create_feat_list(df, non_features):
    feats = list(df.columns.values)
    for var in non_features:
        feats.remove(var)
    return feats


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
        subm = subm.merge(score, on='id')
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
    merge_subms(dict, path, name, 'cost')

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

def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe


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

def feat_importances(frst, feats):
    outputs = pd.DataFrame({'feats': feats,
                            'weight': frst.feature_importances_})
    outputs = outputs.sort(columns='weight', ascending=False)
    print outputs

############################## Executions ###################################
# Prep data
all_df = clean_rossman(RAW)

columns = ['Store', 'DayOfWeek', 'Promo']
not_test = (all_df.is_test == 0)
scores = all_df[not_test].groupby(columns)['Sales'].median().reset_index()
scores.columns = columns + ['medians']
all_df = all_df.merge(scores, on=columns)

# Create list of features
non_feat = ['Id', 'is_test', 'is_val', 'Sales', 'Date', 'Customers',
            'StateHoliday', 'medians']
Xfeats = create_feat_list(all_df, non_feat)

all_df.Sales[all_df.Sales < 0] = 0
all_df = all_df[all_df.Open == 1]

# Separate samples
val = all_df[(all_df.is_val == 1) & (all_df.is_test == 0)]
trn = all_df[(all_df.is_val == 0) & (all_df.is_test == 0)]
test = all_df[(all_df.is_test == 1)]

frst = RandomForestRegressor(n_estimators=1000, max_depth=22,
                             max_features=int(len(Xfeats)*.66))
frst.fit(trn[Xfeats], np.log(trn.Sales+1))
feat_importances(frst, Xfeats)
rmspe(np.exp(frst.predict(val[Xfeats]))-1, val.Sales.values)


# Run model
param = {'max_depth': 10, 'eta': .08,  'silent': 2, 'subsample': .75}
# Gradient boosting
xgb_trn = xgb.DMatrix(np.array(trn[Xfeats]),
                      label=trn.Sales.map(lambda x: np.log(x+1)))
xgb_val = xgb.DMatrix(np.array(val[Xfeats]),
                      label=val.Sales.map(lambda x: np.log(x+1)))
xgb_test = xgb.DMatrix(np.array(test[Xfeats]))
for eta in [.04, ]:
    t = time.time()
    param['eta'] = eta
    watch = [(xgb_val, 'eval'), (xgb_trn, 'train')]
    xboost = xgb.train(param.items(), xgb_trn, 750, evals=watch, verbose_eval=True)

    val = write_xgb_preds(val, xgb_val, xboost, '_xgb')
    test = write_xgb_preds(test, xgb_test, xboost, '_xgb')

    # Score excludes zero sales days
    val = val[val.Sales > 0]
    score = rmspe(val.preds_xgb.values, val.Sales.values)
    median = rmspe(val.medians, val.Sales.values)
    print "For eta %s score is: %s compared to median: %s" % (eta, score, median)
    print "Loop took %s minutes" % ((time.time() - t)/60)

# Create a submission
test_preds = .1*test.medians + .9*test.preds_xgb
subm = pd.DataFrame(zip(test.Id, test_preds),
                    columns=['Id', 'Sales'])
subm.Id = subm.Id.astype('int')
subm.to_csv(RAW + "../Submission/xgb w month and year blended with medians.csv", index=False)

subm_correl(RAW + r"../Submission/xgb w month and year.csv",
            RAW + r"../Submission/xgb w month and year blended with medians.csv", r'Id', r'Sales')

