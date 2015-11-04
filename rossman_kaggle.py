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

def clean_stores(path):
    # load data about indvidual stores
    stores = pd.read_csv(path)
    # rename cols
    stores.columns = [w.replace('Competition', '').replace('Since', '')
                      for w in stores.columns]
    # Create logged distance
    stores['logged_dist'] = np.log(stores.Distance)
    # create continuous date variable
    stores['open_since'] = stores.OpenYear * 100 + stores.OpenMonth
    stores['promo_since'] = stores.Promo2Year * 100 + stores.Promo2Week
    return stores


def clean_date(df):
    df['day'] = to_dt(df.Date).dt.day
    df['month'] = df.Date.apply(lambda x: pd.to_datetime(x).month)
    df['year'] = df.Date.apply(lambda x: pd.to_datetime(x).year)
    df['quarter'] = ((df.month - 1)//3 + 1)
    df['prev_day_closed'] = df.Open.shift(1) == 0
    df['next_day_closed'] = df.Open.shift(-1) == 0
    return df


def clean_rossman(PATH):
    df = loadappend_data(PATH)
    df = clean_date(df)
    # merge stores to main
    stores = clean_stores(RAW + 'store.csv')
    df = df.merge(stores, on='Store')
    # merge on locations compliments of the forums
    df = df.merge(pd.read_csv(RAW + 'store_states.csv'), on='Store')
    # encode strings
    enc = LabelEncoder()
    cols_to_enc = ['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday',
                   'State']
    for col in cols_to_enc:
        df[col] = enc.fit_transform(df[col])
    # tag validation
    df = tag_validation(df)
    # Create median, mean, std
    df = create_categorical_medians(df, ['Store', 'DayOfWeek', 'Promo'])
    # fill missings
    df.Open.fillna(1, inplace=True)
    df.fillna(-1, inplace=True)
    return df


def create_categorical_medians(df, columns):
    grpd = df[(df.is_test == 0) & (df.is_val == 0)].groupby(columns)['Sales']
    cat_moments = grpd.aggregate([np.median, np.mean, np.std]).reset_index()
    df = df.merge(cat_moments, on=columns)
    return df

def tag_validation(df, val_pct=.011):
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
all_df.Sales[all_df.Sales < 0] = 0

# Create weights
all_df['weights'] = 1 - 1/(to_dt(all_df.Date).astype('int')/10**18)**3

# Create list of features
non_feat = ['Id', 'is_test', 'is_val', 'Sales', 'Date', 'Customers', 'Open',
            'weights']
Xfeats = create_feat_list(all_df, non_feat)

# Separate samples
trn = all_df[(all_df.is_val == 0) & (all_df.is_test == 0) & (all_df.Open == 1)]
val = all_df[(all_df.is_val == 1) & (all_df.is_test == 0) & (all_df.Sales > 0)]
test = all_df[(all_df.is_test == 1)]

# Run random forest
num_feats = int(len(Xfeats)*.2)
trees = 200
frst = Forest(n_estimators=trees, max_depth=35, max_features=4, n_jobs=-1, bootstrap=True)
frst.fit(trn[Xfeats], np.log(trn.Sales+1), np.array(trn.weights))

# Evaluate features and model
feat_importances(frst, Xfeats)
score = rmspe(np.exp(frst.predict(val[Xfeats]))-1, val.Sales.values)
print "Forest w %s features & %s trees: %s" % (num_feats, trees, score)
test['preds_frst'] = np.exp(frst.predict(test[Xfeats])) - 1

# Run model
param = {'max_depth': 10,  'silent': 1, 'subsample': .65,
         'colsample_bytree': .6}
# Gradient boosting
xgb_trn = xgb.DMatrix(np.array(trn[Xfeats]),
                      label=trn.Sales.map(lambda x: np.log(x+1)),
                      weight=np.array(trn.weights))

xgb_val = xgb.DMatrix(np.array(val[Xfeats]),
                      label=val.Sales.map(lambda x: np.log(x+1)))
xgb_test = xgb.DMatrix(np.array(test[Xfeats]))
for eta in [.011, ]: #.01
    t = time.time()
    param['eta'] = eta
    watch = [(xgb_val, 'eval'), (xgb_trn, 'train')]
    xboost = xgb.train(param.items(), xgb_trn, 3000, evals=watch) #3000

    val = write_xgb_preds(val, xgb_val, xboost, '_xgb')
    test = write_xgb_preds(test, xgb_test, xboost, '_xgb')

    # Score excludes zero sales days
    score = rmspe(val.preds_xgb.values, val.Sales.values)
    median = rmspe(val['median'], val.Sales.values)
    print "For eta %s score is: %s compared to median: %s" % (eta, score, median)
    print "Loop took %s minutes" % ((time.time() - t)/60)

# Create a submission
test_preds = .6*test.preds_xgb + .4*test.preds_frst
subm = pd.DataFrame(zip(test.Id, test_preds),
                    columns=['Id', 'Sales'])
subm.Id = subm.Id.astype('int')
subm.to_csv(RAW + "../Submission/011 validation sample.csv", index=False)


