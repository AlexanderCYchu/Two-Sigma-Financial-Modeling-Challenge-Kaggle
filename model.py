import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.precision", 12)

'''
 calculate the r instead of the normal r2_score
'''
def score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    print 'r2', r2, '\n'
    print 'abs(r2)**(1/2.0)', abs(r2)**(1/2.0), '\n'
    return np.sign(r2) * abs(r2)**(1/2.0)

# it is needed to read the .h5 file
# into a pandas dataframe in order to process it.
with pd.HDFStore("train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")

# It is also needed to split the data
# into training and testing data set by the timestamp 0 to 905 and 906 to 1812
train = df.loc[df['timestamp'] < 906]

test = df.loc[df['timestamp'] >= 906]

'''
For testing data set it is also needed to
split the target and the feature for validation
'''
test_features = test.drop('y', axis=1)

test_target = test[['id', 'y']].copy()

'''
Since the stock data are oftentimes have some fluctuation in some time interval,
 I first need to calculate the standard deviation with a moving time windows
 which is using 10 timestamp.
'''
excl = ['id', 'sample', 'y', 'timestamp']
cols = [c for c in train.columns if c not in excl]

roll_std = train.groupby('timestamp').y.mean().rolling(window=10).std().fillna(0)

# I only use the data with standard deviation less than 0.009 to train our model.
train_idx = train.timestamp.isin(roll_std[roll_std < 0.009].index)

y_train = train['y'][train_idx]

'''
After that, convert the pandas dataframe,both features and target,
to xgboost own data structure. DMatrix is a internal data structure that
used by XGBoost which is optimized for both memory efficiency and training speed.
'''
xgmat_train = xgb.DMatrix(train.loc[train_idx, cols],
                          label=y_train,
                          feature_names=cols)

params_xgb = {'objective'        : 'reg:linear',
              'tree_method'      : 'hist',
              'grow_policy'      : 'depthwise',
              'eta'              : 0.05,
              'subsample'        : 0.6,
              'max_depth'        : 12,
              'min_child_weight' : y_train.size*0.0001,
              'colsample_bytree' : 1,
              'base_score'       : y_train.mean(),
              'silent'           : True,
}

num_boost_round = 16

# train
bst_lst = []
for i in range(8):
    params_xgb['seed'] = 623913 + 239467 * i
    bst_lst.append(xgb.train(params_xgb,
                             xgmat_train,
                             num_boost_round=num_boost_round,
                             verbose_eval=False).__copy__())

# test
pr_lst = []

xgmat_test = xgb.DMatrix(test_features[cols])

for bst in bst_lst:
    pr_lst.append(bst.predict(xgmat_test))

pred = test_target.copy()

pred['y'] = np.array(pr_lst).mean(0)

# Validation score
print score(test_target['y'], pred['y'])

# plot two stock id to see what the prediction looks like
stock_id = 112
print(stock_id)
temp = pred[pred.id == stock_id]
temp['test_target'] = test_target[test_target['id'] == stock_id]['y']
temp[['y', 'test_target']].iloc[:100,:].plot(marker='.')

stock_id = 2047
print(stock_id)
temp = pred[pred.id == stock_id]
temp['test_target'] = test_target[test_target['id'] == stock_id]['y']
temp[['y', 'test_target']].iloc[:100,:].plot(marker='.')
