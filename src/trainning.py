import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost.callback import EarlyStopping
import yaml
import logging
import os
import pickle
import sys
logging.basicConfig(level=logging.INFO)

# reading the params

params = yaml.safe_load(open('params.yaml'))['train']

cv = params['cv']
model_path = params['model_path']
train_csv = params['train_csv']

# hyperparameters
num_boost_round = params['num_boost_round']
params = dict(
    max_depth=params['max_depth'],
    learning_rate=params['learning_rate'],
    objective=params['objective'],
    verbosity=params['verbosity'],
    eval_metric=params['eval_metric'],
    early_stopping_rounds=params['early_stopping_rounds']
)


if not (os.path.exists(model_path)):
    os.mkdir(model_path)

# reading the train csv

df = pd.read_csv(train_csv)

# stratified and random forest class

skf = StratifiedKFold(n_splits=cv)


# defining the X & y

X = df.copy()
y = df['stroke']

for i, (train, val) in enumerate(skf.split(X, y)):

    X_train = X.loc[train]
    y_train = y.loc[train]

    X_val = X.loc[val]
    y_val = y.loc[val]

    # dropping stroke and id from the training set

    X_train.drop(columns=['id', 'stroke'], inplace=True)
    X_val.drop(columns=['id', 'stroke'], inplace=True)

    # converting to DMatrix
    train_set = xgb.DMatrix(X_train, y_train)
    val_set = xgb.DMatrix(X_val, y_val)
    train_val_set = [(train_set, 'train'), (val_set, 'val')]

    model = xgb.train(
        num_boost_round=num_boost_round,
        params=params,
        dtrain=train_set,
        evals=train_val_set,
        verbose_eval=False,
        callbacks=[EarlyStopping(params['early_stopping_rounds'],
                                 data_name='val', save_best=True)]

    )

    y_pred = model.predict(val_set)

    score = roc_auc_score(y_val, y_pred)

    logging.info(f"AUC Score for {i}th iteration = {score}")


# save the file

model_name = os.path.join(model_path, 'model.pkl')

with open(model_name, "wb") as f:
    pickle.dump(model, f)

sys.stderr.write(f"Train complete.\n")
