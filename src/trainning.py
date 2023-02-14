import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
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

n_estimators = params['n_estimators']
criterion = params['criterion']
max_depth = params['max_depth']
min_samples_split = params['min_samples_split']
min_samples_leaf = params['min_samples_leaf']
bootstrap = True if (params['bootstrap'] == 'True') else False
random_state = params['random_state']


if not (os.path.exists(model_path)):
    os.mkdir(model_path)

# reading the train csv

df = pd.read_csv(train_csv)

# stratified and random forest class

skf = StratifiedKFold(n_splits=cv)

rfc = RandomForestClassifier(
    n_estimators=n_estimators,
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    bootstrap=bootstrap,
    random_state=random_state
)


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

    rfc.fit(X_train, y_train)

    y_pred = rfc.predict_proba(X_val)

    score = roc_auc_score(y_val, y_pred[:, 1])

    logging.info(f"AUC Score for {i}th iteration = {score}")


# save the file

model_name = os.path.join(model_path, 'model.pkl')

with open(model_name, "wb") as f:
    pickle.dump(rfc, f)

sys.stderr.write(f"Train complete.\n")
