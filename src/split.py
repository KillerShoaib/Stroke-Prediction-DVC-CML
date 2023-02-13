import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import yaml
import sys

params = yaml.safe_load(open('params.yaml'))['split']

n_splits = params['n_splits']
train_path = params['train_path']
test_path = params['test_path']
csv = params['csv_path']

if not (os.path.exists(train_path) and os.path.exists(test_path)):
    os.makedirs(train_path)
    os.makedirs(test_path)

df = pd.read_csv(csv)
skf = StratifiedKFold(n_splits=n_splits)

y = df[['stroke']]
X = df.drop(columns=['stroke'])


for train, test in skf.split(X, y):

    train_csv = df.loc[train]

    test_csv = df.loc[test]

    # running only once
    break

train_path = os.path.join(train_path, "train_split.csv")
test_path = os.path.join(test_path, "eval.csv")

train_csv.to_csv(train_path, index=False)
test_csv.to_csv(test_path, index=False)

sys.stderr.write(f"Train File stored in :{train_path}\n")
sys.stderr.write(f"Test File stored in :{test_path}\n")
