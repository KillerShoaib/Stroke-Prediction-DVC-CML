import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import json
import pickle
import matplotlib.pyplot as plt
import yaml
import os
import logging
logging.basicConfig(level=logging.INFO)


# read params

params = yaml.safe_load(open('params.yaml'))['eval']

model_path = params['model_path']
eval_csv_path = params['eval_csv']
metrics_file = params['score']
png_file = params['png']


# load the model and test csv

model = pickle.load(open(model_path, 'rb'))
test = pd.read_csv(eval_csv_path)
y_true = test['stroke']

# drop id,stroke column

test.drop(columns=['id', 'stroke'], inplace=True)

# predict

y_pred = model.predict_proba(test)

roc_score = roc_auc_score(y_true, y_pred[:, 1])
fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])


# creating the folders seperately

if not (os.path.exists(metrics_file)):
    os.mkdir(metrics_file)

# outs and metrics folders need to be created separately
if not (os.path.exists(png_file)):
    os.mkdir(png_file)

metrics_json = os.path.join(metrics_file, 'metrics.json')
metrics_png = os.path.join(png_file, 'metrics.png')

with open(metrics_json, 'w') as fjson:
    json.dump({'roc': roc_score}, fjson)


plt.plot(fpr, tpr, label=f"AUC = {roc_score}")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig(metrics_png)

logging.info(f"AUC Score for Evaluation Set = {roc_score}")
