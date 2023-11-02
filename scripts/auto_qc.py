import warnings
warnings.simplefilter("ignore")

import pandas as pd
import os
import sys
import numpy as np
import pickle
from xgboost import XGBClassifier
import xgboost as xgb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inp", help=""" Input metrics path """)
parser.add_argument("--model_path", help="""Path to where the QC model is saved """)
parser.add_argument("--out", help=""" QC file """)
args = parser.parse_args()

# Read the metrics dataframe
df = pd.read_csv(args.inp + "/metrics.csv")
X_new = df.drop(['subjectID'], axis=1)

# Load the model 
filename = args.model_path + '/model_qc.sav'
xgb_clf = XGBClassifier()
xgb_clf.load_model(filename)

# Predict the labels
y_pred = xgb_clf.predict(X_new)

df_new = df.copy()
df_new["QC_label"] = y_pred
df_new.to_csv(args.out + "/metrics_Final.csv", index=False)

print("Done with QC!")

