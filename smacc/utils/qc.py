#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 2 2024

@author: shruti
"""

import os
import importlib_resources
import pandas as pd
from glob import glob
from xgboost import XGBClassifier

import warnings
warnings.simplefilter("ignore")


def collate(inp):
    CSVs = sorted(glob(inp+"/*csv"))
    dfs = []
    for csv in CSVs:
        sid = os.path.basename(csv).split(".csv")[0]
        sdf = pd.read_csv(csv, index_col='Measures')
        sdf = sdf.transpose()
        sdf["subjectID"] = [sid]
        dfs.append(sdf)
    df = pd.concat(dfs)
        
    return df


def auto_qc(inp, out, qc_flag):
    # combine the metrics for each subject
    df = collate(inp)
    
    if qc_flag is True:
        print("Auto QC for segmentations-->")
        # Read the metrics dataframe
        X = df.drop(['subjectID'], axis=1)
        
        # Load the model 
        filename = importlib_resources.files('smacc') / 'model/model_qc.sav'
        # filename = os.path.join(model_path, 'model_qc.sav')
        xgb_clf = XGBClassifier()
        xgb_clf.load_model(filename)
        
        # Predict the labels
        y_pred = xgb_clf.predict(X)
        
        # Save the dataframe with the QC labels
        df["QC_label"] = y_pred
        df.to_csv(os.path.join(out, "smaCC_metrics_QCed.csv"), index=False)
        
    else:
        df.to_csv(os.path.join(out, "smaCC_metrics.csv"), index=False)

