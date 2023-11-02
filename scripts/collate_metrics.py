#!/usr/bin/env python
import pandas as pd
from glob import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inp", help=""" Input folder path where each subject's metrics are saved""")
parser.add_argument("--out", help=""" Output folder path """)
args = parser.parse_args()

CSVs = sorted(glob(args.inp+"/*csv"))
dfs = []
for csv in CSVs:
    sid = os.path.basename(csv).split(".csv")[0]
    print(sid)
    sdf = pd.read_csv(csv, index_col='Measures')
    sdf = sdf.transpose()
    sdf.index = [sid]
    dfs.append(sdf)
df = pd.concat(dfs)

outcsv=args.out+"/metrics.csv"
df.to_csv(outcsv, index_label='subjectID')

