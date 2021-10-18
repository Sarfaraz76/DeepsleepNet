
# coding: utf-8

# In[ ]:


import argparse
import os
import re

import numpy as np

from sklearn.metrics import confusion_matrix, f1_score

from deepsleep.sleep_stage import W, N1, N2, N3, REM


def print_performance(cm):
    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float) # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float) # sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    print("Sample: {}".format(np.sum(cm)))
    print("W: {}".format(tpfn[W]))
    print("N1: {}".format(tpfn[N1]))
    print("N2: {}".format(tpfn[N2]))
    print("N3: {}".format(tpfn[N3]))
    print("REM: {}".format(tpfn[REM]))
    print("Confusion matrix:")
    print(cm)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("Overall accuracy: {}".format(acc))
    print("Macro-F1 accuracy: {}".format(mf1))


def perf_overall(data_dir):
    # Remove non-output files, and perform ascending sort
    allfiles = os.listdir(data_dir)
    outputfiles = []
    for idx, f in enumerate(allfiles):
        if re.match("^output_.+\d+\.npz", f):
            outputfiles.append(os.path.join(data_dir, f))
    outputfiles.sort()

    y_true = []
    y_pred = []
    for fpath in outputfiles:
        with np.load(fpath,allow_pickle=True) as f:
            print((f["y_true"].shape))
            if len(f["y_true"].shape) == 1:
                if len(f["y_true"]) < 10:
                    f_y_true = np.hstack(f["y_true"])
                    f_y_pred = np.hstack(f["y_pred"])
                else:
                    f_y_true = f["y_true"]
                    f_y_pred = f["y_pred"]
            else:
                f_y_true = f["y_true"].flatten()
                f_y_pred = f["y_pred"].flatten()

            y_true.extend(f_y_true)
            y_pred.extend(f_y_pred)

            print("File: {}".format(fpath))
            cm = confusion_matrix(f_y_true, f_y_pred, labels=[0, 1, 2, 3, 4])
            print_performance(cm)
    print(" ")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")

    total = np.sum(cm, axis=1)

    print("DeepSleepNet (current)")
    print_performance(cm)

