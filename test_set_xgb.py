import os
import numpy as np
import matplotlib.pyplot as plt


# !pip install scikit-plot
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
from scipy.stats import ks_2samp
import scikitplot as skplt

from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, auc


def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    # if y_pred_scores is not None:
    #     skplt.metrics.plot_ks_statistic(y, y_pred_scores)
    #     plt.savefig("ks_plot.png")
    #     plt.show()
    #     y_pred_scores = y_pred_scores[:, 1]
    #     auroc = roc_auc_score(y, y_pred_scores)
    #     aupr = average_precision_score(y, y_pred_scores)
    #     performance_metrics = performance_metrics + (auroc, aupr)
    return performance_metrics


def print_metrics_summary(accuracy, recall, precision, f1):
    print()
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
    # if auroc is not None:
    #     print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    # if aupr is not None:
    #     print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))


Y = np.load('dataY.npz')
Y= Y.f.arr_0

X = np.load('dataX.npz')
X = X.f.arr_0

import gc
gc.collect()

X = X.reshape(X.shape[0], 116*44)

import pickle

model = pickle.load(open("xgb-tuning.pkl", "rb"))

import xgboost as xgb
import sklearn.datasets
import sklearn.metrics
import sklearn.feature_selection
import sklearn.feature_extraction
import sklearn.model_selection
import gc

#https://gist.github.com/ylogx/53fef94cc61d6a3e9b3eb900482f41e0
import time, gc

def Average(lst):
    return sum(lst) / len(lst)

batch_size = 50000
iterations = 1
# model = None
count = 0
acc_avg = []
y_class_all = []
inference_avg = []

for i in range(iterations):
    gc.collect()
    count += 1
    for start in range(0, len(X), batch_size):
      x_te = X[start:start+batch_size]
      y_te = Y[start:start+batch_size]
      start_t = time.time()
      y_pr = model.predict(xgb.DMatrix(x_te))
      end = time.time()
      print(f"    Runtime of the program is {end - start_t}")
      total_time = end - start_t
      timesample = total_time/y_pr.shape[0]
      aux_inference = timesample*1000000
      print(f"    us/sample is {aux_inference}")
      inference_avg.append(aux_inference)
      y_pr_class = (y_pr > 0.5).astype("int64")
      acc_aux = sklearn.metrics.accuracy_score(y_te, y_pr_class)
      acc_avg.append(acc_aux)
      y_class_all.append(y_pr_class)
      print('    ACCURACY itr@{}: {}'.format(int(start/batch_size), acc_aux))
      del x_te
      del y_te
    print('\n\n    ACCURACY AVG: {}'.format(Average(acc_avg)))
    print('\n\n    Inference Time AVG (us/sample): {}'.format(Average(inference_avg)))


print('\n\n    ACCURACY AVG: {}'.format(Average(acc_avg)))



y_all_class_np = np.array([])
for npa in y_class_all:
  y_all_class_np = np.append(y_all_class_np, npa)


y_all_class_np.shape



accuracy, recall, precision, f1 = compute_performance_metrics(Y, y_all_class_np, None)


import seaborn as sns
print(f'Results for : Recall of {recall}; accuracy of {accuracy}; precision of {precision}; f1 of {f1};')

sns.set(rc={'figure.figsize':(8,8)})
subplot = skplt.metrics.plot_confusion_matrix(Y, y_all_class_np, normalize=True)
subplot.set_ylim(-0.5, 1.5)
plt.savefig("conf_matrix_TESTE.png")
plt.show()


del X
del Y
gc.collect()
