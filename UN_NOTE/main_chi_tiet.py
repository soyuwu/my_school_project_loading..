from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
#import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# 全局设置输出图片大小 1280 x 720 像素
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import KernelPCA
from sklearn import svm
import numpy as np
from sklearn.linear_model import LogisticRegression
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from pandas import read_csv
from sklearn.decomposition import PCA
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import scipy.stats
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

import pre_run
np.random.seed(1)
import pandas as pd
global max_acc
X_train, X_test, y_train, y_test,length,imbalanceLabel,att= pre_run.dataset('train.CSV')# 第五个参数为不平衡

data = pd.read_csv('t_last.csv', encoding='gbk')
y_train = data.iloc[:, -1:]
X_train= data.iloc[:, :-1]


undersample = TomekLinks()
X_train, y_train = undersample.fit_resample(X_train, y_train)

#model=svm.SVC(probability=True)
#model=KNeighborsClassifier()
#model=DecisionTreeClassifier()
#model=MLPClassifier(solver='lbfgs')
model = XGBClassifier()
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=7)
predictions = model_selection.cross_val_predict(model, X_test, y_test, cv=kfold)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
cm = confusion_matrix(y_test, predictions)
#target_names = ['hy', 'ga', 'Ip', 'Ct']
n_classes = 4
specificity_list = []
ppv_list = []
npv_list = []
for t in range(n_classes):
        # 将第i类设为正样本，其他类别设为负样本
    y_true_t = [1 if y == t else 0 for y in y_test]
    y_pred_t = [1 if y == t else 0 for y in predictions]
    conf_matrix_i = confusion_matrix(y_true_t, y_pred_t)
    tn_i, fp_i, fn_i, tp_i = conf_matrix_i.ravel()
    specificity_i = tn_i / (tn_i + fp_i)
    ppv_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0
    npv_i = tn_i / (tn_i + fn_i)
    specificity_list.append(specificity_i)
    ppv_list.append(ppv_i)
    npv_list.append(npv_i)

cm = confusion_matrix(y_test, predictions)




# max_acc = 0
# results = []

# print(cm)
# print(max_acc)

# iris_target_names = ['Ah', 'Aga', 'Ip', 'Ct']
# fig, ax = plt.subplots()
# im = ax.imshow(cm, cmap='Blues')
# ax.set_xticks(np.arange(len(iris_target_names)))
# ax.set_yticks(np.arange(len(iris_target_names)))
# ax.set_xticklabels(iris_target_names, rotation=45, ha='right', va='center')
# ax.set_yticklabels(iris_target_names, rotation=0, ha='right', va='center')
# plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
# for i in range(len(iris_target_names)):
#     for j in range(len(iris_target_names)):
#         text = ax.text(j, i, cm[i, j], ha='center', va='center', color='k')
# ax.set_title('Confusion Matrix')
# plt.tight_layout()
# plt.colorbar(im)
# ax.set_xlim(-0.5, 4 - 0.5)
# ax.set_ylim(4 - 0.5, -0.5)
# plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85, wspace=0.1, hspace=0.1)
# plt.show()






