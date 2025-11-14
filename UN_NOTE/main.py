from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score, f1_score, \
    precision_score, roc_auc_score

from imblearn.combine import SMOTETomek
from sklearn.metrics import classification_report
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import KMeansSMOTE,BorderlineSMOTE,ADASYN,RandomOverSampler
import pre
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

np.random.seed(1)
X_train, X_test, y_train, y_test,length,imbalanceLabel,att= pre.dataset('train.CSV')# 第五个参数为不平衡
#clf=svm.SVC(probability=True)
clf = XGBClassifier(max_depth=4)
##clf=LogisticRegression()
#clf=LinearDiscriminantAnalysis()
# model=KNeighborsClassifier()
#clf=DecisionTreeClassifier()
#clf=MLPClassifier()
#clf=GaussianNB()
#clf = GradientBoostingClassifier(n_estimators=50)
#clf = KNeighborsClassifier()

smote=RandomOverSampler(random_state=10)
X_train, y_train=smote.fit_resample(X_train, y_train)
'''
data = pd.read_csv('t_last.csv', encoding='gbk')
y_train = data.iloc[:, -1:]
X_train= data.iloc[:, :-1]


undersample = TomekLinks()
X_train, y_train = undersample.fit_resample(X_train, y_train)
'''
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)

n_classes = 4
auc_scores = []
for i in range(n_classes):
    y_test_binary = (y_test == i)
    clf_pred_binary = (clf_pred == i)
    auc_i = roc_auc_score(y_test_binary, clf_pred_binary)
    auc_scores.append(auc_i)
# 计算宏观平均AUC
macro_auc = sum(auc_scores) / n_classes
print("AUC Scores for Each Class:", auc_scores)



from sklearn import metrics
acc=accuracy_score(y_test, clf_pred)
recall=recall_score(y_test, clf_pred,average='macro')
F1=f1_score(y_test, clf_pred,average='macro')
pre = precision_score(y_test, clf_pred, average='macro')



#print("auc: ",auc)
print("acc: ",acc)
print("pre: ",pre)
print("recall: ",recall)
print("F1: ",F1)
print("Macro AUC:", macro_auc)
