from imblearn.under_sampling import TomekLinks
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from xgboost import XGBClassifier
from sklearn import svm
import pre
import numpy as np
# 加载数据集
X_train, X_test, y_train, y_test,length,imbalanceLabel,att= pre.dataset('train.CSV')# 第五个参数为不平衡
# 划分训练集和测试集

data = pd.read_csv('t_last.csv', encoding='gbk')
y_train = data.iloc[:, -1:]
X_train= data.iloc[:, :-1]


undersample = TomekLinks()
X_train, y_train = undersample.fit_resample(X_train, y_train)
# 定义基础模型

model1 = XGBClassifier(max_depth=6)
model2 = KNeighborsClassifier()
model3 = DecisionTreeClassifier()
model4 = MLPClassifier()
model5 = GradientBoostingClassifier(n_estimators=100)
model6 = svm.SVC(probability=True)
# 定义Bagging集成模型
bc = BaggingClassifier(base_estimator= model1, n_estimators=3, max_samples=1.0, max_features=1.0, random_state=42)
# 使用集成模型进行训练
bc.estimators_ = [model2,model6,model4,model5,model3]
bc.fit(X_train, y_train)

pred = bc.predict(X_test)
# 输出集成之后的准确率
accuracy = accuracy_score(y_test, pred)
print("集成之后的准确率：", accuracy)

n_classes = 4
auc_scores = []
for i in range(n_classes):
    y_test_binary = (y_test == i)
    clf_pred_binary = (pred == i)
    auc_i = roc_auc_score(y_test_binary, clf_pred_binary)
    auc_scores.append(auc_i)
# 计算宏观平均AUC
macro_auc = sum(auc_scores) / n_classes
print("AUC Scores for Each Class:", auc_scores)



from sklearn import metrics
acc=accuracy_score(y_test, pred)
recall=recall_score(y_test, pred,average='macro')
F1=f1_score(y_test, pred,average='macro')
pre = precision_score(y_test, pred, average='macro')
#print("auc: ",auc)
print("acc: ",acc)
print("pre: ",pre)
print("recall: ",recall)
print("F1: ",F1)
print("Macro AUC:", macro_auc)


probabilities = bc.predict_proba(X_test)
cm = confusion_matrix(y_test, pred)
target_names = ['hy', 'ga', 'Ip', 'Ct']
n_classes = 4
specificity_list = []
ppv_list = []
npv_list = []
for t in range(n_classes):
        # 将第i类设为正样本，其他类别设为负样本
    y_true_t = [1 if y == t else 0 for y in y_test]
    y_pred_t = [1 if y == t else 0 for y in pred]
    conf_matrix_i = confusion_matrix(y_true_t, y_pred_t)
    tn_i, fp_i, fn_i, tp_i = conf_matrix_i.ravel()
    specificity_i = tn_i / (tn_i + fp_i)
    ppv_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0
    npv_i = tn_i / (tn_i + fn_i)
    specificity_list.append(specificity_i)
    ppv_list.append(ppv_i)
    npv_list.append(npv_i)

cm = confusion_matrix(y_test, pred)




max_acc = 0
results = []

print(cm)
print(max_acc)

iris_target_names = ['Ah', 'Aga', 'Ip', 'Ct']
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(np.arange(len(iris_target_names)))
ax.set_yticks(np.arange(len(iris_target_names)))
ax.set_xticklabels(iris_target_names, rotation=45, ha='right', va='center')
ax.set_yticklabels(iris_target_names, rotation=0, ha='right', va='center')
plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
for i in range(len(iris_target_names)):
    for j in range(len(iris_target_names)):
        text = ax.text(j, i, cm[i, j], ha='center', va='center', color='k')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.colorbar(im)
ax.set_xlim(-0.5, 4 - 0.5)
ax.set_ylim(4 - 0.5, -0.5)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85, wspace=0.1, hspace=0.1)
plt.show()

