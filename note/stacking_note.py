from imblearn.under_sampling import TomekLinks
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from xgboost import XGBClassifier
import pandas as pd
import pre_run_note
# 加载数据集
X_train, X_test, y_train, y_test,length,imbalanceLabel,att= pre_run_note.dataset()# 第五个参数为不平衡
# 划分训练集和测试集

data = pd.read_csv('t_last.csv', encoding='gbk')
y_train = data.iloc[:, -1:]
X_train= data.iloc[:, :-1]


undersample = TomekLinks()
X_train, y_train = undersample.fit_resample(X_train, y_train)


# 划分Meta集
X_train_base, X_meta, y_train_base, y_meta = train_test_split(X_train, y_train, test_size=0.5, random_state=1)
# 定义基础模型
model1 = XGBClassifier(max_depth=8)
model2 = KNeighborsClassifier()
model3 = DecisionTreeClassifier()
model4 = MLPClassifier()
model5 = GradientBoostingClassifier(n_estimators=100)
model6 = svm.SVC(probability=True)
# 使用第一部分的训练数据训练基础模型
model1.fit(X_train_base, y_train_base)
model2.fit(X_train_base, y_train_base)
model3.fit(X_train_base, y_train_base)
model4.fit(X_train_base, y_train_base)
model5.fit(X_train_base, y_train_base)
model6.fit(X_train_base, y_train_base)
# 使用基础模型对Meta集进行预测
pred_meta1 = model1.predict(X_meta)
pred_meta2 = model2.predict(X_meta)
pred_meta3 = model3.predict(X_meta)
pred_meta4 = model4.predict(X_meta)
pred_meta5 = model5.predict(X_meta)
pred_meta6 = model6.predict(X_meta)
# 构建Meta集特征
X_stack = np.column_stack((pred_meta1, pred_meta2, pred_meta3, pred_meta4, pred_meta5,pred_meta6))
# 定义Meta模型
meta_model = XGBClassifier()
# 使用第一部分的训练数据训练Meta模型
meta_model.fit(X_stack, y_meta)
# 使用基础模型和Meta模型对测试集进行预测
pred_test1 = model1.predict(X_test)
pred_test2 = model2.predict(X_test)
pred_test3 = model3.predict(X_test)
pred_test4 = model4.predict(X_test)
pred_test5 = model5.predict(X_test)
pred_test6 = model6.predict(X_test)
X_test_stack = np.column_stack((pred_test1, pred_test2, pred_test3,pred_test4, pred_test5, pred_test6))
pred_final = meta_model.predict(X_test_stack)
# 输出集成之前的性能指标
pred_base = np.column_stack((pred_test1, pred_test2, pred_test3,pred_test4, pred_test5, pred_test6))
pred_base_final = np.round(np.mean(pred_base, axis=1))
accuracy_base = accuracy_score(y_test, pred_base_final)
print("集成之前的准确率：", accuracy_base)
# 输出集成之后的性能指标
accuracy_ensemble = accuracy_score(y_test, pred_final)

print("集成之后的准确率：", accuracy_ensemble)

n_classes = 4
auc_scores = []
for i in range(n_classes):
    y_test_binary = (y_test == i)
    clf_pred_binary = (pred_final == i)
    auc_i = roc_auc_score(y_test_binary, clf_pred_binary)
    auc_scores.append(auc_i)
# 计算宏观平均AUC
macro_auc = sum(auc_scores) / n_classes
print("AUC Scores for Each Class:", auc_scores)



from sklearn import metrics
acc=accuracy_score(y_test, pred_final)
recall=recall_score(y_test, pred_final,average='macro')
F1=f1_score(y_test, pred_final,average='macro')
pre = precision_score(y_test, pred_final, average='macro')



#print("auc: ",auc)
print("acc: ",acc)
print("pre: ",pre)
print("recall: ",recall)
print("F1: ",F1)
print("Macro AUC:", macro_auc)