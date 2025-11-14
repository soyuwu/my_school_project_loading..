# %%
from imblearn.under_sampling import TomekLinks
# from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from xgboost import XGBClassifier
from sklearn import svm
import pre_run_note
import numpy as np

# Tải dữ liệu và chia tập huấn luyện - kiểm tra
# Tham số thứ 5 trả về thông tin mất cân bằng
X_train, X_test, y_train, y_test, length, imbalanceLabel, att = pre_run_note.dataset("C:/NCKH/PIMA_RUN/diab.csv")

# Giảm mất cân bằng dữ liệu bằng TomekLinks (undersampling)
undersample = TomekLinks()
X_train, y_train = undersample.fit_resample(X_train, y_train)

# Định nghĩa các mô hình cơ sở (base models)
model1 = XGBClassifier(max_depth=6)
model2 = KNeighborsClassifier()
model3 = DecisionTreeClassifier()
model4 = MLPClassifier()
model5 = GradientBoostingClassifier(n_estimators=100)
model6 = svm.SVC(probability=True)

# Định nghĩa mô hình Bagging tổng hợp
bc = BaggingClassifier(estimator=model1, n_estimators=3, max_samples=1.0, max_features=1.0, random_state=42)

# Gán thủ công các mô hình con vào Bagging (thay vì để Bagging tự chọn)
bc.estimators_ = [model2, model6, model4, model5, model3]

# Huấn luyện mô hình tổng hợp
bc.fit(X_train, y_train)

# Dự đoán tập kiểm tra
pred = bc.predict(X_test)

# In ra độ chính xác sau khi tổng hợp
accuracy = accuracy_score(y_test, pred)
print("Độ chính xác sau khi tổng hợp:", accuracy)

# Tính AUC cho bài toán nhị phân
auc = roc_auc_score(y_test, pred)
print("AUC:", auc)

# Tính và in các chỉ số đánh giá khác
from sklearn import metrics
acc = accuracy_score(y_test, pred)
recall = recall_score(y_test, pred, average='binary')
F1 = f1_score(y_test, pred, average='binary')
pre = precision_score(y_test, pred, average='binary')

print("Accuracy:", acc)
print("Precision:", pre)
print("Recall:", recall)
print("F1-score:", F1)
print("AUC:", auc)

# Tính ma trận nhầm lẫn và các chỉ số thống kê cho từng lớp
probabilities = bc.predict_proba(X_test)
cm = confusion_matrix(y_test, pred)
target_names = ['class 0', 'class 1']
n_classes = 2
specificity_list = []
ppv_list = []
npv_list = []

for t in range(n_classes):
    # Xem lớp t là dương tính, các lớp khác là âm tính
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

# Tính lại ma trận nhầm lẫn
cm = confusion_matrix(y_test, pred)

# Khởi tạo biến để lưu kết quả độ chính xác cao nhất
max_acc = 0
results = []

# In ra ma trận nhầm lẫn và độ chính xác tối đa
print("Confusion Matrix:")
print(cm)
print("Max Accuracy (init):", max_acc)

# (Tùy chọn) Vẽ ma trận nhầm lẫn trực quan bằng matplotlib
# iris_target_names = ['Class 0', 'Class 1']
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
# ax.set_xlim(-0.5, 2 - 0.5)
# ax.set_ylim(2 - 0.5, -0.5)
# plt.show()

# %%