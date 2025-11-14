# ===============================================
# 🎯 Mục tiêu: Huấn luyện XGBoost, đánh giá bằng ma trận nhầm lẫn
#              và tính thêm các chỉ số y học: Specificity, PPV, NPV
# ===============================================

# 🧰 Import thư viện
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
import pre  # module tiền xử lý

# ===============================================
# 1️⃣ Tiền xử lý dữ liệu
# ===============================================
X_train, X_test, y_train, y_test = pd.read_csv("C:/NCKH/PIMA_RUN/diab.csv")

# ===============================================
# 2️⃣ Cân bằng dữ liệu bằng kỹ thuật TomekLinks
# ===============================================
undersample = TomekLinks()
X_train, y_train = undersample.fit_resample(X_train, y_train)

# ===============================================
# 3️⃣ Huấn luyện mô hình XGBoost
# ===============================================
xgb = XGBClassifier( random_state = 42, eval_metric = 'logloss')
#Chia dữ liệu để kiểm định chéo: 
#  - Chia tập dữ liệu thành K phần bằng nhau
#  - Lặp lại K lần, mỗi lần 1 fold để test còn lại là tập train
#  - Lấy trung bình kết quả để đánh giá độ ổn định của mô hình
# n_split: chia K lần
# shuffle: true nếu muốn trộn data ban đầu

kfold = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
predictions = model_selection.cross_val_predict(xgb, X_train, y_train, cv=kfold)
xgb.fit(X_train, y_train)
Confu_Matrix = confusion_matrix(y_train, predictions)


# ===============================================
# 4️⃣ Ma trận nhầm lẫn và chỉ số thống kê y học
# ===============================================
# n_classes = 2  
# y_true = y_test
# y_pred = predictions

# tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
# specificity = tn / (tn + fp)
# ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
# npv = tn / (tn + fn) if (tn + fn) > 0 else 0

# print("Specificity:", specificity)
# print("PPV (Precision):", ppv)
# print("NPV:", npv)



#  Các chỉ số cơ bản
# acc = accuracy_score(y_test, y_pred)
# pre = precision_score(y_test, y_pred)       # PPV
# rec = recall_score(y_test, y_pred)          # Sensitivity
# f1 = f1_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)

#  Các chỉ số y học
#specificity = TN / (TN + FP)
#ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
#npv = TN / (TN + FN) if (TN + FN) > 0 else 0

# ===============================================
# 5️⃣ In kết quả chi tiết
# ===============================================
# print(f"Confusion Matrix:\n{cm}\n")
# print(f"Accuracy     : {acc:.4f}")
# print(f"Precision(PPV): {pre:.4f}")
# print(f"Recall (Sens): {rec:.4f}")
# print(f"F1-score     : {f1:.4f}")
#print(f"Specificity  : {specificity:.4f}")
#print(f"NPV          : {npv:.4f}")

# ===============================================
# ✅ Ghi chú:
# - TP: Dự đoán đúng lớp 1
# - TN: Dự đoán đúng lớp 0
# - Specificity = TN / (TN + FP)
# - Sensitivity (Recall) = TP / (TP + FN)
# - PPV (Precision) = TP / (TP + FP)
# - NPV = TN / (TN + FN)
# ===============================================
