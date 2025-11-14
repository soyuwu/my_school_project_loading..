# ===============================================
# File: main.py
# Mục tiêu: Huấn luyện mô hình cơ bản (XGBoost, SVM, KNN, MLP, Decision Tree)
#            cho bài toán phân loại bệnh tim (4 lớp).
# ===============================================

# 🧰 Import thư viện
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
import pre_run_note  # module tiền xử lý

# ===============================================
# 1️⃣ Tiền xử lý dữ liệu
# - Đọc dữ liệu train.CSV
# - Mã hóa label, chuẩn hóa đặc trưng
# - Tách dữ liệu train/test
# ===============================================
X_train, X_test, y_train, y_test, length, imbalanceLabel, att = pre_run_note.dataset('train.CSV')

# ===============================================
# 2️⃣ Cân bằng dữ liệu bằng kỹ thuật Oversampling
# - Dùng RandomOverSampler để nhân bản các lớp thiểu số
# ===============================================
smote = RandomOverSampler(random_state=10)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ===============================================
# 3️⃣ Huấn luyện mô hình học máy (XGBoost)
# - Có thể thay bằng các mô hình khác (SVM, MLP, KNN...)
# ===============================================
clf = XGBClassifier(max_depth=4, random_state=10)
clf.fit(X_train, y_train)

# ===============================================
# 4️⃣ Dự đoán và đánh giá mô hình
# - Tính Accuracy, Precision, Recall, F1, AUC cho từng lớp
# ===============================================
clf_pred = clf.predict(X_test)
acc = accuracy_score(y_test, clf_pred)
recall = recall_score(y_test, clf_pred, average='macro')
F1 = f1_score(y_test, clf_pred, average='macro')
pre = precision_score(y_test, clf_pred, average='macro')

# Nếu là bài toán đa lớp, tính AUC trung bình macro
auc_scores = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(clf_pred), average='macro')

# ===============================================
# 5️⃣ In kết quả ra màn hình
# ===============================================
print("AUC Scores for Each Class:", auc_scores)
print("Accuracy:", round(acc, 4))
print("Precision:", round(pre, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(F1, 4))
print("Macro AUC:", round(auc_scores, 4))

# ===============================================
# ✅ Kết luận:
# - File này giúp kiểm tra hiệu quả của mô hình cơ bản.
# - Kết quả dùng để chọn mô hình mạnh nhất trước khi ensemble.
# ===============================================
