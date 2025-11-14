# ===============================================
# File: µ╖╖µ╖åτƒ⌐Θÿ╡.py
# Mục tiêu: Huấn luyện XGBoost, đánh giá bằng ma trận nhầm lẫn
#            và tính thêm chỉ số y học: Specificity, PPV, NPV
# ===============================================

# 🧰 Import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import TomekLinks
from xgboost import XGBClassifier
import pre_run_note  # module tiền xử lý

# ===============================================
# 1️⃣ Tiền xử lý dữ liệu
# - Đọc train.CSV qua module pre
# - Có thể đọc thêm tệp dữ liệu mở rộng (t_last.CSV)
# ===============================================
X_train, X_test, y_train, y_test, length, imbalanceLabel, att = pre.dataset('train.CSV')

# Đọc thêm file t_last.csv nếu có (bổ sung dữ liệu mới)
data = pd.read_csv('t_last.csv', encoding='gbk')
X_train = data.iloc[:, :-1]
y_train = data.iloc[:, -1:]

# ===============================================
# 2️⃣ Cân bằng dữ liệu bằng kỹ thuật TomekLinks
# - Loại bỏ các mẫu gần biên giữa hai lớp để dữ liệu "sạch" hơn
# ===============================================
undersample = TomekLinks()
X_train, y_train = undersample.fit_resample(X_train, y_train)

# ===============================================
# 3️⃣ Huấn luyện mô hình (XGBoost)
# - Dùng cross-validation để đánh giá ổn định
# ===============================================
model = XGBClassifier(random_state=10)
kfold = model_selection.KFold(n_splits=10)
predictions = model_selection.cross_val_predict(model, X_test, y_test, cv=kfold)

# ===============================================
# 4️⃣ Ma trận nhầm lẫn và các chỉ số thống kê y học
# ===============================================
cm = confusion_matrix(y_test, predictions)

# Tính các chỉ số tổng thể
acc = accuracy_score(y_test, predictions)
pre = precision_score(y_test, predictions, average='macro')
rec = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')

# Tính chỉ số y học (Specificity, PPV, NPV)
# Với bài toán nhiều lớp, ta tính trung bình các giá trị này
specificity_list, ppv_list, npv_list = [], [], []
for i in range(len(cm)):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FN + FP)
    specificity_list.append(TN / (TN + FP))
    ppv_list.append(TP / (TP + FP))
    npv_list.append(TN / (TN + FN))

specificity = np.mean(specificity_list)
ppv = np.mean(ppv_list)
npv = np.mean(npv_list)

# ===============================================
# 5️⃣ Trực quan hóa kết quả bằng biểu đồ ma trận nhầm lẫn
# ===============================================
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix (4-class Heart Disease)")
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# ===============================================
# 6️⃣ In kết quả chi tiết
# ===============================================
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"PPV (Positive Predictive Value): {ppv:.4f}")
print(f"NPV (Negative Predictive Value): {npv:.4f}")

# ===============================================
# ✅ Kết luận:
# - Đây là bản mở rộng của main.py với đánh giá y học chi tiết.
# - Có hình ảnh minh họa kết quả (Confusion Matrix).
# - Phù hợp khi cần báo cáo hoặc kiểm chứng mô hình.
# ===============================================
