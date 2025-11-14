#Mô hình XGB base
# %%
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix


# %%
df_diabetes = pd.read_csv("C:/NCKH/PIMA_RUN/diab.csv")
print(df_diabetes.head(3))



# %%
# X là tập thuộc tính, đã bỏ nhãn
X = df_diabetes.drop('Outcome', axis=1)
# y là nhãn cho tập X
y = df_diabetes['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
# train_test_split: chia tập dữ liệu thành tập train và test
#stratify: đảm bảo tỉ lệ của y và X là giống với dữ liệu gốc.
print("🔹 Phân bố nhãn ban đầu:", Counter(y_train))


# %%
#cài đặt và các tham số
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train,y_train)


#%%
y_pred = xgb.predict(X_test)

#  Các chỉ số cơ bản
cm = confusion_matrix(y_test, y_pred)
#TN, FP, FN, TP = cm.ravel()
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)       # PPV
rec = recall_score(y_test, y_pred)          # Sensitivity
f1 = f1_score(y_test, y_pred)

#  Các chỉ số y học
#specificity = TN / (TN + FP)
#ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
#npv = TN / (TN + FN) if (TN + FN) > 0 else 0

# ===============================================
# 5️⃣ In kết quả chi tiết
# ===============================================
print(f"Confusion Matrix:\n{cm}\n")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision(PPV): {pre:.4f}")
print(f"Recall (Sens): {rec:.4f}")
print(f"F1-score     : {f1:.4f}")
#print(f"Specificity  : {specificity:.4f}")
#print(f"NPV          : {npv:.4f}")
