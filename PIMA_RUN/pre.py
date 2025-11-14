# ===============================================
# File: pre.py
# Mục tiêu: Tiền xử lý dữ liệu đầu vào cho hệ thống dự đoán bệnh tiểu đường
# Chức năng:
#   - Đọc file CSV (train/test)
#   - Chuẩn hóa dữ liệu (MinMaxScaler)
#   - Tách train/test
#   - Trả về dữ liệu đã xử lý cho mô hình huấn luyện
# ===============================================

# 🧰 Import thư viện
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ===============================================
# 🔧 Hàm chính: dataset()
# Đầu vào: tên file CSV chứa dữ liệu (ví dụ 'train.csv')
# Đầu ra:
#   X_train, X_test, y_train, y_test: dữ liệu đã chia train/test
#   length: số lượng mẫu
#   imbalanceLabel: thống kê số lượng từng lớp
#   att: danh sách các thuộc tính
# ===============================================
def dataset(file):
    # 1️⃣ Đọc dữ liệu
    data = pd.read_csv(file)

    # 2️⃣ Kiểm tra cột nhãn (cột cuối cùng)
    target = data.columns[-1]

    # 3️⃣ Chuẩn hóa các thuộc tính (trừ cột nhãn)
    scaler = MinMaxScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

    # 4️⃣ Tách dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:, :-1],
        data.iloc[:, -1],
        test_size=0.2,
        shuffle=True,
        random_state=10
    )

    # 5️⃣ Thông tin thống kê
    length = len(data)
    imbalanceLabel = data[target].value_counts()
    att = list(data.columns[:-1])

    # ✅ 6️⃣ Trả về kết quả
    return X_train, X_test, y_train, y_test, length, imbalanceLabel, att


# ===============================================
# 🧪 Test nhanh
# ===============================================
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, length, imbalanceLabel, att = dataset("C:/NCKH/PIMA_RUN/diab.csv")
    print("✅ Dữ liệu đã xử lý thành công!")
    print("Tổng số mẫu:", length)
    print("Phân bố nhãn:\n", imbalanceLabel)
    print("Thuộc tính đầu vào:", att)
