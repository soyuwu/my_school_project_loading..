# ===============================================
# File: pre.py
# Mục tiêu: Tiền xử lý dữ liệu đầu vào cho hệ thống dự đoán bệnh tim.
# Chức năng chính:
#   - Đọc file CSV (train/test)
#   - Mã hóa nhãn (LabelEncoder)
#   - Chuẩn hóa dữ liệu (MinMaxScaler)
#   - Tách train/test
#   - Trả về dữ liệu đã xử lý cho mô hình huấn luyện
# ===============================================

# 🧰 Import thư viện
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# ===============================================
# 🔧 Hàm chính: dataset()
# Đầu vào: tên file CSV chứa dữ liệu (ví dụ 'train.CSV')
# Đầu ra:
#   X_train, X_test, y_train, y_test: dữ liệu đã chia train/test
#   length: số lượng mẫu
#   imbalanceLabel: thống kê số lượng từng lớp
#   att: danh sách các thuộc tính
# ===============================================
def dataset(file):
    # -------------------------------------------
    # 1️⃣ Đọc dữ liệu từ file CSV
    # -------------------------------------------
    data = pd.read_csv(file, encoding='gbk')  # Dữ liệu có thể chứa ký tự tiếng Trung
    
    # -------------------------------------------
    # 2️⃣ Xử lý cột nhãn (label)
    # - Cột cuối cùng là nhãn phân loại
    # - Sử dụng LabelEncoder để biến đổi thành dạng số
    # -------------------------------------------
    encoder = LabelEncoder()
    data.iloc[:, -1] = encoder.fit_transform(data.iloc[:, -1])
    
    # Lưu lại thông tin nhãn sau khi mã hóa (0, 1, 2, 3)
    imbalanceLabel = data.iloc[:, -1].value_counts()
    
    # -------------------------------------------
    # 3️⃣ Chuẩn hóa các thuộc tính (feature scaling)
    # - Giúp mô hình học ổn định hơn
    # -------------------------------------------
    scaler = MinMaxScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
    
    # -------------------------------------------
    # 4️⃣ Chia dữ liệu thành train/test
    # - Tỉ lệ: 80% train / 20% test
    # - shuffle = True giúp ngẫu nhiên dữ liệu
    # -------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:, :-1],
        data.iloc[:, -1],
        test_size=0.2,
        shuffle=True,
        random_state=10
    )
    
    # -------------------------------------------
    # 5️⃣ Lưu thông tin bổ sung
    # -------------------------------------------
    length = len(data)  # số lượng mẫu
    att = list(data.columns[:-1])  # danh sách tên các thuộc tính
    
    # -------------------------------------------
    # ✅ 6️⃣ Trả về kết quả
    # -------------------------------------------
    return X_train, X_test, y_train, y_test, length, imbalanceLabel, att


# ===============================================
# 🧪 Test nhanh (tùy chọn)
# Khi chạy file trực tiếp, sẽ in thông tin dữ liệu
# ===============================================
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, length, imbalanceLabel, att = dataset('train.CSV')
    print("✅ Dữ liệu đã xử lý thành công!")
    print("Tổng số mẫu:", length)
    print("Phân bố nhãn:\n", imbalanceLabel)
    print("Thuộc tính đầu vào:", att)
