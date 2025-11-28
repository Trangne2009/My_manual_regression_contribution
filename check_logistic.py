import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from statsmodels.api import Logit, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE

# 1. Đọc và chuẩn bị dữ liệu
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# 2. Kiểm tra mất cân bằng dữ liệu
def check_class_balance(data, target_column):
    class_balance = data[target_column].value_counts(normalize=True).max()
    print("\nPhân phối biến mục tiêu:")
    print(data[target_column].value_counts(normalize=True))
    return class_balance

# 3. Chuyển đổi biến mục tiêu nếu cần
def encode_target_variable(data, target_column):
    if data[target_column].dtype == 'object':
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
    return data

def check_vif(X_train):
    # Thay thế giá trị vô hạn và loại bỏ NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Chỉ chọn các cột số
    X_train = X_train.select_dtypes(include=[np.number])
    
    # Thêm cột hằng số cho mô hình
    X_train_const = add_constant(X_train)

    # Kiểm tra cột hằng số
    if X_train_const.isnull().any().any():
        print("Cảnh báo: Dữ liệu chứa giá trị NaN sau khi thêm hằng số.")
    
    try:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_train_const.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X_train_const.values, i)
            for i in range(X_train_const.shape[1])
        ]
        # Loại bỏ cột hằng số
        vif_data = vif_data[vif_data["Feature"] != "const"]
        
        print("\nVIF của các biến:")
        print(vif_data)
        return vif_data

    except Exception as e:
        print("Lỗi khi tính toán VIF:", str(e))
        raise

# 5. Phân tích và loại bỏ ngoại lệ (Cook's Distance và Leverage Scores)
def detect_outliers(X_train_const, y_train):
    logit_model = Logit(y_train, X_train_const).fit()
    influence = logit_model.get_influence()

    cooks_d = influence.cooks_distance[0]
    leverage_scores = influence.hat_matrix_diag

    return cooks_d, leverage_scores

def remove_outliers(X_train, y_train, cooks_d, leverage_scores):
    # Loại bỏ các điểm có Cook's Distance > 4/n
    threshold_cooks = 0.2
    outlier_indices_cooks = np.where(cooks_d > threshold_cooks)[0]
    print(f"Số điểm có Cook's Distance vượt ngưỡng: {len(outlier_indices_cooks)}")

    # Loại bỏ các điểm có leverage score > 0.2
    threshold_leverage = 0.3
    outlier_indices_leverage = np.where(leverage_scores > threshold_leverage)[0]
    print(f"Số điểm có Leverage Score vượt ngưỡng: {len(outlier_indices_leverage)}")
    
    # Kết hợp các chỉ số ngoại lệ (Cook's Distance và Leverage Scores)
    outlier_indices = np.union1d(outlier_indices_cooks, outlier_indices_leverage)
    print(f"Số điểm ngoại lệ tổng cộng: {len(outlier_indices)}")

    # Loại bỏ các điểm ngoại lệ từ X_train và y_train
    X_train_filtered = X_train.drop(index=X_train.index[outlier_indices])
    y_train_filtered = y_train.drop(index=y_train.index[outlier_indices])

    return X_train_filtered, y_train_filtered, outlier_indices_cooks, outlier_indices_leverage

# 7. Huấn luyện mô hình Logistic Regression
def train_logistic_regression(X_train, y_train):
    logreg_model = LogisticRegression(max_iter=5000)
    logreg_model.fit(X_train, y_train)
    return logreg_model

# 8. Đánh giá độ chính xác của mô hình
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Huấn luyện và kiểm tra độ chính xác
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    # In kết quả
    print(f"Độ chính xác trên tập huấn luyện: {train_accuracy:.4f}")
    print(f"Độ chính xác trên tập kiểm tra: {test_accuracy:.4f}")
    
    # Báo cáo chi tiết
    y_pred = model.predict(X_test)
    print("Báo cáo phân loại:")
    print(classification_report(y_test, y_pred))
    
# Hàm tính toán Cook's Distance và Leverage Scores
def calculate_influence(X_train, y_train, model):
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Tính toán ma trận dự đoán
    y_pred = model.predict_proba(X_scaled)[:, 1]

    # Tính residuals (phần dư)
    residuals = y_train - y_pred

    # Tính hat matrix (ma trận h)
    h = np.dot(X_scaled, np.linalg.pinv(np.dot(X_scaled.T, X_scaled))) @ X_scaled.T
    leverage_scores = np.diagonal(h)

    # Tính Cook's Distance
    cooks_d = (residuals ** 2) / (model.intercept_ + model.coef_.sum()) * (leverage_scores / (1 - leverage_scores))

    return cooks_d, leverage_scores
    
# Hàm xử lý điểm ngoại lệ (Cook's Distance và Leverage Scores)
def process_outliers(X_train, y_train, cooks_d, leverage_scores):
    # Loại bỏ các điểm có Cook's Distance > 4/n
    threshold_cooks = 4 / len(cooks_d)
    outlier_indices = np.where(cooks_d > threshold_cooks)[0] # Ngưỡng Leverage

    # Loại bỏ các điểm có leverage score > 0.2
    threshold_leverage = 0.2
    outlier_leverage_indices = np.where(leverage_scores > threshold_leverage)[0]
    
    # Loại bỏ các điểm ngoại lệ từ X_train và y_train
    X_train_filtered = X_train.drop(index=X_train.index[outlier_indices])
    y_train_filtered = y_train.drop(index=y_train.index[outlier_indices])

    X_train_filtered = X_train_filtered.drop(index=X_train_filtered.index[outlier_leverage_indices])
    y_train_filtered = y_train_filtered.drop(index=y_train_filtered.index[outlier_leverage_indices])

    return X_train_filtered, y_train_filtered, outlier_indices, outlier_leverage_indices

def remove_multicollinearity(X_train, threshold=5):
    X_train_const = add_constant(X_train)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_train_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_train_const.values, i)
        for i in range(X_train_const.shape[1])
    ]

    # Loại bỏ cột 'const' khỏi bảng VIF
    vif_data = vif_data[vif_data["Feature"] != "const"]

    # Loại bỏ các biến có VIF > threshold
    while vif_data["VIF"].max() > threshold:
        max_vif = vif_data.loc[vif_data["VIF"].idxmax()]
        print(f"\nLoại bỏ biến {max_vif['Feature']} với VIF = {max_vif['VIF']}")
        X_train = X_train.drop(columns=[max_vif["Feature"]])

        # Tính lại VIF sau khi loại bỏ biến
        X_train_const = add_constant(X_train)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_train_const.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X_train_const.values, i)
            for i in range(X_train_const.shape[1])
        ]
        vif_data = vif_data[vif_data["Feature"] != "const"]

    return X_train, vif_data

# Áp dụng PCA vào dữ liệu
def apply_pca(X, n_components=0.95):  # Giữ lại 95% biến thể
    try:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        print(f"Số lượng thành phần chính sau PCA: {X_pca.shape[1]}")
        return pd.DataFrame(X_pca)
    except Exception as e:
        print(f"Lỗi khi áp dụng PCA: {e}")
        raise

# Hàm chính để thực thi toàn bộ quy trình
def main(file_path):
    # 1. Load data
    data = load_data(file_path)
    
    target_column = 'loan_status'  # Đổi tên nếu cần
    
    # 2. Kiểm tra mất cân bằng dữ liệu
    class_balance = check_class_balance(data, target_column)
    
    # 3. Chuyển đổi biến mục tiêu
    data = encode_target_variable(data, target_column)
    
    # 4. Tách dữ liệu huấn luyện và kiểm tra
    independent_vars = data.drop(columns=[target_column])
    X = independent_vars.select_dtypes(include=[np.number])  # Chỉ chọn biến số
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.select_dtypes(include=[np.number])
    
    # 5. Kiểm tra VIF
    X_train_filtered, vif_data = remove_multicollinearity(X_train)
    print(X_train_filtered.columns)  # Kiểm tra các cột trong dữ liệu
    
    # Đồng bộ hóa các biến của X_test với X_train_filtered
    X_test = X_test[X_train_filtered.columns]
    
    # Chuẩn hóa dữ liệu mà không làm mất tên cột
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test)

    # Chuyển lại thành DataFrame với tên cột
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_filtered.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print("\nVIF sau khi loại bỏ các biến có VIF cao:")
    print(vif_data)
    
    # 6. Phát hiện ngoại lệ (Cook's Distance, Leverage Scores)
    logreg_model = train_logistic_regression(X_train_scaled, y_train)
    X_train_const = add_constant(X_train_scaled)
    cooks_d, leverage_scores = calculate_influence(X_train_scaled, y_train, logreg_model)
    
    # 7. Loại bỏ các điểm ngoại lệ
    X_train_scaled, y_train_filtered, outlier_indices, outlier_leverage_indices = remove_outliers(X_train_scaled, y_train, cooks_d, leverage_scores)
    
    # Kiểm tra tỷ lệ lớp sau khi loại bỏ ngoại lệ
    print("\nSố điểm ngoại lệ đã bị loại bỏ:", len(outlier_indices) + len(outlier_leverage_indices))
    
    # Kiểm tra số lượng mẫu sau khi loại bỏ ngoại lệ
    print(f"Số lượng mẫu trong X_train_filtered sau khi loại bỏ ngoại lệ: {X_train_scaled.shape[0]}")
    print(f"Số lượng mẫu trong y_train_filtered sau khi loại bỏ ngoại lệ: {y_train_filtered.shape[0]}")
    
    # 8. Áp dụng SMOTE để cân bằng dữ liệu (nếu cần thiết)
    if class_balance > 0.7:
        print("\nDữ liệu bị mất cân bằng, áp dụng SMOTE để cân bằng lớp.")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_filtered)
        print("\nPhân phối lớp trong dữ liệu huấn luyện sau SMOTE:")
        print(y_train_smote.value_counts(normalize=True))
    else:
        print("\nDữ liệu không bị mất cân bằng nghiêm trọng, không cần áp dụng SMOTE.")
        X_train_smote, y_train_smote = X_train_scaled, y_train_filtered
    
    # 9. Huấn luyện mô hình Logistic Regression
    logreg_model = train_logistic_regression(X_train_smote, y_train_smote) 
    
    # 10. Đánh giá mô hình
    evaluate_model(logreg_model, X_train_smote, y_train_smote, X_test, y_test)
    
    # 11. Kiểm tra giả định và kết luận
    result = check_assumptions(class_balance, vif_data, outlier_indices, outlier_leverage_indices)
    print("\nDebug: Giá trị của result:", result)
    print("\nKết luận: ", result)    
        
    # Kiểm tra kết luận và xử lý
    if result is not None and "phù hợp" in result.lower():
        print("Kết quả: Phù hợp")
    else:
        print("Kết quả: Không xác định hoặc không phù hợp")

    # Lưu dữ liệu sau PCA
    data_after_pca = apply_pca(X_train_scaled)
    data_after_pca.to_excel("logistic_data_cleaned.xlsx", index=False)
    print("Dữ liệu đã được lưu vào file 'logistic_data_cleaned.xlsx'.")

# Hàm kiểm tra giả định của mô hình
def check_assumptions(class_balance, vif_data, outlier_indices, outlier_leverage_indices):
    print("\n--- Kết luận ---")
    if class_balance > 0.9:
        print("Dữ liệu bị mất cân bằng. Cần xử lý vấn đề này trước khi tiếp tục.")
    else:
        print("Dữ liệu phân phối hợp lý.")
        
    vif_data = vif_data[vif_data["Feature"] != "const"]

    if (vif_data["VIF"] > 5).any():
        print("Dữ liệu có vấn đề đa cộng tuyến. Cần giảm thiểu đa cộng tuyến.")
    else:
        print("Không có đa cộng tuyến nghiêm trọng.")

    if len(outlier_indices) > 0:
        print(f"Có {len(outlier_indices)} điểm dữ liệu có Cook's Distance cao. Cần xử lý điểm ngoại lệ.")
    else:
        print("Không có điểm dữ liệu với Cook's Distance quá cao.")

    if len(outlier_leverage_indices) > 0:
        print(f"Có {len(outlier_leverage_indices)} điểm dữ liệu có leverage scores cao. Cần xử lý điểm ngoại lệ.")
    else:
        print("Không có điểm dữ liệu có leverage scores quá cao.")

    # Kiểm tra tổng quan
    if (class_balance <= 0.9 and
        all(vif_data["VIF"] <= 5) and
        len(outlier_indices) == 0 and
        len(outlier_leverage_indices) == 0):
        print("Dữ liệu đáp ứng các giả định cơ bản của hồi quy logistic.")
    else:
        print("Dữ liệu chưa hoàn toàn đáp ứng các giả định của hồi quy logistic, cần xử lý các vấn đề nêu trên.")

# Chạy chương trình
file_path = 'logistic_data.xlsx'  # Đường dẫn tới file dữ liệu
main(file_path)
