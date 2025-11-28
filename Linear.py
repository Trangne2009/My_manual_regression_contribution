import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle 
import math

# Đọc và xử lý dữ liệu
def load_and_prepare_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[
       'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
        'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
        'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
    ])
    df.dropna(inplace=True)
    x = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
            'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 
            'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']].values
    y = df['price'].values
    return x, y
"""
def load_and_prepare_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=['Temperature (F)', 'Tourists', 'Sunny Days', 'Ice Cream Sale (dollar in thousand)'])
    df.dropna(inplace=True)
    x = df[['Temperature (F)', 'Tourists', 'Sunny Days']].values
    y = df['Ice Cream Sale (dollar in thousand)'].values
    return train_test_split(x, y, test_size=0.2, random_state=42)
"""
# Hồi quy tuyến tính tự viết
def hoi_quy_tuyen_tinh(x, y):
    # Đảm bảo rằng X là ma trận và y là vector cột
    x = np.array(x)
    y = np.array(y)
    x_b = np.c_[np.ones((x.shape[0], 1)), x] # Thêm cột 1 vào x
    print("Kích thước của X_b sau khi thêm cột 1:", x_b.shape)
    print(y.shape) 
    XTX = x_b.T.dot(x_b)  # Tính X^T.X
    cond_number = np.linalg.cond(XTX)
    print("Condition number of X^T * X:", cond_number)

    # Nếu điều kiện số quá lớn, áp dụng ma trận giả nghịch
    if cond_number > 1e12:
        print("Warning: X^T * X có thể gặp vấn đề về độ điều kiện (ill-conditioned), sử dụng ma trận giả nghịch.")
        XTX_inv = np.linalg.pinv(XTX)  # tính giả nghịch (X^TX)+
    else:
        XTX_inv = np.linalg.inv(XTX)
        
    B = XTX_inv.dot(x_b.T).dot(y)
    return B

# Huấn luyện và dự đoán với code tự viết
def train_and_predict(x_train, y_train, x_test):
    x_b_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]  
    x_b_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]  

    start_time = time.time()
    B = hoi_quy_tuyen_tinh(x_train, y_train)
    running_time = time.time() - start_time
    
    y_train_pred = x_b_train.dot(B)  
    y_test_pred = x_b_test.dot(B)    
    
    return B, y_train_pred, y_test_pred, running_time

# Huấn luyện và dự đoán với sklearn
def train_and_predict_sklearn(x_train, y_train, x_test):
    start_time = time.time()
    model = LinearRegression().fit(x_train, y_train)
    running_time_sklearn = time.time() - start_time
    y_train_pred_sklearn = model.predict(x_train)
    y_test_pred_sklearn = model.predict(x_test)
    return model, y_train_pred_sklearn, y_test_pred_sklearn, running_time_sklearn

# MSE: đo lường độ lệch trung bình bình phương
def mse_my_code(y_true, y_pred):
    # Đảm bảo y_true và y_pred có cùng chiều dài
    if len(y_true) != len(y_pred):
        raise ValueError("y_true và y_pred phải có cùng số lượng phần tử.")
    
    # Tính MSE
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)
    return mse

# Hàm tính R-squared (độ chính xác)
def r_squared(y_true, y_pred):
    # Đảm bảo y_true và y_pred có cùng chiều dài
    if len(y_true) != len(y_pred):
        raise ValueError("y_true và y_pred phải có cùng số lượng phần tử.")
    
    # Tính giá trị trung bình của y_true
    y_mean = sum(y_true) / len(y_true)
    
    # Tính tổng sai số (ss_total)
    ss_total = sum((y - y_mean) ** 2 for y in y_true)
    
    # Tính tổng sai số dựa trên sự khác biệt giữa giá trị thực tế và giá trị dự đoán (ss_residual)
    ss_residual = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    
    # Tính R²
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Vẽ biểu đồ
def plot_predictions(y_train, y_test, y_test_pred, y_train_pred, y_train_pred_sklearn, y_test_pred_sklearn):
    plt.figure(figsize=(12, 6))
    
    # Biểu đồ phân tán cho My Code
    plt.subplot(1, 2, 1)  # Biểu đồ đầu tiên
    plt.scatter(y_train, y_train_pred, color='blue', label='Predictions Train (My Code)', alpha=0.7)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='orange', linestyle='--', label='True values Train')
    plt.scatter(y_test, y_test_pred, color='green', label='Predictions Test (My Code)', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='True values Test')
    plt.title('Actual vs Predicted Train & Test (My Code)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.axis('equal')  # Đảm bảo tỷ lệ trục x và y giống nhau

    # Biểu đồ phân tán cho Sklearn
    plt.subplot(1, 2, 2)  # Biểu đồ thứ hai
    plt.scatter(y_train, y_train_pred_sklearn, color='blue', label='Predictions Train (Sklearn)', alpha=0.7)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='orange', linestyle='--', label='True values Train')
    plt.scatter(y_test, y_test_pred_sklearn, color='green', label='Predictions Test (Sklearn)', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='True values Test')
    plt.title('Actual vs Predicted (Sklearn)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.axis('equal')  # Đảm bảo tỷ lệ trục x và y giống nhau

    plt.tight_layout()
    plt.show()

def save_model(file_path, model):
    with open(file_path, 'wb') as file: 
        pickle.dump(model, file)
    print(f"Model đã được lưu vào: {file_path}") 
    
def load_saved_model(file_path): 
    with open(file_path, 'rb') as file:
        model = pickle.load(file) 
    print(f"Model đã được tải từ: {file_path}")
    return model

# Hàm tải và chuẩn hóa dữ liệu mới
def load_new_data(file_path, sheet_name):
    df_new = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
        'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 
        'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
    ])
    df_new.dropna(inplace=True)
    x_new = df_new.values
    return x_new

# Hàm dự đoán từ dữ liệu mới
def predict_new_data(x_new_scaled, B=None, sklearn_model=None):
    if B is not None:
        y_new_pred = x_new_scaled.dot(B)
    elif sklearn_model is not None:
        y_new_pred = sklearn_model.predict(x_new_scaled[:, 1:])  # Không cần cột 1 cho mô hình sklearn
    else:
        raise ValueError("Cần cung cấp B hoặc sklearn_model để dự đoán.")
    return y_new_pred
    
def process_single_data_point(new_data, scaler):
    # Chuyển đổi new_data thành mảng numpy và chuẩn hóa
    new_data = np.array(new_data).reshape(1, -1)  # Đảm bảo là mảng 2D
    new_data_scaled = scaler.transform(new_data)  # Chuẩn hóa dữ liệu
    new_data_scaled = np.c_[np.ones((new_data_scaled.shape[0], 1)), new_data_scaled]  # Thêm cột 1s
    return new_data_scaled

def main():
    #file_path = 'D:\\Đồ án chuyên ngành\\Nhóm 07\\linear_data.xlsx'
    file_path = 'D:\\Đồ án chuyên ngành\\Nhóm 07\\processed_data.xlsx'
    #file_path = 'D:\\Đồ án chuyên ngành\\Nhóm 07\\Excel\\Linear.xlsx'
    #sheet_name = 'Worksheet'
    sheet_name = 'Sheet1'
    #sheet_name = 'temp-sale'
    
    x, y = load_and_prepare_data(file_path, sheet_name)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Chuẩn hóa dữ liệu 
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)  
    x_test_scaled = scaler.transform(x_test)  
    
    # Huấn luyện và dự đoán với mô hình tự viết
    B, y_train_pred, y_test_pred, running_time = train_and_predict(x_train_scaled, y_train, x_test_scaled)
    mse_train = mse_my_code(y_train, y_train_pred)
    mse_test = mse_my_code(y_test, y_test_pred)
    r2_train = r_squared(y_train, y_train_pred)  
    r2_test = r_squared(y_test, y_test_pred)  
    
    # Huấn luyện và dự đoán với mô hình sklearn
    model, y_train_pred_sklearn, y_test_pred_sklearn, running_time_sklearn = train_and_predict_sklearn(x_train_scaled, y_train, x_test_scaled)
    mse_train_sklearn = mean_squared_error(y_train, y_train_pred_sklearn)
    mse_test_sklearn = mean_squared_error(y_test, y_test_pred_sklearn)
    r2_train_sklearn = r2_score(y_train, y_train_pred_sklearn)  
    r2_test_sklearn = r2_score(y_test, y_test_pred_sklearn)
    
    print(f"- MSE Train: {mse_train}, MSE Test: {mse_test}")
    print(f"- MSE Train Sklearn: {mse_train_sklearn}, MSE Test Sklearn: {mse_test_sklearn}")
    print(f"- R² Train: {r2_train}, R² Test: {r2_test}")
    print(f"- R² Train Sklearn: {r2_train_sklearn}, R² Test Sklearn: {r2_test_sklearn}")
    print("Running Time (My Code):", running_time, "giây")
    print("Running Time (Sklearn):", running_time_sklearn, "giây")
    
    plot_predictions(y_train, y_test, y_test_pred, y_train_pred, y_train_pred_sklearn, y_test_pred_sklearn)
    
    df_results_train = pd.DataFrame({
        'Y_true (Train)': y_train,
        'y_pred train (My code)': y_train_pred,
        'y_pred train (Sklearn)': y_train_pred_sklearn,
    })

    # Lưu bảng kết quả vào file Excel
    df_results_train.to_excel('prediction_train.xlsx', index=False)

    print("Bảng kết quả đã được lưu vào 'prediction_train.xlsx'")
    
    df_results_test = pd.DataFrame({
    'Y_true (Test)': y_test,
        'y_pred test (My code)': y_test_pred,
        'y_pred test (Sklearn)': y_test_pred_sklearn,
    })
    
    df_results_test.to_excel('prediction_test.xlsx', index=False)

    print("Bảng kết quả đã được lưu vào 'prediction_test.xlsx'")

# Chạy chương trình chính
main()
