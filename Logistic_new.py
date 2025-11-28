import math
import pandas as pd
import numpy as np
import time 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# Đọc và xử lý dữ liệu
def load_and_prepare_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[
        'person_age', 'person_gender', 'person_income', 'person_emp_exp', 'person_home_ownership', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file', 'loan_status'
    ])
    df.dropna(inplace=True)
    
    # Mã hóa các cột phân loại (categorical)
    label_encoder = LabelEncoder()
    columns = ['person_gender', 'person_home_ownership', 'previous_loan_defaults_on_file']
    # Lặp qua các cột và mã hóa từng cột
    for column in columns:
        print(f"Trước khi mã hóa {column}:\n{df[column]}\n")
        
        # Mã hóa cột phân loại
        df[column] = label_encoder.fit_transform(df[column])
        
        print(f"Sau khi mã hóa {column}:\n{df[column]}\n")
        print(f"Mã ánh xạ cho {column}: {dict(enumerate(label_encoder.classes_))}\n")
    
    x = df[['person_age', 'person_gender',  'person_income', 'person_emp_exp', 'person_home_ownership', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file']].values
    y = df['loan_status'].values
    
    return df, x, y

def calculate_tp_fp(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return TP / (TP + FP)

def precision(y_true, y_pred, class_label):
    if class_label == 1:
        tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positive
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positive
    elif class_label == 0:
        tp = np.sum((y_true == 0) & (y_pred == 0))  # True Negative (Precision của class 0)
        fp = np.sum((y_true == 1) & (y_pred == 0))  # False Negative
    else:
        raise ValueError("class_label must be either 0 or 1")
    
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def precision_sklearn(y_true, y_pred):
    return precision_score(y_true, y_pred)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_m_and_b(x_train, y_train, learning_rate=0.1, epochs=000, tol=1e-5):
    n_samples, n_features = x_train.shape
    m = np.zeros(n_features)  # Trọng số ban đầu
    b = 0  # Bias ban đầu
    
    for epoch in range(epochs):
        z = np.dot(x_train, m) + b  # Sự kết hợp tuyến tính
        p1 = sigmoid(z)  # Xác suất P(1)
        
        # Tính toán sai số
        error = y_train - p1
        
        # Cập nhật trọng số và bias
        m_gradient = - (1/n_samples) * np.dot(x_train.T, error)  # Gradient của m
        b_gradient = - (1/n_samples) * np.sum(error)  # Gradient của b
        
        m_new = m - learning_rate * m_gradient  # Cập nhật m
        b_new = b - learning_rate * b_gradient  # Cập nhật b
        
        # Kiểm tra sự thay đổi giữa các trọng số và bias
        if np.linalg.norm(m_new - m) < tol and abs(b_new - b) < tol:
            print(f"Đạt độ chính xác dừng tại epoch {epoch}")
            break
        m, b = m_new, b_new  # Cập nhật m và b cho lần lặp tiếp theo

    return m, b, z 

def log_loss(y, p1):
    loss_per_sample = []
    for i in range(len(y)):
        # Tránh trường hợp p1[i] bằng 0 hoặc 1
        if p1[i] == 0:
            p1[i] = 1e-15
        elif p1[i] == 1:
            p1[i] = 1 - 1e-15
        # Tính log loss cho từng mẫu
        sample_loss = -(y[i] * math.log(p1[i]) + (1 - y[i]) * math.log(1 - p1[i]))
        loss_per_sample.append(sample_loss)
    return loss_per_sample

def train_and_predict_sklearn(x_train, y_train, x_test):
    start_time = time.time()
    model = LogisticRegression(solver='saga', max_iter=8000, C=0.1, tol=1e-5)
    model.fit(x_train, y_train)
    running_time_sklearn = time.time() - start_time
    y_train_pred_sklearn = model.predict_proba(x_train)[:, 1]
    y_test_pred_sklearn = model.predict_proba(x_test)[:, 1]
    return model, y_train_pred_sklearn, y_test_pred_sklearn, running_time_sklearn

def predict_my_code(x_test, m, b):
    z = np.dot(x_test, m) + b  # Linear combination
    p1 = sigmoid(z)  # Xác suất P(1)
    return p1

def encode_new_data(df_new):
    label_encoder = LabelEncoder()
    df_new['person_gender'] = label_encoder.fit_transform(df_new['person_gender'])
    df_new['person_home_ownership'] = label_encoder.fit_transform(df_new['person_home_ownership'])
    df_new['previous_loan_defaults_on_file'] = label_encoder.fit_transform(df_new['previous_loan_defaults_on_file'])
    return df_new

def load_new_data(file_path, new_data_sheet_name, scaler):
    df_new = pd.read_excel(file_path, sheet_name=new_data_sheet_name, usecols=[ 
        'person_age', 'person_gender', 'person_income', 'person_emp_exp', 'person_home_ownership', 'loan_amnt',
         'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file'
    ])
    df_new.dropna(inplace=True)
    df_new = encode_new_data(df_new)  # Mã hóa dữ liệu
    x_new = df_new.values
    x_new_scaled = scaler.transform(x_new)  # Chuẩn hóa dữ liệu mới
    return x_new_scaled

def predict_new_data(file_path, scaler, model=None, m=None, b=None):
    # Đọc dữ liệu mới
    x_new_scaled = load_new_data(file_path, 'Sheet1', scaler)  # load_new_data sẽ chuẩn hóa dữ liệu mới
    if model is not None:
        y_new_pred_sklearn = model.predict_proba(x_new_scaled)[:, 1]  # Dự đoán xác suất của lớp 1
        y_new_pred_sklearn = (y_new_pred_sklearn > 0.5)  # Áp dụng ngưỡng 0.5
        print("Dự đoán (Sklearn):", y_new_pred_sklearn)
    elif m is not None and b is not None:
        y_new_pred = predict_my_code(x_new_scaled, m, b)  # Dự đoán bằng mã của bạn
        y_new_pred= (y_new_pred > 0.5)  # Áp dụng ngưỡng 0.5
        print("Dự đoán (My Code):", y_new_pred)
    else:
        print("Chưa có mô hình để dự đoán.")
        
def plot_results(precision_test, y_test, y_test_pred, log_loss_train, log_loss_test, log_loss_train_sklearn, 
                 y_test_pred_sklearn, running_time, running_time_sklearn, log_loss_test_sklearn, 
                 precision_test_sklearn, sklearn_log_loss_train, sklearn_log_loss_test, running_time_values):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    step = 200
    
    # Lấy mẫu các giá trị
    log_loss_train_sampled = log_loss_train[::step]
    log_loss_test_sampled = log_loss_test[::step]

    # In giá trị log loss sau khi lấy mẫu
    #print("Log Loss Train (Sampled with step=200):\n", log_loss_train_sampled)
    #print("Log Loss Test (Sampled with step=200):\n", log_loss_test_sampled)

    axes[0, 0].plot(range(1, len(log_loss_train) + 1, step), log_loss_train[::step], label="Log Loss Train", color='blue', linewidth=2)
    axes[0, 0].plot(range(1, len(log_loss_test) + 1, step), log_loss_test[::step], label="Log Loss Test", color='cyan', linestyle='--', linewidth=2)

    axes[0, 0].set_title('Log Loss: Train vs Test (My Code)')
    axes[0, 0].set_ylabel('Log Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, max(max(log_loss_train), max(log_loss_test)) * 1.1)
    
    #Precision vs Running Time
    #print(f"Độ chính xác của My Code: {precision_test}, Độ chính xác của Sklearn: {precision_test_sklearn}")
    #print(f"Thời gian chạy của My Code: {running_time}, Thời gian chạy của Sklearn: {running_time_sklearn}")
    axes[0, 1].plot([0, precision_test], [0, running_time], label="My Code", color='blue', linewidth=2)
    axes[0, 1].plot([0, precision_test_sklearn], [0, running_time_sklearn], label="Sklearn", color='cyan', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Running Time & Precision: My Code vs Sklearn')
    axes[0, 1].set_xlabel('Precision (Accuracy)')
    axes[0, 1].set_ylabel('Running Time (seconds)')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, max(np.max(running_time_values), np.max(running_time_values)) * 1.1)

    # Compare Log Loss: Train vs Test for Sklearn
    # Lấy mẫu các giá trị
    log_loss_train_sampled_sklearn = log_loss_train_sklearn[::step]
    log_loss_test_sampled_sklearn = log_loss_test_sklearn[::step]

    # In giá trị log loss sau khi lấy mẫu
    #print("Log Loss Train (Sampled with step=200):\n", log_loss_train_sampled)
    #print("Log Loss Test (Sampled with step=200):\n", log_loss_test_sampled)
    axes[1, 0].plot(range(1, len(log_loss_train_sklearn) + 1, step), log_loss_train_sklearn[::step], label="Log Loss Train", color='blue', linewidth=2)
    axes[1, 0].plot(range(1, len(log_loss_test_sklearn) + 1, step), log_loss_test_sklearn[::step], label="Log Loss Test", color='cyan', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Log Loss: Train vs Test (Sklearn)')
    axes[1, 0].set_ylabel('Log Loss')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, max(max(log_loss_train_sklearn), max(log_loss_test_sklearn)) * 1.1)

    # Compare Log Loss Train vs Log Loss Test: My Code vs Sklearn
    
    print(f"Log loss của tập Train (My Code): {np.mean(log_loss_train)}, Log loss của tập Train (Sklearn): {sklearn_log_loss_train}")
    print(f"Log loss của tập Test (My Code): {np.mean(log_loss_test)}, Log loss của tập Test (Sklearn): {sklearn_log_loss_test}")
    axes[1, 1].plot([0, np.mean(log_loss_train)], [0, np.mean(log_loss_test)], label="My Code", color='blue', linewidth=2)
    axes[1, 1].plot([0, sklearn_log_loss_train], [0, sklearn_log_loss_test], label="Sklearn", color='cyan', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Log loss Train & Log loss Test: My Code vs Sklearn')
    axes[1, 1].set_xlabel('Log loss Train')  
    axes[1, 1].set_ylabel('Log loss Test')  
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, max(np.mean(log_loss_test), sklearn_log_loss_test) * 1.1)

    # Tighten layout and show the plot
    plt.tight_layout()
    plt.show()
    
def main():
    # Đường dẫn và xử lý dữ liệu
    #file_path = r'D:/Đồ án chuyên ngành/Nhóm 07/Logistic_test.xlsx'
    file_path = r'D:/Đồ án chuyên ngành/Nhóm 07/processed_logistic.xlsx'
    sheet_name = 'Sheet1' 
    #sheet_name = 'Sheet1'
    
    df, x, y = load_and_prepare_data(file_path, sheet_name)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Huấn luyện mô hình tự viết
    start_time = time.time()
    m, b, z = calculate_m_and_b(x_train, y_train, learning_rate=0.1, epochs=8000, tol=1e-5)
    y_test_pred = predict_my_code(x_test, m, b)
    y_train_pred = predict_my_code(x_train, m, b)
    running_time = time.time() - start_time
    
    # Huấn luyện mô hình sklearn
    model_sklearn, y_train_pred_sklearn, y_test_pred_sklearn, running_time_sklearn = train_and_predict_sklearn(x_train, y_train, x_test)
    
if __name__ == "__main__":
    main()
    
    