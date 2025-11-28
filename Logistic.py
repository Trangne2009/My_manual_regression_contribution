import math
import pandas as pd
import numpy as np
import time 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss as sklearn_log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df = pd.read_excel('processed_logistic.xlsx')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_accuracy(y_true, y_pred):
    # Tính Accuracy thủ công
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    return correct_predictions / total_samples

def calculate_m_and_b(x_train, y_train, x_test, y_test, learning_rate=0.1, epochs=500, tol=1e-5):
    n_samples, n_features = x_train.shape
    m = np.zeros(n_features)  # Trọng số ban đầu
    b = 0  # Bias ban đầu
    train_loss_history = []  # Lưu trữ giá trị loss của train
    test_loss_history = []  # Lưu trữ giá trị loss của test
    train_accuracy_history = []  # Lưu trữ giá trị accuracy của train
    test_accuracy_history = []  # Lưu trữ giá trị accuracy của test

    for epoch in range(epochs):
        z_train = np.dot(x_train, m) + b  # Sự kết hợp tuyến tính cho train
        p_train = sigmoid(z_train)  # Xác suất P(1) cho train

        z_test = np.dot(x_test, m) + b  # Sự kết hợp tuyến tính cho test
        p_test = sigmoid(z_test)  # Xác suất P(1) cho test

        # Tính toán log loss
        train_loss = -np.mean(y_train * np.log(p_train) + (1 - y_train) * np.log(1 - p_train))
        test_loss = -np.mean(y_test * np.log(p_test) + (1 - y_test) * np.log(1 - p_test))

        # Lưu trữ loss
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # Tính toán Accuracy
        train_accuracy = calculate_accuracy(y_train, (p_train >= 0.5).astype(int))
        test_accuracy = calculate_accuracy(y_test, (p_test >= 0.5).astype(int))

        # Lưu trữ Accuracy
        train_accuracy_history.append(train_accuracy)
        test_accuracy_history.append(test_accuracy)

        # Cập nhật trọng số và bias
        m_gradient = - (1/n_samples) * np.dot(x_train.T, (y_train - p_train))  # Gradient của m
        b_gradient = - (1/n_samples) * np.sum(y_train - p_train)  # Gradient của b
        
        m_new = m - learning_rate * m_gradient  # Cập nhật m
        b_new = b - learning_rate * b_gradient  # Cập nhật b
        
        # Kiểm tra sự thay đổi giữa các trọng số và bias
        if np.linalg.norm(m_new - m) < tol and abs(b_new - b) < tol:
            print(f"Đạt độ chính xác dừng tại epoch {epoch}")
            break
        m, b = m_new, b_new  # Cập nhật m và b cho lần lặp tiếp theo

    return m, b, train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history

def log_loss(y, p1):
    loss_per_sample = []
    for i in range(len(y)):
        # Tránh trường hợp p1[i] bằng 0 hoặc 1
        if p1[i] == 0:
            p1[i] = 1e-15
        elif p1[i] == 1:
            p1[i] = 1 - 1e-15
        # Tính log loss cho từng mẫu
        sample_loss = -(y.iloc[i] * np.log(p1[i]) + (1 - y.iloc[i]) * np.log(1 - p1[i]))
        loss_per_sample.append(sample_loss)
    return loss_per_sample
 
def train_and_predict_sklearn(x_train, y_train, x_test, y_test, epochs=500, tol=1e-5):
    
    train_loss_sklearn_history = []
    test_loss_sklearn_history = []
    train_accuracy_sklearn_history = []
    test_accuracy_sklearn_history = []
    
    for epoch in range(1, epochs + 1):
        # Thuật toán Scikit-learn (huấn luyện lại theo từng epoch)
        model_sklearn = LogisticRegression(solver='saga', max_iter=epoch, C=0.1, tol=tol)
        model_sklearn.fit(x_train, y_train)
        y_train_pred_sklearn = model_sklearn.predict_proba(x_train)[:, 1]
        y_test_pred_sklearn = model_sklearn.predict_proba(x_test)[:, 1]
        
        train_loss_sklearn = sklearn_log_loss(y_train, y_train_pred_sklearn)
        test_loss_sklearn = sklearn_log_loss(y_test, y_test_pred_sklearn)
        
        # Tính Accuracy
        train_accuracy_sklearn = accuracy_score(y_train, (y_train_pred_sklearn >= 0.5))
        test_accuracy_sklearn = accuracy_score(y_test, (y_test_pred_sklearn >= 0.5))

        # Lưu lại vào lịch sử
        train_loss_sklearn_history.append(train_loss_sklearn)
        test_loss_sklearn_history.append(test_loss_sklearn)
        train_accuracy_sklearn_history.append(train_accuracy_sklearn)
        test_accuracy_sklearn_history.append(test_accuracy_sklearn)
    
    return train_loss_sklearn_history, test_loss_sklearn_history, train_accuracy_sklearn_history, test_accuracy_sklearn_history, model_sklearn

def predict_my_code(x_test, m, b):
    z = np.dot(x_test, m) + b  
    p1 = sigmoid(z)  
    return p1

def plot(train_loss, test_loss, train_accuracy, test_accuracy, 
         train_loss_sklearn_history, test_loss_sklearn_history, train_accuracy_sklearn_history, test_accuracy_sklearn_history):
    # Vẽ biểu đồ loss cho mô hình tự viết
    plt.plot(train_loss, label='Train Loss (My Code)', color='blue')
    plt.plot(test_loss, label='Test Loss (My Code)', color='orange')

    # Vẽ biểu đồ loss cho mô hình sklearn
    plt.plot(train_loss_sklearn_history, label='Train Loss (Sklearn)', color='green', linestyle='--')
    plt.plot(test_loss_sklearn_history, label='Test Loss (Sklearn)', color='red', linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss for My Code and Sklearn')
    plt.legend()
    plt.show()

    # Vẽ biểu đồ accuracy cho mô hình tự viết
    plt.plot(train_accuracy, label='Train Accuracy (My Code)', color='blue')
    plt.plot(test_accuracy, label='Test Accuracy (My Code)', color='orange')

    # Vẽ biểu đồ accuracy cho mô hình sklearn
    plt.plot(train_accuracy_sklearn_history, label='Train Accuracy (Sklearn)', color='green', linestyle='--')
    plt.plot(test_accuracy_sklearn_history, label='Test Accuracy (Sklearn)', color='red', linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for My Code and Sklearn')
    plt.legend()
    plt.show()
    
def plot_misclassifications(misclassified_train_my_code, misclassified_test_my_code,
                            misclassified_train_sklearn, misclassified_test_sklearn):
    # Dữ liệu để vẽ biểu đồ
    categories = ['Train', 'Test']
    my_code_values = [misclassified_train_my_code, misclassified_test_my_code]
    sklearn_values = [misclassified_train_sklearn, misclassified_test_sklearn]

    x = np.arange(len(categories))  # Vị trí các cột
    width = 0.35  # Độ rộng của mỗi cột

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, my_code_values, width, label='My Code', color='blue')
    ax.bar(x + width/2, sklearn_values, width, label='Sklearn', color='orange')

    # Thiết lập nhãn và tiêu đề
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Number of Misclassifications', fontsize=12)
    ax.set_title('Misclassifications (My Code vs Sklearn)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)

    # Hiển thị giá trị trên các cột
    for i, v in enumerate(my_code_values):
        ax.text(i - width/2, v + 0.5, str(v), ha='center', fontsize=10)
    for i, v in enumerate(sklearn_values):
        ax.text(i + width/2, v + 0.5, str(v), ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()
    
def plot_running_time(running_time_my_code, running_time_sklearn):
    # Dữ liệu để vẽ biểu đồ
    methods = ['My Code', 'Sklearn']
    running_times = [running_time_my_code, running_time_sklearn]

    x = np.arange(len(methods))  # Vị trí các cột

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(x, running_times, color=['blue', 'orange'], width=0.5)

    # Thiết lập nhãn và tiêu đề
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Running Time (seconds)', fontsize=12)
    ax.set_title('Comparison of Running Time', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)

    # Hiển thị giá trị trên các cột
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{height:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

def main():
    x = df.drop(columns=['loan_status'])  # Dữ liệu đầu vào (loại bỏ cột loan_status)
    y = df['loan_status']  # Cột loan_status là nhãn mục tiêu

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình tự viết
    start_time = time.time()
    m, b, train_loss, test_loss, train_accuracy, test_accuracy = calculate_m_and_b(x_train, y_train, x_test, y_test, learning_rate=0.1, epochs=500)
    y_test_pred = (predict_my_code(x_test, m, b) >= 0.5).astype(int)
    y_train_pred = (predict_my_code(x_train, m, b) >= 0.5).astype(int)
    running_time = time.time() - start_time
    # Huấn luyện mô hình sklearn
    start_time = time.time()
    train_loss_sklearn_history, test_loss_sklearn_history, train_accuracy_sklearn_history, test_accuracy_sklearn_history, model_sklearn = train_and_predict_sklearn(x_train, y_train, x_test, y_test, epochs=500, tol=1e-5)
    y_train_pred_sklearn = model_sklearn.predict(x_train)  # Dự đoán xác suất cho lớp 1
  # Dự đoán lớp cho train từ sklearn
    y_test_pred_sklearn = model_sklearn.predict(x_test)
    running_time_sklearn = time.time() - start_time

    df_comparison_train = pd.DataFrame({
    'Giá trị thực tế Train': y_train,
    'Giá trị dự đoán Train (My Code)': y_train_pred,
    'Giá trị dự đoán Train (Sklearn)': y_train_pred_sklearn,
    })
    
    # Lưu bảng vào file Excel
    df_comparison_train.to_excel('comparison(train)_results.xlsx', index=False)

    print("Bảng đã được lưu vào 'comparison(train)_results.xlsx'")
    
    df_comparison_test = pd.DataFrame({
    'Giá trị thực tế Test': y_test,
    'Giá trị dự đoán Test (My Code)': y_test_pred,
    'Giá trị dự đoán Test (Sklearn)': y_test_pred_sklearn
    })
    
    # Lưu bảng vào file Excel
    df_comparison_test.to_excel('comparison(test)_results.xlsx', index=False)

    print("Bảng đã được lưu vào 'comparison(test)_results.xlsx'")
    
    print("Running Time (My Code):", running_time, "giây")
    print("Running Time (Sklearn):", running_time_sklearn, "giây")
    
    # Tạo DataFrame với các giá trị train và test loss và accuracy
    df_results = pd.DataFrame({
        'Epoch': range(1, len(train_loss_sklearn_history) + 1),
        'Train Loss (Sklearn)': train_loss_sklearn_history,
        'Train Loss (My Code)': train_loss,
        'Test Loss (Sklearn)': test_loss_sklearn_history,
        'Test Loss (My Code)': test_loss,
        'Train Accuracy (Sklearn)': train_accuracy_sklearn_history,
        'Train Accuracy (My Code)': train_accuracy,
        'Test Accuracy (Sklearn)': test_accuracy_sklearn_history,
        'Test Accuracy (My Code)': test_accuracy,
    })

    # Lưu bảng kết quả vào file Excel
    df_results.to_excel('loss_accuracy.xlsx', index=False)

    print("Bảng kết quả đã được lưu vào 'loss_accuracy.xlsx'")

    plot(train_loss, test_loss, train_accuracy, test_accuracy, 
         train_loss_sklearn_history, test_loss_sklearn_history, train_accuracy_sklearn_history, test_accuracy_sklearn_history)
    
    # Tính số lượng dự đoán sai cho train và test (My Code)
    misclassified_train_my_code = np.sum(y_train != y_train_pred)
    misclassified_test_my_code = np.sum(y_test != y_test_pred)

    # Tính số lượng dự đoán sai cho train và test (Sklearn)
    misclassified_train_sklearn = np.sum(y_train != y_train_pred_sklearn)
    misclassified_test_sklearn = np.sum(y_test != y_test_pred_sklearn)

    # Vẽ biểu đồ thể hiện số lượng dự đoán sai
    plot_misclassifications(misclassified_train_my_code, misclassified_test_my_code,
                            misclassified_train_sklearn, misclassified_test_sklearn)
    
    # Vẽ biểu đồ so sánh thời gian chạy
    plot_running_time(running_time, running_time_sklearn)
     
if __name__ == "__main__":
    main() 
    


    