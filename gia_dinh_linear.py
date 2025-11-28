import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import Linear
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Giả định 1: Kiểm tra độ độc lập của sai số - Dùng đồ thị phân tán của residuals
def check_independence_of_errors(model, X, y):
    if isinstance(model, sm.OLS):  # Nếu là mô hình statsmodels
        residuals = model.resid
        fitted_values = model.fittedvalues
    elif hasattr(model, 'predict'):  # Nếu là mô hình có phương thức predict (như sklearn)
        fitted_values = model.predict(X)  # Dự đoán giá trị từ mô hình sklearn
        residuals = y - fitted_values  # Residuals là sự khác biệt giữa y thực tế và giá trị dự đoán
    else:  # Nếu model là numpy array chứa giá trị dự đoán
        fitted_values = model  # Nếu model là một array numpy chứa giá trị dự đoán
        residuals = y - fitted_values  # Residuals là sự khác biệt giữa y thực tế và giá trị dự đoán
    
    # Vẽ đồ thị phân tán của residuals vs fitted values
    plt.scatter(fitted_values, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

# Giả định 2: Kiểm tra Homoscedasticity - Dùng đồ thị phân tán residuals
def check_homoscedasticity(model):
    # Lấy residuals
    residuals = model.resid
    # Lấy fitted values
    fitted_values = model.fittedvalues
    
    # Vẽ đồ thị phân tán residuals vs fitted values
    plt.scatter(fitted_values, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Fitted Values (Check for Homoscedasticity)')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

    # Kiểm tra bằng test Breusch-Pagan
    from statsmodels.stats.diagnostic import het_breuschpagan
    _, pval, _, _ = het_breuschpagan(residuals, model.model.exog)
    print(f"Breusch-Pagan p-value: {pval}")
    if pval < 0.05:
        print("Có hiện tượng heteroscedasticity (Sai số không đồng đều).")
    else:
        print("Không có hiện tượng heteroscedasticity (Sai số đồng đều).")

# Giả định 3: Kiểm tra phân phối chuẩn của sai số
def check_normality_of_errors(model):
    residuals = model.resid
    # Vẽ histogram của residuals
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.show()
    
    # Kiểm tra phân phối chuẩn với test Shapiro-Wilk
    from scipy import stats
    stat, pval = stats.shapiro(residuals)
    print(f"Shapiro-Wilk p-value: {pval}")
    if pval < 0.05:
        print("Sai số không phân phối chuẩn.")
    else:
        print("Sai số phân phối chuẩn.")

# Giả định 4: Kiểm tra đa cộng tuyến
def check_multicollinearity(X):
    # Tính VIF cho mỗi biến độc lập
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print("Variance Inflation Factors (VIF):")
    print(vif_data)
    
    # Nếu VIF > 10, thì có thể có vấn đề với đa cộng tuyến
    if any(vif_data["VIF"] > 10):
        print("Có vấn đề với đa cộng tuyến!")
    else:
        print("Không có vấn đề với đa cộng tuyến.")

def main(): 
    file_path = 'D:\\Đồ án chuyên ngành\\Nhóm 07\\linear_data.xlsx'
    #file_path = 'D:\\Đồ án chuyên ngành\\Nhóm 07\\Excel\\Linear.xlsx'
    sheet_name = 'Worksheet'
    #sheet_name = 'temp-sale'
    # Đường dẫn và tên sheet cho dữ liệu mới
    new_data_sheet_name = 'Sheet1'
    #new_data = [93, 1212, 6]  # Một điểm dữ liệu mới: [Temperature (F), Tourists, Sunny Days]
    
    x, y = Linear.load_and_prepare_data(file_path, sheet_name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    my_code_model = Linear.load_saved_model('my_code.pkl')
    print(type(my_code_model))
    sklearn = Linear.load_saved_model('sklearn_model.pkl')
    print("Kích thước của y_train:", y_train.shape)
    print("Kích thước của x_train:", x_train.shape)
    check_independence_of_errors(my_code_model, x_train, y_train)  # Kiểm tra độc lập sai số
    check_homoscedasticity(my_code_model)  # Kiểm tra đồng đều sai số
    check_normality_of_errors(my_code_model)  # Kiểm tra phân phối chuẩn sai số
    check_multicollinearity(x_train) 

    # Áp dụng các hàm kiểm tra giả định
    check_independence_of_errors(sklearn, x_train, y_train)  # Kiểm tra độc lập sai số
    check_homoscedasticity(sklearn)  # Kiểm tra đồng đều sai số
    check_normality_of_errors(sklearn)  # Kiểm tra phân phối chuẩn sai số
    check_multicollinearity(x_train)  # Kiểm tra đa cộng tuyến với X
    
main()
