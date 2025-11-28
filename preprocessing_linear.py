import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Hàm xử lý ngoại lai bằng cách thay thế bằng min_value và max_value
def replace_outliers(df, columns):
    for column in columns:
        if column not in df.columns:
            print(f"Warning: Cột {column} không tồn tại trong DataFrame")
            continue  # Skip this column if it does not exist
        
       
        
        # Xử lý NaN
        df[column] = df[column].fillna(df[column].mean())

        # Sắp xếp dữ liệu
        df = df.sort_values(by=column)
        
        # Tính IQR và thay thế ngoại lai
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Thay thế ngoại lai bằng min_value hoặc max_value
        df[column] = np.where(df[column] < lower_bound, lower_bound, 
                              np.where(df[column] > upper_bound, upper_bound, df[column]))
    
    return df

# Hàm chuẩn hóa Min-Max
def min_max_scaling(df, columns):
    # Duyệt qua từng cột cần chuẩn hóa
    for col in columns:
        if col not in df.columns:  
            print(f"Warning: Cột {col} không tồn tại trong DataFrame")
            continue  # Bỏ qua cột nếu không tồn tại trong DataFrame
        
        # Tính min và max cho cột
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Áp dụng Min-Max Scaling
        df[col] = (df[col] - col_min) / (col_max - col_min)
    
    return df

# Hàm tóm tắt dữ liệu
def summarize_data(df):
    print(f"Dữ liệu có {df.shape[0]} dòng và {df.shape[1]} cột.")
    print("Thông tin mô tả dữ liệu:")
    print(df.describe(include='all'))
    print("\nCác giá trị null:")
    print(df.isnull().sum())

    # Trực quan hóa phân phối dữ liệu
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Ma trận tương quan")
    plt.show()

def fill_missing_values(df):
    for col in ['waterfront', 'view', 'yr_renovated']:
        if col in df.columns:
            mode_value = df[col].mode()
            if not mode_value.empty:  # Kiểm tra xem mode có rỗng không
                df[col] = df[col].fillna(mode_value[0])
            else:
                print(f"Cảnh báo: Cột {col} không có giá trị mode, không thể thay thế NaN.")
    return df

def main():
    # Đọc dữ liệu
    df = pd.read_excel("linear_data.xlsx")
    
    # Tóm tắt dữ liệu ban đầu
    print("===== Tóm tắt dữ liệu ban đầu =====")
    summarize_data(df)
    
    # Làm sạch và xử lý dữ liệu
    print("\n===== Làm sạch dữ liệu =====")
    continuous_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                          'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 
                          'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    
    # Xử lý ngoại lai
    df_cleaned = replace_outliers(df, continuous_columns)
    
    # Chuẩn hóa dữ liệu liên tục
    df_cleaned = min_max_scaling(df_cleaned, continuous_columns)
    
    # Thay thế NaN trong các cột categorical
    df_cleaned = fill_missing_values(df_cleaned)
    
    # Tóm tắt dữ liệu sau khi làm sạch
    print("\n===== Tóm tắt dữ liệu sau khi làm sạch =====")
    summarize_data(df_cleaned)
    
    # Lưu dữ liệu đã xử lý
    df_cleaned.to_excel("linear_processed_data.xlsx", index=False)
    print("\nDữ liệu đã xử lý được lưu thành 'linear_processed_data.xlsx'.")

if __name__ == "__main__":
    main()
