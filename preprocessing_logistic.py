import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

df = pd.read_csv('logistic_data.csv')

# Hàm xử lý ngoại lai bằng cách thay thế bằng min_value và max_value
def replace_outliers(df, columns):
    for column in columns:
        if column not in df.columns:
            print(f"Warning: Cột {column} không tồn tại trong DataFrame")
            continue  # Skip this column if it does not exist
        
        # Chuyển đổi dữ liệu
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Xử lý NaN
        df[column].fillna(df[column].mean(), inplace=True)
        
        # Sắp xếp dữ liệu
        df = df.sort_values(by=column)
        
        # Tính IQR và thay thế ngoại lai
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Áp dụng việc thay thế ngoại lai
        df[column] = df[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    
    return df

# 1. Tóm tắt mô tả về dữ liệu
def summarize_data(df):
    print(f"Dữ liệu có {df.shape[0]} dòng và {df.shape[1]} cột.")
    summary = df.describe(include='all')
    print("Thông tin mô tả dữ liệu:")
    print(summary)
    
    # Thiết lập kích thước cho các biểu đồ
    plt.figure(figsize=(15, 25))

    # Vẽ Boxplot cho các dữ liệu liên tục và rời rạc
    boxplot_group1 = ["person_income", "loan_amnt", "credit_score"]
    boxplot_group2 = ["loan_int_rate", "person_age", "person_emp_exp", "cb_person_cred_hist_length"]
    boxplot_columns = boxplot_group1 + boxplot_group2

    plt.subplot(4, 1, 1)
    sns.boxplot(data=df[["person_income"]], orient="h", color="red")
    plt.title("Boxplot cho thuộc tính: person_income")
    plt.xlabel("Giá trị")
    plt.ylabel("person_income")
    
    plt.subplot(4, 1, 2)
    sns.boxplot(data=df[["loan_amnt"]], orient="h", color="blue")
    plt.title("Boxplot cho thuộc tính: loan_amnt")
    plt.xlabel("Giá trị")
    plt.ylabel("loan_amnt")
    
    plt.subplot(4, 1, 3)
    sns.boxplot(data=df[["credit_score"]], orient="h", color="yellow")
    plt.title("Boxplot cho thuộc tính: credit_score")
    plt.xlabel("Giá trị")
    plt.ylabel("credit_score")
    
    plt.subplot(4, 1, 4)
    sns.boxplot(data=df[boxplot_group2], orient="h", palette="Pastel1")
    plt.title("Boxplot cho các thuộc tính còn lại")
    plt.xlabel("Giá trị")
    plt.ylabel("Thuộc tính")
    
        
    # Tính toán các thông số cho Boxplot
    for column in boxplot_columns:
        stats = df[column].describe()
        q1 = stats['25%']
        median = stats['50%']
        q3 = stats['75%']
        iqr = q3 - q1
        lower_whisker = max(df[column].min(), q1 - 1.5 * iqr)
        upper_whisker = min(df[column].max(), q3 + 1.5 * iqr)
        
        print(f"{column}:")
        print(f"  Min (Không ngoại lai): {lower_whisker}")
        print(f"  Q1 (Phân vị 25%): {q1}")
        print(f"  Median (Trung vị): {median}")
        print(f"  Q3 (Phân vị 75%): {q3}")
        print(f"  Max (Không ngoại lai): {upper_whisker}")
        print(f"  Outliers (Số lượng): {(df[column] < lower_whisker).sum() + (df[column] > upper_whisker).sum()}")
        print()

    # Vẽ Bar Chart cho dữ liệu phân loại
    categorical_columns = [
        "person_gender", "person_education", "person_home_ownership",
        "loan_intent", "previous_loan_defaults_on_file", "loan_status"
    ]

    plt.figure(figsize=(20, 15))
    for i, column in enumerate(categorical_columns, 1):
        plt.subplot(3, 2, i)
        sns.countplot(x=column, data=df, hue=column, palette="Set3", legend=False)
        plt.title(f"Bar Chart của {column}")
        plt.xlabel(column)
        plt.ylabel("Số lượng")
        plt.xticks(rotation=45)
        #print(df[categorical_columns].value_counts())
    
    for column in categorical_columns:
        counts = df[column].value_counts()
        print(f"Thông số số lượng cho cột '{column}':")
        for value, count in counts.items():
            print(f"{value}: {count}")
        print()  
        
    plt.tight_layout()
    plt.show()
    
# Xử lý ngoại lai cho các cột liên quan
def handle_outliers(df):
    continuous_columns = ["person_income", "loan_amnt", "credit_score", 
                          "loan_int_rate", "person_age", "person_emp_exp", 
                          "cb_person_cred_hist_length"]
    for column in continuous_columns:
        df = replace_outliers(df, column)
  
    return df

# Xử lý dữ liệu phân loại (Label Encoding)
def encode_data(df):
    label_mappings = {}
    categorical_columns = ['person_gender', 'person_home_ownership', 'person_education', 
                           'loan_intent', 'previous_loan_defaults_on_file']
    
    for col in categorical_columns:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            label_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    
    return df, label_mappings 

# Điều chỉnh và gộp nhóm phân loại
def adjust_data(df):
    # Gộp nhóm 'person_education' thành 2 nhóm: 'Master/Doctorate' và các nhóm còn lại
    df['person_education'] = df['person_education'].replace({
        'Master': 'Master/Doctorate',
        'Doctorate': 'Master/Doctorate'
    })
    
    # Gộp nhóm 'person_home_ownership' thành nhóm 'Mortgage/Own/Other' và nhóm còn lại là 'Rent'
    df['person_home_ownership'] = df['person_home_ownership'].replace({
        'OWN': 'Mortgage/Own/Other',
        'MORTGAGE': 'Mortgage/Own/Other',
        'OTHER': 'Mortgage/Own/Other'
    })
    
    # Tạo từ điển ánh xạ các giá trị cũ thành nhóm mới
    loan_intent_mapping = {
        'EDUCATION': 'Mục đích cá nhân',  
        'MEDICAL': 'Mục đích cá nhân',  
        'VENTURE': 'Mục đích cá nhân',  
        'PERSONAL': 'Mục đích cá nhân', 
        'DEBTCONSOLIDATION': 'Tài chính và cải thiện tài sản',  
        'HOMEIMPROVEMENT': 'Tài chính và cải thiện tài sản'   
    }
    
    # Kiểm tra phân bố trước khi ánh xạ
    print("Phân bố cột 'loan_intent' trước khi ánh xạ:")
    print(df['loan_intent'].value_counts())
    
    # Áp dụng ánh xạ vào cột 'loan_intent'
    df['loan_intent'] = df['loan_intent'].map(loan_intent_mapping)
    
    # Kiểm tra phân bố sau khi ánh xạ
    print("\nPhân bố cột 'loan_intent' sau khi ánh xạ:")
    print(df['loan_intent'].value_counts())
    
    # Nếu có giá trị NaN trong cột 'loan_intent' sau khi ánh xạ, thay thế bằng 'Khác'
    df['loan_intent'] = df['loan_intent'].fillna('Khác')
    
    return df

def downsample_classes(df, target_column):
    # Phân tách các lớp
    classes = df[target_column].unique()
    class_dfs = {cls: df[df[target_column] == cls] for cls in classes}
    
    # Xác định lớp nhỏ nhất
    min_count = min(len(class_dfs[cls]) for cls in classes)
    print(f"Lớp nhỏ nhất có {min_count} dòng.")
    
    # Lấy mẫu ngẫu nhiên từ các lớp lớn hơn để cân bằng với lớp nhỏ nhất
    downsampled_dfs = [
        resample(class_dfs[cls], 
                 replace=False, 
                 n_samples=min_count, 
                 random_state=42) 
        for cls in classes
    ]
    
    # Kết hợp lại thành DataFrame mới
    balanced_df = pd.concat(downsampled_dfs)
    print(f"Dữ liệu sau khi cân bằng có {balanced_df.shape[0]} dòng.")
    
    return balanced_df

# 2. Làm sạch dữ liệu
def clean_data(df):
    df = handle_outliers(df)
    df = adjust_data(df)
    df, label_mappings = encode_data(df)
    
    return df, label_mappings
    
def z_score(df, columns):
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
    return df

def main():
    # Quy trình xử lý dữ liệu
    print("===== Tóm tắt dữ liệu =====")
    summarize_data(df)

    # Làm sạch và xử lý dữ liệu
    print("\n===== Làm sạch và xử lý dữ liệu =====")
    df_processed, label_mappings = clean_data(df)
    
    print("\nÁnh xạ Label Encoding ban đầu: ")
    for col, mapping in label_mappings.items():
        print(f"{col}: {mapping}")
        
    # Chuẩn hóa Z-score
    continuous_columns = ["person_income", "loan_amnt", "credit_score", 
                          "loan_int_rate", "person_age", "person_emp_exp", 
                          "cb_person_cred_hist_length"]
    df_normalized = z_score(df_processed, continuous_columns)
    
    # Cân bằng lớp
    print("\n===== Cân bằng lớp loan_status =====")
    balanced_df = downsample_classes(df_normalized, 'loan_status')
    print("\nSố lượng mẫu mỗi lớp sau cân bằng: ")
    print(balanced_df['loan_status'].value_counts())
    
    # Kiểm tra và xử lý ngoại lai sau khi cân bằng lớp
    print("\n===== Kiểm tra và xử lý ngoại lai cho credit_score =====")
    balanced_df = replace_outliers(balanced_df, continuous_columns)
    
    # Tóm tắt dữ liệu sau cân bằng lớp và xử lý ngoại lai
    print("\n===== Tóm tắt dữ liệu sau xử lý cân bằng lớp và ngoại lai =====")
    summarize_data(balanced_df)
    
    #Lưu lại dữ liệu đã xử lý
    balanced_df.to_excel('processed_logistic.xlsx', index=False)
    
    return balanced_df

# Chạy hàm main
df_final = main()