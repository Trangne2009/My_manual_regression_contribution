import pandas as pd
import numpy as np
import time

pd.set_option('future.no_silent_downcasting', True)
        
def read_excel(file_path): 
    df = pd.read_excel(file_path)  # Đọc dữ liệu từ file Excel
    return df

def var_blank_miss(df):
    df = df.astype(str)
    df.replace(r'^\s*\?\s*$', np.nan, regex=True, inplace=True) 
    
    numeric_columns = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15'] 
    
    # Chuyển đổi lại các cột số về kiểu số sau khi đã thay thế '?'
    for column in numeric_columns:
        if column in df.columns:
    # Chuyển các cột có thể chuyển đổi về kiểu số
            df[column] = pd.to_numeric(df[column], errors='coerce')

    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].mean())
    
    # Tính phương sai
    variance = df.select_dtypes(include=[np.number]).var()
    
    # Đếm ô trống
    blank_cells = (df == "").sum()
    
    # Đếm ô thiếu dữ liệu
    missing_data_per_column = df.isna().sum()
    
    # Kết quả
    print("Phương sai của các cột số:\n", variance)
    print("Số ô trống:\n", blank_cells)
    print("Số ô thiếu dữ liệu:\n", missing_data_per_column)

def remove_numbers_from_text(df):
    # Loại bỏ tất cả số trong các cột chữ
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].apply(lambda x: ''.join([i for i in str(x) if not i.isdigit()]) if isinstance(x, str) else x)
    return df

def remove_text_from_numbers(df):
    # Chuyển các giá trị không phải số thành NaN trong các cột số
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

#1. Làm sạch dữ liệu:
#a. Bỏ qua các dòng có dữ liệu thiếu 
def bo_qua_dl_thieu(df, col_indices=None):
    if col_indices is None:
        cols_chu = df.select_dtypes(include=['object']).columns  # Lấy các cột có dữ liệu chữ
    else:
        cols_chu = df.columns[col_indices]
    df = df.replace('?', np.nan)  # Chuyển '?' thành NaN
    df = df.dropna(subset=cols_chu)  # Bỏ các hàng có giá trị thiếu ở các cột chữ
    return df

#b. Điền thông tin thủ công
def dien_thu_cong(df, column_index, value="default_value"):
    # Thay '?' thành NaN trước khi điền
    df.replace('?', np.nan, inplace=True)
    df.iloc[:, column_index] = df.iloc[:, column_index].fillna(value)
    return df
 
#c. Điền tự động theo giá trị toàn cục
def dien_theo_gtri_toan_cuc(df, cols): 
    for col in cols:
        most_common_value = df.iloc[:, col].mode()[0]  # Lấy giá trị phổ biến nhất
        df.iloc[:, col] = df.iloc[:, col].fillna(most_common_value)
    return df

#d. Điền giá trị thiếu bằng trung bình
def dien_gtri_trung_binh(df, column_index): 
    mean_value = df.iloc[:, column_index].mean()
    df.iloc[:, column_index] = df.iloc[:, column_index].fillna(mean_value)
    return df

#e. Điền giá trị thiếu bằng trung bình thuộc tính cho tất cả các mẫu cùng chung một lớp
def trung_binh_thuoc_tinh(df, class_column, target_columns):
    for col in target_columns:
        # Chuyển đổi các cột sang kiểu số và thay thế giá trị không hợp lệ bằng NaN
        df[df.columns[target_columns]] = df[df.columns[target_columns]].apply(pd.to_numeric, errors='coerce')

        df.iloc[:, col] = df.groupby(df.columns[class_column])[df.columns[col]].transform(lambda x: x.fillna(x.mean()))
    return df 

#2. Bớt Dữ Liệu Nhiễu 
def bo_gtri_bien(df, threshold=3, column_names=None):
    if column_names is None:
        column_names = df.select_dtypes(include=[np.number]).columns
    for col in column_names:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        df = df[np.abs(z_scores) < threshold]
    return df

#3. Chuẩn Hóa Dữ Liệu
#a. Chuẩn hóa Min-Max
def min_max(df, cols=None): 
    if cols is None:
        cols_so = df.select_dtypes(include=[np.number]).columns  # Lấy các cột có kiểu số
    else:
        cols_so = df.columns[cols]
    
    for col in cols_so:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)  # Chuẩn hóa Min-Max
    return df 

#b. Chuẩn hóa Z-Score
def z_score(df, cols=None): 
    if cols is None:
        cols_so = df.select_dtypes(include=[np.number]).columns
    else:
        cols_so = df.columns[cols]
    
    for col in cols_so:
        mean_val = df[col].mean()
        std_dev = df[col].std()
        df[col] = (df[col] - mean_val) / std_dev
    return df

#c. Chuẩn hóa tỉ lệ
def chuan_hoa_ti_le(df, cols=None, scale=100): 
    if cols is None:
        cols_so = df.select_dtypes(include=[np.number]).columns
    else:
        cols_so = df.columns[cols]
    
    for col in cols_so:
        df[col] = df[col] * scale
    return df 
            
def main(): 
    start_time = time.time() 
    
    file_path = "D:\\Đồ án chuyên ngành\\Nhóm 07\\Python\\Ptích dl Chapter11-1a.xlsx"
    
    df = read_excel(file_path) 
    
    var_blank_miss(df)
    
    df = remove_numbers_from_text(df)
    
    df = remove_text_from_numbers(df)
    
    print("Bỏ qua các dòng thiếu dữ liệu...")
    df = bo_qua_dl_thieu(df)

    print("Điền thông tin thủ công...")
    df = dien_thu_cong(df, column_index=10, value="0")

    print("Điền giá trị theo giá trị toàn cục cho cột 2...")
    df = dien_theo_gtri_toan_cuc(df, cols=[1, 2])
    
    print("Điền giá trị thiếu theo trung bình lớp cho cột 3...")
    df = trung_binh_thuoc_tinh(df, class_column=0, target_columns=[13, 14]) 
    
    df= dien_gtri_trung_binh(df, column_index=7)
    
    # Chuẩn hóa Min-Max
    df = min_max(df) 
    
    # Chuẩn hóa Z-Score
    df= z_score(df) 
    
    print("Bỏ các giá trị biên...")
    df = bo_gtri_bien(df) 
    
    # Chuẩn hóa tỷ lệ cho cột 4 với tỷ lệ 2
    df = chuan_hoa_ti_le(df, scale=100) 
    
    # Lưu dữ liệu đã xử lý vào file mới
    df.to_excel("processed_data.xlsx", index=False)
    print("Đã lưu file processed_data.xlsx")
    
    print("Tổng thời gian chạy:", time.time() - start_time, "giây")
    
if __name__ == "__main__": 
    main()