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
        
# Dự đoán trên dữ liệu mới (cả 2 mô hình)
    load_new_data(file_path, new_data_sheet_name, scaler)
    predict_new_data(file_path, scaler, model=model_sklearn)  # Dự đoán bằng Sklearn
    # Mô hình tự viết:
    predict_new_data(file_path, scaler, m=m, b=b)