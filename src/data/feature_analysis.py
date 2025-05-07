import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

def scale_features(X, one_hot_prefixes=None, scaler=None):
    """
    Chuẩn hóa dữ liệu: scale các cột numeric, giữ nguyên one-hot.
    Trả về DataFrame đã chuẩn hóa và các nhóm cột.
    """
    if one_hot_prefixes is None:
        one_hot_prefixes = []

    one_hot_cols = [
        col for col in X.columns if any(col.startswith(prefix) for prefix in one_hot_prefixes)
    ]
    binary_cols = [
        col for col in X.columns
        if X[col].nunique() == 2 and set(X[col].unique()).issubset({0, 1}) and col not in one_hot_cols
    ]
    numeric_cols = [col for col in X.columns if col not in one_hot_cols + binary_cols]

    if scaler is None:
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X[numeric_cols])
    else:
        X_numeric_scaled = scaler.transform(X[numeric_cols])

    X_scaled = pd.concat([
        pd.DataFrame(X_numeric_scaled, columns=numeric_cols, index=X.index),
        X[one_hot_cols + binary_cols]
    ], axis=1)

    return X_scaled, scaler

def get_rf_importance(X, y):
    """Đánh giá feature importance bằng Random Forest."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    return importance

def get_log_importance(X, y):
    """Đánh giá feature importance bằng Logistic Regression."""
    X_scaled, _ = scale_features(X)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_scaled, y)
    importance = pd.DataFrame({
        'Feature': X_scaled.columns,
        'Importance': np.abs(log_reg.coef_[0])
    }).sort_values(by='Importance', ascending=False)
    return importance

def get_svc_selection(X, y):
    """Đánh giá feature importance bằng LinearSVC."""
    X_scaled, _ = scale_features(X)

    svc = LinearSVC(max_iter=1000, random_state=42)
    svc.fit(X_scaled, y)
    # Kiểm tra số lớp và xử lý coef_
    if len(np.unique(y)) > 2:  # Nếu là bài toán đa lớp
        print("Cảnh báo: LinearSVC phát hiện bài toán đa lớp, lấy trung bình coef_")
        coef = np.abs(svc.coef_).mean(axis=0)  # Trung bình các hệ số theo lớp
    else:
        coef = np.abs(svc.coef_.ravel())
    importance = pd.DataFrame({
        'Feature': X_scaled.columns,
        'Importance': coef
    }).sort_values(by='Importance', ascending=False)
    return importance

def normalize_importance(importance_df):
    """
    Chuẩn hóa cột Importance về thang 0-1.

    Parameters:
    - importance_df (pd.DataFrame): DataFrame chứa cột 'Feature' và 'Importance'.

    Returns:
    - pd.DataFrame: DataFrame với cột 'Importance' đã chuẩn hóa.
    """
    importance_df = importance_df.copy()
    max_importance = importance_df['Importance'].max()
    min_importance = importance_df['Importance'].min()
    if max_importance == min_importance:
        importance_df['Importance'] = 1.0
    else:
        importance_df['Importance'] = (importance_df['Importance'] - min_importance) / (max_importance - min_importance)
    return importance_df

def aggregate_feature_importance(X, y):
    """
    Tổng hợp feature importance từ 3 mô hình: Random Forest, Logistic Regression, LinearSVC.

    Parameters:
    - X (pd.DataFrame): Dữ liệu đầu vào (features).
    - y (pd.Series): Biến mục tiêu.

    Returns:
    - pd.DataFrame: DataFrame chứa feature và điểm importance trung bình (đã chuẩn hóa).
    """
    rf_importance = normalize_importance(get_rf_importance(X, y))
    log_importance = normalize_importance(get_log_importance(X, y))
    svc_importance = normalize_importance(get_svc_selection(X, y))

    # Gộp dữ liệu
    combined_importance = pd.DataFrame({
        'Feature': X.columns,
        'RF_Importance': rf_importance.set_index('Feature')['Importance'],
        'Log_Importance': log_importance.set_index('Feature')['Importance'],
        'SVC_Importance': svc_importance.set_index('Feature')['Importance']
    })

    # Tính điểm trung bình
    combined_importance['Average_Importance'] = combined_importance[
        ['RF_Importance', 'Log_Importance', 'SVC_Importance']].mean(axis=1)
    combined_importance = combined_importance.sort_values(by='Average_Importance', ascending=False)
    return combined_importance

def prepare_features(df, target_column='build_failed'):
    """Chuẩn bị dữ liệu cho phân tích feature importance.
    Cảnh báo nếu có cột không phải kiểu số."""
    numeric_types = ['int64', 'float64', 'int32', 'float32']

    excluded_columns = [target_column, "gh_build_started_at", "gh_project_name"]
    non_numeric_cols = [
        col for col in df.columns
        if df[col].dtype.name not in numeric_types and col not in excluded_columns
    ]
    if non_numeric_cols:
        print(f"[WARNING] The following columns are not numeric and will be discarded: {non_numeric_cols}")
    # Chuẩn bị dữ liệu đầu vào và nhãn
    X = df.select_dtypes(include=numeric_types).drop(columns=excluded_columns, errors='ignore')
    y = df[target_column]
    return X, y

def print_nan_columns(dfs=None, selected_projects=None, df=None):
    """
    In ra danh sách các cột có giá trị NaN trong DataFrame.

    Parameters:
    - dfs (dict, optional): Từ điển chứa các DataFrame, với key là tên dự án.
    - selected_projects (list, optional): Danh sách các dự án được chọn để gộp.
    - df (pd.DataFrame, optional): DataFrame đầu vào trực tiếp (nếu không dùng dfs).

    Returns:
    - None: Chỉ in ra tên các cột có NaN.
    """
    # Kiểm tra đầu vào
    if df is None and (dfs is None or selected_projects is None):
        raise ValueError("Phải cung cấp hoặc 'df' hoặc cả 'dfs' và 'selected_projects'.")

    # Nếu dùng dfs và selected_projects, gộp thành DataFrame
    if df is None:
        selected_dfs = [dfs[project] for project in selected_projects if project in dfs]
        if not selected_dfs:
            print("Không có dự án nào được chọn hoặc tồn tại trong dfs.")
            return
        df = pd.concat(selected_dfs, ignore_index=True)

    # Tìm các cột có NaN
    nan_columns = df.columns[df.isna().any()].tolist()

    # In danh sách cột có NaN
    if nan_columns:
        print("Các cột có giá trị NaN:")
        for col in nan_columns:
            print(col)
    else:
        print("Không có cột nào chứa giá trị NaN.")