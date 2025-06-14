# src/data/processing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import glob
from collections import defaultdict

# Định nghĩa dtype cho các cột đặc biệt
DTYPE_SPEC = {
    "git_diff_src_churn": "float32",
    "gh_diff_files_modified": "float32",
    "gh_test_lines_per_kloc": "float32"
}

# Danh sách 25 cột mục tiêu
TARGET_COLUMNS = [
    "tr_status", "gh_project_name", "gh_is_pr", "gh_team_size", "git_num_commits",
    "gh_num_issue_comments", "gh_num_commit_comments", "gh_num_pr_comments",
    "git_diff_src_churn", "git_diff_test_churn", "gh_diff_files_added",
    "gh_diff_files_deleted", "gh_diff_files_modified", "gh_diff_tests_added",
    "gh_diff_tests_deleted", "gh_diff_src_files", "gh_diff_doc_files",
    "gh_diff_other_files", "gh_num_commits_on_files_touched", "gh_sloc",
    "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc",
    "gh_by_core_team_member", "gh_build_started_at",
    "tr_duration", "tr_build_id", "tr_job_id"
]

# Ánh xạ tên cột cho từng nhóm
MAPPINGS = {
    "Group1": {
        "tr_status": "tr_status", "gh_project_name": "gh_project_name", "gh_is_pr": "gh_is_pr",
        "gh_team_size": "gh_team_size", "git_num_commits": "gh_num_commits_in_push",
        "gh_num_issue_comments": "gh_num_issue_comments", "gh_num_commit_comments": "gh_num_commit_comments",
        "gh_num_pr_comments": "gh_num_pr_comments", "git_diff_src_churn": "git_diff_src_churn",
        "git_diff_test_churn": "git_diff_test_churn", "gh_diff_files_added": "gh_diff_files_added",
        "gh_diff_files_deleted": "gh_diff_files_deleted", "gh_diff_files_modified": "gh_diff_files_modified",
        "gh_diff_tests_added": "gh_diff_tests_added", "gh_diff_tests_deleted": "gh_diff_tests_deleted",
        "gh_diff_src_files": "gh_diff_src_files", "gh_diff_doc_files": "gh_diff_doc_files",
        "gh_diff_other_files": "gh_diff_other_files", "gh_num_commits_on_files_touched": "gh_num_commits_on_files_touched",
        "gh_sloc": "gh_sloc", "gh_test_lines_per_kloc": "gh_test_lines_per_kloc",
        "gh_test_cases_per_kloc": "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc": "gh_asserts_cases_per_kloc",
        "gh_by_core_team_member": "gh_by_core_team_member", "gh_build_started_at": "gh_build_started_at",
        "tr_duration": "tr_duration",
        "tr_build_id": "tr_build_id", "tr_job_id": "tr_job_id",
    },
    "Group2": {
        "tr_status": "status", "gh_project_name": "project_name", "gh_is_pr": "is_pr",
        "gh_team_size": "team_size", "git_num_commits": "num_commits",
        "gh_num_issue_comments": "num_issue_comments", "gh_num_commit_comments": "num_commit_comments",
        "gh_num_pr_comments": "num_pr_comments", "git_diff_src_churn": "src_churn",
        "git_diff_test_churn": "test_churn", "gh_diff_files_added": "files_added",
        "gh_diff_files_deleted": "files_deleted", "gh_diff_files_modified": "files_modified",
        "gh_diff_tests_added": "tests_added", "gh_diff_tests_deleted": "tests_deleted",
        "gh_diff_src_files": "src_files", "gh_diff_doc_files": "doc_files",
        "gh_diff_other_files": "other_files", "gh_num_commits_on_files_touched": "commits_on_files_touched",
        "gh_sloc": "sloc", "gh_test_lines_per_kloc": "test_lines_per_kloc",
        "gh_test_cases_per_kloc": "test_cases_per_kloc", "gh_asserts_cases_per_kloc": "asserts_per_kloc",
        "gh_by_core_team_member": "main_team_member", "gh_build_started_at": "started_at",
        "tr_duration": "duration",
        "tr_build_id": "build_id", "tr_job_id": "job_id",
    },
    "Group3": {
        "tr_status": "tr_status", "gh_project_name": "gh_project_name", "gh_is_pr": "gh_is_pr",
        "gh_team_size": "gh_team_size", "git_num_commits": "git_num_commits",
        "gh_num_issue_comments": "gh_num_issue_comments", "gh_num_commit_comments": "gh_num_commit_comments",
        "gh_num_pr_comments": "gh_num_pr_comments", "git_diff_src_churn": "gh_src_churn",
        "git_diff_test_churn": "gh_test_churn", "gh_diff_files_added": "gh_files_added",
        "gh_diff_files_deleted": "gh_files_deleted", "gh_diff_files_modified": "gh_files_modified",
        "gh_diff_tests_added": "gh_tests_added", "gh_diff_tests_deleted": "gh_tests_deleted",
        "gh_diff_src_files": "gh_src_files", "gh_diff_doc_files": "gh_doc_files",
        "gh_diff_other_files": "gh_other_files", "gh_num_commits_on_files_touched": "gh_commits_on_files_touched",
        "gh_sloc": "gh_sloc", "gh_test_lines_per_kloc": "gh_test_lines_per_kloc",
        "gh_test_cases_per_kloc": "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc": "gh_asserts_cases_per_kloc",
        "gh_by_core_team_member": "gh_by_core_team_member", "gh_build_started_at": "tr_started_at",
        "tr_duration": "tr_duration",
        "tr_build_id": "tr_build_id", "tr_job_id": "tr_job_id",
    },
    "Group4": {
        "tr_status": "tr_status", "gh_project_name": "gh_project_name", "gh_is_pr": "gh_is_pr",
        "gh_team_size": "gh_team_size", "git_num_commits": "gh_num_commits_in_push",
        "gh_num_issue_comments": "gh_num_issue_comments", "gh_num_commit_comments": "gh_num_commit_comments",
        "gh_num_pr_comments": "gh_num_pr_comments", "git_diff_src_churn": "git_diff_src_churn",
        "git_diff_test_churn": "git_diff_test_churn", "gh_diff_files_added": "gh_diff_files_added",
        "gh_diff_files_deleted": "gh_diff_files_deleted", "gh_diff_files_modified": "gh_diff_files_modified",
        "gh_diff_tests_added": "gh_diff_tests_added", "gh_diff_tests_deleted": "gh_diff_tests_deleted",
        "gh_diff_src_files": "gh_diff_src_files", "gh_diff_doc_files": "gh_diff_doc_files",
        "gh_diff_other_files": "gh_diff_other_files", "gh_num_commits_on_files_touched": "gh_num_commits_on_files_touched",
        "gh_sloc": "gh_sloc", "gh_test_lines_per_kloc": "gh_test_lines_per_kloc",
        "gh_test_cases_per_kloc": "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc": "gh_asserts_cases_per_kloc",
        "gh_by_core_team_member": "gh_by_core_team_member", "gh_build_started_at": "gh_build_started_at",
        "tr_duration": "tr_duration",
        "tr_build_id": "tr_build_id", "tr_job_id": "tr_job_id",
    }
}

def group_files_by_columns(folder_path):
    """Phân loại file CSV theo cấu trúc cột."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    column_groups = defaultdict(list)
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, nrows=0)
            columns = tuple(df.columns.tolist())
            column_groups[columns].append(file_path)
        except Exception as e:
            print(f"Lỗi đọc {file_path}: {e}")
    return list(column_groups.items())

def process_files(file_list, group_name, mapping):
    """Đọc và xử lý các file trong danh sách theo nhóm."""
    dfs = []
    for file in file_list:
        try:
            df = pd.read_csv(file, dtype=DTYPE_SPEC, low_memory=False)
            df = df.rename(columns={v: k for k, v in mapping[group_name].items()})
            available_columns = [col for col in TARGET_COLUMNS if col in df.columns]
            if available_columns:
                df = df[available_columns]
                dfs.append(df)
            else:
                print(f"Warning: No target columns found in {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    return dfs

def combine_datasets(folder_path, output_path="../data/combined/combined_travistorrent.csv"):
    """Kết hợp tất cả file CSV trong thư mục thành một file duy nhất."""
    groups_files = group_files_by_columns(folder_path)
    group_dfs = []
    for i, (columns, files) in enumerate(groups_files, start=1):
        group_name = f"Group{i}"
        dfs = process_files(files, group_name, MAPPINGS)
        group_dfs.append(dfs)
        print(f"Processed {len(dfs)} DataFrames for {group_name}")

    all_dfs = []
    for dfs in group_dfs:
        all_dfs.extend(dfs)

    all_dfs = [
        df.dropna(how='all', axis=1)
        for df in all_dfs
        if not df.empty and not df.isna().all().all()
    ]

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df[[col for col in TARGET_COLUMNS if col in combined_df.columns]]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        print("Kích thước DataFrame cuối cùng:", combined_df.shape)
        print("Các cột trong DataFrame:", combined_df.columns.tolist())
        return combined_df
    else:
        print("Không có DataFrame nào được xử lý thành công.")
        return None

def add_build_features(df):
    """Thêm các feature mới liên quan đến build."""
    df = df.copy()

    # Sắp xếp theo thời gian để tính các feature liên quan đến lịch sử
    df = df.sort_values(['gh_project_name', 'gh_build_started_at'])
    df['gh_build_started_at'] = pd.to_datetime(df['gh_build_started_at'])

    # 1. year_of_start, month_of_start, day_of_start, hour_of_start
    df['year_of_start'] = df['gh_build_started_at'].dt.year
    df['month_of_start'] = df['gh_build_started_at'].dt.month
    df['day_of_start'] = df['gh_build_started_at'].dt.day
    df['hour_of_start'] = df['gh_build_started_at'].dt.hour

    # 2. elapsed_days_last_build: Số ngày kể từ build trước
    df['elapsed_days_last_build'] = df.groupby('gh_project_name')['gh_build_started_at'].diff().dt.total_seconds() / (
                24 * 3600)
    df['elapsed_days_last_build'] = df['elapsed_days_last_build'].fillna(0)

    # 3. same_committer: Committer có giống build trước không (dùng gh_by_core_team_member)
    df['prev_committer'] = df.groupby('gh_project_name')['gh_by_core_team_member'].shift(1)
    df['same_committer'] = (df['gh_by_core_team_member'] == df['prev_committer']).astype(int)
    df['same_committer'] = df['same_committer'].fillna(0)  # Điền 0 cho build đầu tiên

    # 4. proj_fail_rate_history: Tỷ lệ thất bại lịch sử của dự án
    df['proj_fail_rate_history'] = df.groupby('gh_project_name')['build_failed'].expanding().mean().reset_index(level=0,
                                                                                                                drop=True)

    # 5. proj_fail_rate_recent: Tỷ lệ thất bại gần đây (10 build trước)
    df['proj_fail_rate_recent'] = df.groupby('gh_project_name')['build_failed'].rolling(window=10,
                                                                                        min_periods=1).mean().reset_index(
        level=0, drop=True)

    # 6. comm_fail_rate_history: Tỷ lệ thất bại lịch sử của committer
    df['comm_fail_rate_history'] = df.groupby('gh_by_core_team_member')['build_failed'].expanding().mean().reset_index(
        level=0, drop=True)

    # 7. comm_fail_rate_recent: Tỷ lệ thất bại gần đây của committer (10 build trước)
    df['comm_fail_rate_recent'] = df.groupby('gh_by_core_team_member')['build_failed'].rolling(window=10,
                                                                                               min_periods=1).mean().reset_index(
        level=0, drop=True)

    # 8. comm_avg_experience: Số build trung bình của committer (dùng số lần xuất hiện)
    comm_experience = df.groupby('gh_by_core_team_member').cumcount() + 1
    df['comm_avg_experience'] = comm_experience

    # 10. no_config_edited: Có chỉnh sửa config file không (giả định dựa trên gh_diff_doc_files)
    df['no_config_edited'] = (df['gh_diff_doc_files'] > 0).astype(int)

    # 11. num_files_edited: Tổng số file chỉnh sửa
    df['num_files_edited'] = df[['gh_diff_files_added', 'gh_diff_files_deleted', 'gh_diff_files_modified']].sum(axis=1)

    # 12. num_distinct_authors: Số tác giả khác nhau (giả định 1 committer/build, dùng groupby cumcount)
    if 'tr_build_id' in df.columns:
        df['num_distinct_authors'] = df.groupby('tr_build_id')['gh_by_core_team_member'].transform('nunique')
    else:
        df['num_distinct_authors'] = df['gh_by_core_team_member'].notna().astype(int)

    # 13. prev_build_result: Kết quả build trước
    df['prev_build_result'] = df.groupby('gh_project_name')['build_failed'].shift(1).fillna(0)

    # 14. day_week: Ngày trong tuần
    df['day_week'] = df['gh_build_started_at'].dt.dayofweek

    return df

def load_data(file_path):
    """Tải dữ liệu từ CSV với dtype cụ thể."""
    df = pd.read_csv(file_path, dtype=DTYPE_SPEC, low_memory=False)
    # df['tr_log_num_tests_failed'] = pd.to_numeric(df['tr_log_num_tests_failed'], errors='coerce').astype('float32')
    return df


def process_status(df: pd.DataFrame, status_column: str) -> pd.DataFrame:
    """
    Chuẩn hóa mọi cột trạng thái về build_failed (0/1) và loại bỏ giá trị lạ.

    - Mọi giá trị 0/1 (nếu đã numeric) hoặc 'passed'/'failed'/'errored'
      đều được map về 0/1.
    - Các giá trị khác thành NaN rồi bị drop.
    """
    df = df.copy()
    df = df[df[status_column].isin(['failed', 'errored', 'passed'])]
    m = {0: 0, 1: 1, 'passed': 0, 'failed': 1, 'errored': 1}
    df['build_failed'] = df[status_column].map(m)
    df = df.dropna(subset=['build_failed'])
    df['build_failed'] = df['build_failed'].astype(int)
    if status_column not in ['build_failed']:
        df = df.drop(columns=[status_column])
    return df

def summarize_projects(df: pd.DataFrame,
                       project_column: str = 'gh_project_name',
                       min_rows: int = 10000,
                       balance_threshold: float = 0.6) -> pd.DataFrame:
    """
    Giả sử df đã có cột 'build_failed' (0/1, không NaN).
    Tóm tắt per-project: số dòng, missing_ratio, passed/failed ratio.
    """
    rows = []
    for proj, g in df.groupby(project_column):
        n = len(g)
        if n < min_rows:
            continue
        miss = g.isna().mean().mean()
        vals = g['build_failed'].dropna().astype(int)
        if vals.empty:
            continue
        d = vals.value_counts(normalize=True)
        p, f = d.get(0, 0.0), d.get(1, 0.0)
        if max(p, f) > balance_threshold:
            continue
        rows.append((proj, n, round(miss, 4), round(p, 4), round(f, 4)))

    return (pd.DataFrame(rows,
             columns=['project', 'num_rows', 'missing_ratio', 'passed_ratio', 'failed_ratio'])
            .sort_values('num_rows', ascending=False))

import pandas as pd
import numpy as np

def fill_nan_values(df):
    """Điền giá trị NaN theo logic phù hợp với ý nghĩa của từng feature."""
    df = df.copy()
    groupby_project = df.groupby('gh_project_name')

    if 'tr_duration' in df.columns:
        df['tr_duration'] = groupby_project['tr_duration'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
                       .ffill()
                       .bfill()
                       .fillna(x.mean() if not pd.isna(x.mean()) else 0)
        )

    integer_cols = [
        'git_num_commits', 'gh_num_issue_comments', 'gh_num_pr_comments', 'gh_num_commit_comments',
        'git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_added', 'gh_diff_files_deleted',
        'gh_diff_files_modified', 'gh_diff_src_files', 'gh_diff_doc_files', 'gh_diff_other_files',
        'gh_diff_tests_added', 'gh_diff_tests_deleted', 'gh_num_commits_on_files_touched'
    ]
    for col in integer_cols:
        if col in df.columns:
            df[col] = groupby_project[col].transform(lambda x: x.fillna(0))

    density_cols = [
        'gh_test_lines_per_kloc', 'gh_test_cases_per_kloc', 'gh_asserts_cases_per_kloc'
    ]
    for col in density_cols:
        if col in df.columns:
            df[col] = groupby_project[col].transform(
                lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 0)
            )

    if 'gh_by_core_team_member' in df.columns:
        df['gh_by_core_team_member'] = groupby_project['gh_by_core_team_member'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0)
        )

    if 'gh_team_size' in df.columns:
        df['gh_team_size'] = groupby_project['gh_team_size'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 1)
        )

    if 'gh_sloc' in df.columns:
        df['gh_sloc'] = groupby_project['gh_sloc'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
        )
    return df
def encode_categorical_columns(df, columns, project_col='gh_project_name'):
    """Mã hóa cột phân loại theo từng project."""
    df_encoded_list = []
    encoders_by_project = {}

    for project, group in df.groupby(project_col):
        group_encoded = group.copy()
        encoders = {}
        for col in columns:
            le = LabelEncoder()
            group_encoded[col] = le.fit_transform(group_encoded[col].astype(str))
            encoders[col] = le
        encoders_by_project[project] = encoders
        df_encoded_list.append(group_encoded)

    df_encoded = pd.concat(df_encoded_list, ignore_index=True)
    return df_encoded, encoders_by_project


def normalize_numerical_columns(df, columns, project_col='gh_project_name'):
    """Chuẩn hóa cột số theo từng project."""
    df_normalized_list = []
    scalers_by_project = {}

    for project, group in df.groupby(project_col):
        group_normalized = group.copy()
        scalers = {}
        for col in columns:
            if col in group.columns:
                scaler = MinMaxScaler()
                group_normalized[col] = scaler.fit_transform(group[[col]])
                scalers[col] = scaler
        scalers_by_project[project] = scalers
        df_normalized_list.append(group_normalized)

    df_normalized = pd.concat(df_normalized_list, ignore_index=True)
    return df_normalized, scalers_by_project


def encode_cyclical_time_features(df, columns, periods, project_col='gh_project_name'):
    """Mã hóa các cột thời gian tuần hoàn theo từng project."""
    df_encoded_list = []

    for project, group in df.groupby(project_col):
        group_encoded = group.copy()
        for col in columns:
            if col in group_encoded.columns and col in periods:
                period = periods[col]
                group_encoded[f'{col}_sin'] = np.sin(2 * np.pi * group_encoded[col] / period)
                group_encoded[f'{col}_cos'] = np.cos(2 * np.pi * group_encoded[col] / period)
                group_encoded = group_encoded.drop(columns=[col])
        df_encoded_list.append(group_encoded)

    df_encoded = pd.concat(df_encoded_list, ignore_index=True)
    return df_encoded

def drop_low_importance_features(X, importance_df, threshold=0.001):
    """
    Loại bỏ các feature có Average_Importance nhỏ hơn ngưỡng từ DataFrame.

    Parameters:
    - X (pd.DataFrame): Dữ liệu đầu vào (features).
    - importance_df (pd.DataFrame): DataFrame chứa cột 'Feature' và 'Average_Importance'.
    - threshold (float): Ngưỡng để xác định importance xấp xỉ 0 (mặc định 0.01).

    Returns:
    - pd.DataFrame: DataFrame đã loại bỏ các feature có importance xấp xỉ 0.
    - list: Danh sách các feature bị loại bỏ.
    """
    features_to_drop = importance_df[importance_df['Average_Importance'] < threshold]['Feature'].tolist()

    # In danh sách feature bị loại bỏ
    if features_to_drop:
        print("Các feature có importance xấp xỉ 0 (dưới ngưỡng", threshold, ") bị loại bỏ:")
        for feature in features_to_drop:
            print(feature)
    else:
        print("Không có feature nào có importance xấp xỉ 0 (dưới ngưỡng", threshold, ").")

    # Loại bỏ feature khỏi X
    X_filtered = X.drop(columns=features_to_drop)
    return X_filtered, features_to_drop

def save_projects_to_files(df, output_dir="../data/processed", project_column='gh_project_name'):
    """Lưu từng dự án thành file CSV."""
    os.makedirs(output_dir, exist_ok=True)
    grouped = df.groupby(project_column)
    saved_files = {}
    for project, group_df in grouped:
        safe_project_name = project.replace('/', '_')
        file_path = os.path.join(output_dir, f"{safe_project_name}.csv")
        group_df.to_csv(file_path, index=False)
        saved_files[project] = file_path
    return saved_files

def boolean_to_float(df):
    for col in df.columns:
        temp_col = df[col].astype(str).str.strip().str.lower()
        if temp_col.isin(['true', 'false', '1', '0', '1.0', '0.0']).all():
            df[col] = temp_col.map({
                'true': 1.0, '1': 1.0, '1.0': 1.0,
                'false': 0.0, '0': 0.0, '0.0': 0.0
            }).astype(float)

        elif df[col].dtype == bool:
            df[col] = df[col].astype(float)
    return df