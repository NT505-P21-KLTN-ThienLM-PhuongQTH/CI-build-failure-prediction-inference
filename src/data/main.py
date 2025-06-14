# src/data/main.py
import os
import logging.config
import pandas as pd
import numpy as np
import logging
from src.data.processing import load_data, process_status, summarize_projects, fill_nan_values, \
    encode_categorical_columns, normalize_numerical_columns, encode_cyclical_time_features, save_projects_to_files, \
    drop_low_importance_features, add_build_features, boolean_to_float
from src.data.feature_analysis import prepare_features, aggregate_feature_importance
import yaml

with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def split_into_sequences(X, y, time_step, return_y=False):
    try:
        if time_step <= 0:
            logging.error(f"time_step must be positive, got {time_step}")
            raise ValueError(f"time_step must be positive, got {time_step}")

        if X.shape[1] == 0:
            logging.warning("No features remain after prepare_features. Returning empty sequences.")
            return np.empty((0, time_step, 0)) if not return_y else (np.empty((0, time_step, 0)), np.array([]))

        if len(X) < time_step:
            logging.warning(f"Data has only {len(X)} rows, which is less than time_step={time_step}")
            return np.empty((0, time_step, X.shape[1]))

        # Tạo các chuỗi theo time_step
        X_values = X.values
        sequences = [X_values[i:i + time_step] for i in range(len(X_values) - time_step + 1)]
        X_sequences = np.array(sequences)

        logging.info(f"Created {len(X_sequences)} sequences with time_step={time_step} and {X.shape[1]} features")
        if return_y:
            y_values = y.values
            y_sequences = y_values[time_step - 1:len(y_values)]
            return X_sequences, y_sequences
        return X_sequences

    except Exception as e:
        logging.error(f"Failed to split data into sequences: {str(e)}")
        raise

def preprocess_data(is_training=None, DO_FEATURE_IMPORTANCE=False, target_features=None, input=None):
    # Setup pandas and logging
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', None)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if target_features is None:
        target_features = config["model"]["target_features"]

    # Load and process label
    if is_training:
        csv_path = os.path.join(PROJECT_ROOT, "../../data/combined/combined_travistorrent.csv")
        combined_df = load_data(csv_path)
        combined_df = process_status(combined_df, 'tr_status').copy()

    else:
        combined_df = input
        combined_df.rename(columns={"build_duration": "tr_duration"}, inplace=True)
        combined_df = process_status(combined_df, 'build_failed').copy()


    if is_training:
        summary_df = summarize_projects(combined_df, min_rows=50000, balance_threshold=0.7)
        selected_projects = summary_df['project'].head(10).tolist()
        logger.info(f"Selected projects: {selected_projects}")
    else:
        selected_projects = None  # Không lọc

    dataset = {
        project: data.copy()
        for project, data in combined_df.groupby(
            ['gh_project_name', 'git_branch'] if 'git_branch' in combined_df.columns else 'gh_project_name'
        )
        if selected_projects is None or project in selected_projects
    }
    logger.info(f"Projects in dataset: {list(dataset.keys())}")

    # Define processing parameters
    categorical_columns = []
    cyclical_time_columns = []
    periods = {"month_of_start": 12, "day_of_start": 31, "hour_of_start": 24, "day_week": 7}
    numerical_columns = [
        "gh_num_issue_comments", "gh_num_pr_comments", "gh_team_size", "gh_sloc",
        "git_diff_src_churn", "git_diff_test_churn", "gh_diff_files_added", "gh_diff_files_deleted",
        "gh_diff_files_modified", "gh_diff_tests_added", "gh_diff_tests_deleted", "gh_diff_src_files",
        "gh_diff_doc_files", "gh_diff_other_files", "gh_num_commits_on_files_touched",
        "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc",
        "gh_num_commit_comments",
        "tr_duration",
        "year_of_start", "elapsed_days_last_build", "proj_fail_rate_history", "proj_fail_rate_recent",
        "comm_fail_rate_history", "comm_fail_rate_recent", "comm_avg_experience",
        "num_files_edited", "num_distinct_authors",
        "month_of_start", "day_of_start", "hour_of_start", "day_week"
    ]

    # Initialize dictionary to store processed DataFrames
    processed_dfs = {}

    # Process each project's DataFrame

    for project_name, df in dataset.items():
        # Sort and filter data
        sorted_df = df.sort_values(by=['gh_build_started_at']).copy()

        # Process NaN and add features
        df_notnan = fill_nan_values(sorted_df)
        df_float = boolean_to_float(df_notnan)
        new_feature_df = add_build_features(df_float)

        merged_df = new_feature_df.copy()

        if is_training:
            # Handle duplicates
            grouped = new_feature_df.groupby('tr_build_id')
            non_identical_duplicates = []
            for build_id, group in grouped:
                if len(group) > 1:
                    if not group.drop(columns='tr_build_id').duplicated().all():
                        non_identical_duplicates.append(group)
            if not non_identical_duplicates:
                logger.info(f"No non-identical duplicates found for project {project_name}.")

            # Merge data
            merged_df = new_feature_df.groupby(['gh_project_name', 'gh_build_started_at', 'build_failed', 'tr_build_id'],
                                               as_index=False).agg({
                'gh_num_issue_comments': 'sum', 'gh_num_pr_comments': 'sum', 'gh_team_size': 'mean', 'gh_sloc': 'mean',
                'git_diff_src_churn': 'sum', 'git_diff_test_churn': 'sum', 'gh_diff_files_added': 'sum',
                'gh_diff_files_deleted': 'sum', 'gh_diff_files_modified': 'sum', 'gh_diff_tests_added': 'sum',
                'gh_diff_tests_deleted': 'sum', 'gh_diff_src_files': 'sum', 'gh_diff_doc_files': 'sum',
                'gh_diff_other_files': 'sum', 'gh_num_commits_on_files_touched': 'sum', 'gh_test_lines_per_kloc': 'mean',
                'gh_test_cases_per_kloc': 'mean', 'gh_asserts_cases_per_kloc': 'mean', 'gh_is_pr': 'max',
                'gh_by_core_team_member': 'max', 'gh_num_commit_comments': 'sum',
                'tr_duration': 'max',
                'year_of_start': 'first', 'month_of_start': 'first', 'day_of_start': 'first', 'hour_of_start': 'first',
                'elapsed_days_last_build': 'first', 'same_committer': 'max', 'proj_fail_rate_history': 'mean',
                'proj_fail_rate_recent': 'mean', 'comm_fail_rate_history': 'mean', 'comm_fail_rate_recent': 'mean',
                'comm_avg_experience': 'mean', 'no_config_edited': 'max',
                'num_files_edited': 'sum', 'num_distinct_authors': 'max', 'prev_build_result': 'first', 'day_week': 'first'
            })
            merged_df.drop_duplicates(inplace=True)

        # Feature encoding and normalization
        trans_df = merged_df.copy()
        trans_df_encoded, _ = encode_categorical_columns(trans_df, categorical_columns)
        trans_df_cyclical = encode_cyclical_time_features(trans_df_encoded, cyclical_time_columns, periods)
        trans_df_processed, _ = normalize_numerical_columns(trans_df_cyclical, numerical_columns)

        final_df = trans_df_processed.drop(columns=['tr_log_num_tests_failed', 'tr_build_id'], errors='ignore').copy()
        processed_dfs[project_name] = final_df

    logger.info({len(processed_dfs)})
    final_combined_df = pd.concat(processed_dfs.values(), ignore_index=True)

    if DO_FEATURE_IMPORTANCE:
        # Feature importance analysis
        X, y = prepare_features(final_combined_df, target_column='build_failed')
        importance_df = aggregate_feature_importance(X, y)
        final_combined_df, _ = drop_low_importance_features(X=final_combined_df, importance_df=importance_df, threshold=0.005)
    else:
        missing_cols = [col for col in target_features if col not in final_combined_df.columns]
        if missing_cols:
            raise ValueError(
                f"[ERROR] Column mismatch in DataFrame:\n"
                f"- Missing columns: {missing_cols}\n"
                f"The DataFrame must contain all columns defined in target_feature."
            )
        final_combined_df = final_combined_df[target_features].copy()

    if is_training:
        output_dir = os.path.join(PROJECT_ROOT, '../../data/processed-local/main')
        os.makedirs(output_dir, exist_ok=True)
        save_projects_to_files(final_combined_df, output_dir, 'gh_project_name')

    # Summarize final data
    summarize_projects(final_combined_df, min_rows=0, balance_threshold=1)
    return final_combined_df

if __name__ == "__main__":
    preprocess_data(is_training=True)