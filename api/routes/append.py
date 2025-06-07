import logging
import os
import pandas as pd
import tempfile
import time
from fastapi import APIRouter, HTTPException
from api.routes.retrain import trigger_training_message
from api.schemas.append import AppendData
from dagshub.upload import Repo
from dagshub.streaming import DagsHubFilesystem
from dotenv import load_dotenv
from src.data.main import preprocess_data
import requests
from io import StringIO

logger = logging.getLogger("api.append")
load_dotenv()

router = APIRouter()


def download_existing_file_with_streaming(dagshub_user, dagshub_name , remote_path):
    try:
        url = f"https://dagshub.com/api/v1/repos/{dagshub_user}/{dagshub_name}/raw/main/{remote_path}"
        headers = {"Authorization": f"Bearer {os.environ.get('DAGSHUB_TOKEN')}"}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        content = response.text
        if content.strip():
            existing_df = pd.read_csv(StringIO(content), encoding='utf-8', on_bad_lines='warn', sep=',')
            logger.info(f"Downloaded the existing file with {len(existing_df)} rows")
            return existing_df
        else:
            logger.warning(f"The file exists but empty: {remote_path}")
            return pd.DataFrame()
    except requests.RequestException as e:
        if e.response and e.response.status_code == 404:
            logger.info(f"The file doesn't exists: {remote_path}")
            return pd.DataFrame()
        logger.warning(f"File download error: {str(e)}")
        raise

def check_duplicate_builds(existing_df, new_df, timestamp_column='gh_build_started_at'):
    """Check for duplicate builds based on timestamp and remove duplicates from new data"""
    if existing_df.empty:
        logger.info("No existing data, returning all new data")
        return new_df

    if timestamp_column not in existing_df.columns or timestamp_column not in new_df.columns:
        logger.warning(f"Timestamp column '{timestamp_column}' not found, skipping duplicate check")
        return new_df

    # Convert timestamps to comparable format
    existing_timestamps = set(existing_df[timestamp_column].astype(str))
    logger.info(f"Found {len(existing_timestamps)} existing timestamps")

    # Filter out duplicates from new data
    mask = ~new_df[timestamp_column].astype(str).isin(existing_timestamps)
    filtered_df = new_df[mask].copy()

    duplicates_count = len(new_df) - len(filtered_df)
    if duplicates_count > 0:
        logger.info(f"Filtered out {duplicates_count} duplicate builds based on {timestamp_column}")
    else:
        logger.info("No duplicates found in new data")

    return filtered_df

def validate_and_process_input(data: AppendData) -> pd.DataFrame:
    # Kiểm tra dữ liệu đầu vào
    if not data or not data.ci_builds:
        logger.error("ci_builds data is empty or invalid")
        raise HTTPException(status_code=400, detail="ci_builds data is empty or invalid")

    # Chuyển đổi ci_builds thành danh sách dictionary
    ci_builds = data.ci_builds
    if not ci_builds:
        logger.error("There are no records in ci_builds")
        raise HTTPException(status_code=400, detail="There are no records in ci_builds")

    # Tạo DataFrame từ ci_builds
    input_df = pd.DataFrame(ci_builds)
    logger.info(f"Input data shape: {input_df.shape}")

    # Xử lý dữ liệu qua preprocess_data
    processed_features = preprocess_data(is_training=False, input=input_df)
    logger.info(f"Processed data shape: {processed_features.shape}")

    return input_df, processed_features

def prepare_dagshub_and_paths(input_df: pd.DataFrame) -> tuple:
    # Khởi tạo Repo DagsHub
    dagshub_user = os.environ.get("DAGSHUB_USER")
    dagshub_name = os.environ.get("DAGSHUB_NAME")
    dagshub_repo = os.environ.get("DAGSHUB_REPO")
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    if not dagshub_user or not dagshub_name:
        logger.error("Lack of environmental variables dagshub_user or dagshub_name")
        raise HTTPException(status_code=500, detail="Lack of Dagshub configuration information")

    repo = Repo(dagshub_user, dagshub_name)

    # Định nghĩa đường dẫn từ xa trên DagsHub
    project_name = input_df["gh_project_name"].iloc[0]
    project_name = project_name.split("/", 1)[-1]
    branch = input_df["git_branch"].iloc[0]
    remote_path = f"data/processed-local/{project_name}-{branch}.csv"

    return repo, dagshub_repo, dagshub_user, dagshub_name, dagshub_token, remote_path, project_name, branch

def merge_existing_and_new_data(existing_df: pd.DataFrame, filtered_new_data: pd.DataFrame) -> pd.DataFrame:
    if not existing_df.empty:
        # Ensure both dataframes have the same columns
        existing_cols = set(existing_df.columns)
        new_cols = set(filtered_new_data.columns)

        if existing_cols != new_cols:
            logger.warning(f"Column mismatch - Existing: {existing_cols}, New: {new_cols}")
            # Add missing columns with NaN values
            for col in existing_cols - new_cols:
                filtered_new_data[col] = None
            for col in new_cols - existing_cols:
                existing_df[col] = None
            # Reorder columns to match
            column_order = list(existing_df.columns)
            filtered_new_data = filtered_new_data[column_order]

        combined_df = pd.concat([existing_df, filtered_new_data], ignore_index=True)
        logger.info(f"Combining {len(existing_df)} existing rows with {len(filtered_new_data)} new rows")
        logger.info(f"Combined data shape: {combined_df.shape}")
    else:
        combined_df = filtered_new_data
        logger.info(f"Creating new file with {len(filtered_new_data)} rows")

    # Verify combined data integrity
    logger.info(f"Final combined data shape: {combined_df.shape}")
    if len(combined_df) < max(len(existing_df), len(filtered_new_data)):
        logger.error("Data loss detected during combination!")
        logger.error(
            f"Expected at least {max(len(existing_df), len(filtered_new_data))} rows, got {len(combined_df)}")

    # Sort by timestamp to maintain order (optional)
    combined_df['gh_build_started_at'] = pd.to_datetime(combined_df['gh_build_started_at'], errors='coerce')
    combined_df = combined_df.sort_values('gh_build_started_at').reset_index(drop=True)

    return combined_df

@router.post("/dataset/append")
async def append(data: AppendData):
    start_time = time.perf_counter()
    local_path = None

    try:
        input_df, processed_features = validate_and_process_input(data)

        repo, dagshub_repo, dagshub_user, dagshub_name, dagshub_token, remote_path, project_name, branch = prepare_dagshub_and_paths(input_df)

        # Download existing file using streaming method
        logger.info(f"Checking for existing file: {remote_path}")
        existing_df = download_existing_file_with_streaming(dagshub_user, dagshub_name, remote_path)

        logger.info(f"Existing data shape: {existing_df.shape}")
        logger.info(f"New processed data shape: {processed_features.shape}")

        # Debug: Log some sample data
        if not existing_df.empty:
            logger.debug(f"Existing data sample:\n{existing_df.head(2)}")
        logger.debug(f"New data sample:\n{processed_features.head(2)}")

        # Check for duplicates and filter new data
        filtered_new_data = check_duplicate_builds(existing_df, processed_features)

        logger.info(f"Filtered new data shape: {filtered_new_data.shape}")

        if filtered_new_data.empty:
            logger.info("No new data to append after filtering duplicates")
            duration = time.perf_counter() - start_time
            return {
                "status": 200,
                "message": "No new data to append (all builds already exist)",
                "remote_path": remote_path,
                "execution_time": duration,
                "rows_added": 0,
                "total_rows": len(existing_df)
            }

        combined_df = merge_existing_and_new_data(existing_df, filtered_new_data)

        # Tạo file tạm thời để lưu dữ liệu đã kết hợp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as temp_file:
            local_path = temp_file.name
            combined_df.to_csv(local_path, index=False)
            logger.info(f"Saved combined data to temporary file: {local_path}")

        # Verify the saved file
        verify_df = pd.read_csv(local_path)
        logger.info(f"Verified saved file shape: {verify_df.shape}")
        if len(verify_df) != len(combined_df):
            logger.error(
                f"File save verification failed! Expected {len(combined_df)} rows, file has {len(verify_df)} rows")

        # Retrieve the last commit hash for the file or branch
        try:
            url = f"https://dagshub.com/api/v1/repos/{dagshub_user}/{dagshub_name}/branches/main"
            headers = {"Authorization": f"Bearer {dagshub_token}"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            last_commit = response.json().get("commit", {}).get("id")
            if not last_commit:
                logger.error("Could not retrieve last commit hash")
                raise HTTPException(status_code=500, detail="Could not retrieve last commit hash")
        except requests.RequestException as e:
            logger.error(f"Error retrieving last commit: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving last commit: {str(e)}")

        # Tải file lên DagsHub with last_commit
        rows_added = len(filtered_new_data)
        commit_message = f"Append {rows_added} rows to {project_name}/{branch} (total: {len(combined_df)} rows)"

        repo.upload(
            local_path=local_path,
            remote_path=remote_path,
            versioning="dvc",
            commit_message=commit_message,
            last_commit=last_commit
        )
        logger.info(f"Successfully uploaded updated file to DagsHub: {remote_path}")

        # Xóa file tạm
        os.remove(local_path)
        logger.debug(f"Deleted temporary file: {local_path}")

        if data.retrain:
            await trigger_training_message(data.model_name)

        duration = time.perf_counter() - start_time
        return {
            "status": 200,
            "message": f"Successfully appended {rows_added} new rows to existing dataset",
            "remote_path": remote_path,
            "execution_time": duration,
            "rows_added": rows_added,
            "total_rows": len(combined_df),
            "duplicates_filtered": len(processed_features) - rows_added
        }

    except Exception as e:
        logger.error(f"Error when adding data to DagsHub: {str(e)}", exc_info=True)
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                logger.debug(f"Deleted temporary file due to error: {local_path}")
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error when adding data: {str(e)}")
