import logging
import os
import pandas as pd
import tempfile
import time
from fastapi import APIRouter, HTTPException
from api.routes.retrain import trigger_training_message
from api.schemas.append import AppendData
from dagshub.upload import Repo
from dotenv import load_dotenv
from src.data.main import preprocess_data

logger = logging.getLogger("api.append")
load_dotenv()

router = APIRouter()

@router.post("/dataset/append")
async def append(data: AppendData):
    start_time = time.perf_counter()
    try:
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

        # Xử lý dữ liệu qua preprocess_data
        processed_features = preprocess_data(is_training=False, input=input_df)

        # Tạo file tạm thời để lưu dữ liệu đã xử lý
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as temp_file:
            local_path = temp_file.name
            processed_features.to_csv(local_path, index=False)
            logger.debug(f"Save the processed data to the temporary file: {local_path}")

        # Khởi tạo Repo DagsHub
        dagshub_user = os.environ.get("DAGSHUB_USER")
        dagshub_name = os.environ.get("DAGSHUB_NAME")
        if not dagshub_user or not dagshub_name:
            logger.error("Lack of environmental variables dagshub_user or dagshub_name")
            raise HTTPException(status_code=500, detail="Lack of Dagshub configuration information")

        repo = Repo(dagshub_user, dagshub_name)

        # Định nghĩa đường dẫn từ xa trên DagsHub
        project_name = input_df["gh_project_name"].iloc[0]
        project_name = project_name.split("/", 1)[-1]
        branch = input_df["git_branch"].iloc[0]
        remote_path = f"data/processed-local/{project_name}-{branch}.csv"

        # Tải file lên DagsHub
        repo.upload(
            local_path=local_path,
            remote_path=remote_path,
            versioning="dvc",
            commit_message=f"Append {project_name} data"
        )
        logger.info(f"Has uploaded the processed file to Dagshub: {remote_path}")

        # Xóa file tạm
        os.remove(local_path)
        logger.debug(f"Delete the temporary file: {local_path}")

        if data.retrain:
            await trigger_training_message()

        duration = time.perf_counter() - start_time
        return {
            "status": 200,
            "message": "Preprocessed data appended successfully to DagsHub",
            "remote_path": remote_path,
            "execution_time": duration,
        }

    except Exception as e:
        logger.error(f"Error when adding data to Dagshub: {str(e)}", exc_info=True)
        if "local_path" in locals():
            try:
                os.remove(local_path)
                logger.debug(f"Delete the temporary file due to the error: {local_path}")
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error when adding data: {str(e)}")
