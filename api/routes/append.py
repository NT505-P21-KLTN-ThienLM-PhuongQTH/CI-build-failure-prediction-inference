import logging
import os
import pandas as pd
import tempfile
import time
from fastapi import APIRouter, HTTPException
from api.schemas.append import AppendData
from dagshub.upload import Repo
from dotenv import load_dotenv
from src.data.main import preprocess_data

logger = logging.getLogger("api.append")
load_dotenv()

router = APIRouter()

@router.post("/dataset/append")
async def append(data: AppendData):
    """
    Endpoint to preprocess ci_builds data and append to DagsHub repository in data/processed-local.

    Args:
        data: List of AppendData containing ci_builds to be preprocessed and uploaded.

    Returns:
        dict: Message indicating success or failure, with remote path.

    Raises:
        HTTPException: If there's an error during processing or upload.
    """
    start_time = time.perf_counter()
    try:
        # Kiểm tra dữ liệu đầu vào
        if not data or not data.ci_builds:
            logger.error("Dữ liệu ci_builds rỗng hoặc không hợp lệ")
            raise HTTPException(status_code=400, detail="Dữ liệu ci_builds rỗng hoặc không hợp lệ")

        # Chuyển đổi ci_builds thành danh sách dictionary
        ci_builds = data.ci_builds
        if not ci_builds:
            logger.error("Không có bản ghi nào trong ci_builds")
            raise HTTPException(status_code=400, detail="Không có bản ghi nào trong ci_builds")

        # Tạo DataFrame từ ci_builds
        input_df = pd.DataFrame(ci_builds)
        logger.debug(f"Các cột của DataFrame đầu vào: {list(input_df.columns)}")

        # Xử lý dữ liệu qua preprocess_data
        processed_features = preprocess_data(is_training=False, input=input_df)
        logger.debug(f"Các cột của DataFrame sau preprocess: {list(processed_features.columns)}")

        # Tạo file tạm thời để lưu dữ liệu đã xử lý
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as temp_file:
            local_path = temp_file.name
            processed_features.to_csv(local_path, index=False)
            logger.debug(f"Đã lưu dữ liệu đã xử lý vào file tạm: {local_path}")

        # Khởi tạo Repo DagsHub
        dagshub_user = os.environ.get("DAGSHUB_USER")
        dagshub_name = os.environ.get("DAGSHUB_NAME")
        if not dagshub_user or not dagshub_name:
            logger.error("Thiếu biến môi trường DAGSHUB_USER hoặc DAGSHUB_NAME")
            raise HTTPException(status_code=500, detail="Thiếu thông tin cấu hình DagsHub")

        repo = Repo(dagshub_user, dagshub_name)

        # Định nghĩa đường dẫn từ xa trên DagsHub
        project_name = input_df["gh_project_name"].iloc[0]
        project_name = project_name.split("/", 1)[-1]
        remote_path = f"data/processed-local/{project_name}.csv"

        # Tải file lên DagsHub
        repo.upload(
            local_path=local_path,
            remote_path=remote_path,
            versioning="dvc",
            commit_message=f"Append {project_name} data"
        )
        logger.info(f"Đã tải file đã xử lý lên DagsHub: {remote_path}")

        # Xóa file tạm
        os.remove(local_path)
        logger.debug(f"Đã xóa file tạm: {local_path}")
        duration = time.perf_counter() - start_time
        return {
            "status": 200,
            "message": "Preprocessed data appended successfully to DagsHub",
            "remote_path": remote_path,
            "execution_time": duration,
        }

    except Exception as e:
        logger.error(f"Lỗi khi thêm dữ liệu vào DagsHub: {str(e)}", exc_info=True)
        if "local_path" in locals():
            try:
                os.remove(local_path)
                logger.debug(f"Đã xóa file tạm do lỗi: {local_path}")
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Lỗi khi thêm dữ liệu: {str(e)}")
