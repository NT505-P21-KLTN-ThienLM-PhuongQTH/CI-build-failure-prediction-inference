from fastapi import APIRouter, HTTPException
from mlflow import MlflowClient
import mlflow
import pandas as pd
import numpy as np
import logging.config
from keras.models import load_model
from keras import losses
from src.data.feature_analysis import prepare_features
from src.data.main import preprocess_data
import os
import hashlib

logger = logging.getLogger("api.predict")

class LSTMPredictionPipeline:
    def __init__(self, predict_name, predict_version=None, padding_name=None, padding_version=None, cache_dir="cache/models"):
        self.predict_name = predict_name
        self.predict_version = predict_version
        self.padding_name = padding_name
        self.padding_version = padding_version
        self.cache_dir = cache_dir
        self.mlflow_client = MlflowClient()
        self.predict = None
        self.padding = None
        self.time_step = None
        self.threshold = None
        self.input_dim = None
        self.load_models()

    def _generate_cache_path(self, model_name, model_version):
        """Generate a unique cache file path for a model based on name and version."""
        cache_key = f"{model_name}_{model_version or 'latest'}"
        cache_filename = hashlib.md5(cache_key.encode()).hexdigest() + ".h5"
        return os.path.join(self.cache_dir, cache_filename)

    def _get_model_version(self, model_name, model_version=None):
        """
        Lấy thông tin phiên bản mô hình từ MLflow. Nếu không chỉ định version, lấy version mới nhất.
        """
        try:
            if model_version:
                return self.mlflow_client.get_model_version(name=model_name, version=model_version), model_version
            versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise Exception(f"Không tìm thấy version nào cho model: {model_name}")
            latest_version_info = max(versions, key=lambda v: int(v.version))
            return latest_version_info, latest_version_info.version

        except Exception as e:
            logger.error(f"Lấy phiên bản mô hình {model_name} thất bại: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Lấy phiên bản mô hình thất bại: {str(e)}")

    def load_models(self):
        """Tải mô hình dự đoán và module padding từ cache hoặc MLflow."""
        try:
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)

            # Custom objects for deserialization
            custom_objects = {
                "mse": losses.MeanSquaredError(),
                "MeanSquaredError": losses.MeanSquaredError(),
                "mean_squared_error": losses.MeanSquaredError()
            }

            # Tải mô hình dự đoán
            predict_info, self.predict_version = self._get_model_version(self.predict_name, self.predict_version)
            predict_cache_path = self._generate_cache_path(self.predict_name, self.predict_version)

            if os.path.exists(predict_cache_path):
                try:
                    logger.info(f"Loading predict model from cache: {predict_cache_path}")
                    self.predict = load_model(predict_cache_path, custom_objects=custom_objects)
                except Exception as e:
                    logger.warning(f"Failed to load predict model from cache: {str(e)}. Removing corrupted cache file.")
                    os.remove(predict_cache_path)  # Remove corrupted cache file
                    raise  # Re-raise to trigger MLflow download
            else:
                predict_uri = f"models:/{self.predict_name}/{self.predict_version}"
                logger.info(f"Downloading predict model from {predict_uri}")
                self.predict = mlflow.keras.load_model(predict_uri)
                # Save model to cache
                self.predict.save(predict_cache_path)
                logger.info(f"Saved predict model to cache: {predict_cache_path}")

            # Tải module padding
            padding_info, self.padding_version = self._get_model_version(self.padding_name, self.padding_version)
            padding_cache_path = self._generate_cache_path(self.padding_name, self.padding_version)

            if os.path.exists(padding_cache_path):
                try:
                    logger.info(f"Loading padding model from cache: {padding_cache_path}")
                    self.padding = load_model(padding_cache_path, custom_objects=custom_objects)
                except Exception as e:
                    logger.warning(f"Failed to load padding model from cache: {str(e)}. Removing corrupted cache file.")
                    os.remove(padding_cache_path)
                    raise
            else:
                padding_uri = f"models:/{self.padding_name}/{self.padding_version}"
                logger.info(f"Downloading padding model from {padding_uri}")
                self.padding = mlflow.keras.load_model(padding_uri)
                self.padding.save(padding_cache_path)
                logger.info(f"Saved padding model to cache: {padding_cache_path}")

            predict_run_id = predict_info.run_id
            predict_run = self.mlflow_client.get_run(predict_run_id)
            params = predict_run.data.params
            self.time_step = int(params.get("best_time_step"))
            self.threshold = float(params.get("best_threshold"))
            self.input_dim = int(params.get("input_dim", 1))

        except Exception as e:
            logger.error(f"Tải mô hình thất bại: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Tải mô hình thất bại: {str(e)}")

    def preprocess_input_data(self, input_data):
        """Tiền xử lý dữ liệu đầu vào."""
        try:
            # Chuyển đổi dữ liệu thành DataFrame
            input_df = pd.DataFrame(input_data)
            logger.debug(f"Các cột của DataFrame đầu vào: {list(input_df.columns)}")

            if len(input_df) < self.time_step:
                short_timestep = len(input_df)
            else:
                input_df = input_df[-self.time_step:]

            # Tiền xử lý dữ liệu
            processed_features = preprocess_data(is_training=False, input=input_df)
            X, y = prepare_features(processed_features, target_column='build_failed')

            # Padding nếu chuỗi ngắn hơn time_step
            if len(X) < self.time_step:
                logger.info(f"Padding dữ liệu từ {len(X)} đến {self.time_step}...")
                X = self.apply_padding(X.values)

            # Tạo sliding window
            input_sequence = X[-self.time_step:]
            model_input = input_sequence[np.newaxis, :, :]  # Thêm batch dimension
            return model_input

        except Exception as e:
            logger.error(f"Tiền xử lý thất bại: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Tiền xử lý thất bại: {str(e)}")

    def apply_padding(self, sequence):
        """Áp dụng padding sử dụng padding module."""
        try:
            # sequence là numpy array với shape [short_len, input_dim]
            target_length = self.time_step
            if len(sequence) >= target_length:
                return sequence[:target_length]

            # Sử dụng logic padding từ PaddingModule.pad_sequence
            padded_sequence = np.zeros((target_length, self.input_dim))
            padded_sequence[-len(sequence):] = sequence

            current_seq = sequence
            for i in range(target_length - len(sequence)):
                input_seq = current_seq[np.newaxis, :, :]
                next_vector = self.padding.predict(input_seq, verbose=0)
                padded_sequence[target_length - len(sequence) - 1 - i] = next_vector
                current_seq = np.vstack([next_vector, current_seq])
                if len(current_seq) > self.time_step:
                    current_seq = current_seq[-self.time_step:]

            return padded_sequence

        except Exception as e:
            logger.error(f"Padding thất bại: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Padding thất bại: {str(e)}")

    def predict_build(self, input_data):
        """Dự đoán trên dữ liệu đầu vào."""
        try:
            logger.info("wendy")
            model_input = self.preprocess_input_data(input_data)
            predictions = self.predict.predict(model_input)
            logger.info(f"Dự đoán thành công cho {len(input_data)} bản build đầu vào.")
            return predictions, self.threshold, self.predict_name, self.predict_version

        except Exception as e:
            logger.error(f"Dự đoán thất bại: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Dự đoán thất bại: {str(e)}")