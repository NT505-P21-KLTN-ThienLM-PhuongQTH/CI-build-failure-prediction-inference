import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import mlflow
import os
import plotly.graph_objects as go
import plotly.subplots as sp


def plot_line(df, column, project, figsize=(10, 4)):
    """Vẽ biểu đồ đường cho một cột."""
    plt.figure(figsize=figsize)
    plt.plot(df[column].reset_index(drop=True), label=column, linewidth=1.2)
    plt.title(f"Line Plot of {column} for {project}")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    # plt.show()

def plot_pie(df, column_counts, title, figsize=(3, 3)):
    """Vẽ biểu đồ tròn."""
    plt.figure(figsize=figsize)
    plt.pie(df['Ratio (%)'], labels=df[column_counts], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title(title)
    plt.axis('equal')
    # plt.show()

def plot_multi_project(df, column, project_column='gh_project_name', figsize=(12, 6), dropna=True):
    """Vẽ biểu đồ cho nhiều dự án, tùy chọn xử lý NaN."""
    plt.figure(figsize=figsize)
    for project_name, group_df in df.groupby(project_column):
        # Chuyển đổi cột thành kiểu số, bỏ qua lỗi
        y_data = pd.to_numeric(group_df[column], errors='coerce')
        if dropna:
            # Chỉ vẽ các giá trị không phải NaN
            mask = ~y_data.isna()
            plt.plot(group_df.index[mask], y_data[mask], label=project_name, alpha=0.6)
        else:
            # Điền NaN bằng 0
            y_data = y_data.fillna(0)
            plt.plot(group_df.index, y_data, label=project_name, alpha=0.6)
    plt.title(f"Biểu đồ giá trị '{column}' theo project")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()


def plot_feature_importance(importance_df, title="Feature Importance (Average Across Models)", top_n=None):
    """
    Vẽ biểu đồ feature importance chỉ dựa trên giá trị trung bình (Average_Importance).

    Parameters:
    - importance_df (pd.DataFrame): DataFrame chứa cột 'Feature' và 'Average_Importance'.
    - title (str): Tiêu đề của biểu đồ.
    - top_n (int, optional): Số lượng feature hàng đầu để hiển thị. Nếu None, hiển thị tất cả.
    """
    if top_n is not None:
        importance_df = importance_df.head(top_n)

    n_features = len(importance_df)
    y_pos = np.arange(n_features)

    # Tạo figure và axes
    plt.figure(figsize=(10, max(6, n_features * 0.4)))
    bar_width = 0.5  # Độ rộng của thanh

    # Vẽ thanh ngang chỉ với Average_Importance
    bars = plt.barh(y_pos, importance_df['Average_Importance'], bar_width, color='#1f77b4', label='Average Importance')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(x=width + 0.02, y=bar.get_y() + bar.get_height() / 2, s=f'{width:.2f}',
                 ha='left', va='center', color='black')

    # Cài đặt trục và nhãn
    plt.yticks(y_pos, importance_df['Feature'])
    plt.xlabel('Feature Importance (Average)')
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.gca().invert_yaxis()
    # plt.show()


def plot_class_distribution(train_sets, test_sets, proj_name):
    """Vẽ biểu đồ tỉ lệ 0 và 1 trong cột build_failed cho từng fold."""
    plt.figure(figsize=(12, 6))
    folds = range(len(train_sets))

    # Tính tỉ lệ cho train và test
    train_ratios = [train_set['build_failed'].value_counts(normalize=True) for train_set in train_sets]
    test_ratios = [test_set['build_failed'].value_counts(normalize=True) for test_set in test_sets]

    # Chuẩn bị dữ liệu để vẽ
    train_0 = [r.get(0, 0) for r in train_ratios]
    train_1 = [r.get(1, 0) for r in train_ratios]
    test_0 = [r.get(0, 0) for r in test_ratios]
    test_1 = [r.get(1, 0) for r in test_ratios]

    # Thiết lập vị trí cột
    bar_width = 0.35
    x = np.arange(len(folds))

    # Vẽ biểu đồ
    train_bars_0 = plt.bar(x - bar_width / 2, train_0, bar_width, label='Train 0', color='skyblue')
    train_bars_1 = plt.bar(x - bar_width / 2, train_1, bar_width, bottom=train_0, label='Train 1', color='lightcoral')
    test_bars_0 = plt.bar(x + bar_width / 2, test_0, bar_width, label='Test 0', color='lightgreen')
    test_bars_1 = plt.bar(x + bar_width / 2, test_1, bar_width, bottom=test_0, label='Test 1', color='salmon')

    # Thêm giá trị tỉ lệ trên cột
    for i in range(len(folds)):
        # Train 0
        plt.text(x[i] - bar_width / 2, train_0[i] / 2, f'{train_0[i]:.2f}', ha='center', va='center', color='black')
        # Train 1
        plt.text(x[i] - bar_width / 2, train_0[i] + train_1[i] / 2, f'{train_1[i]:.2f}', ha='center', va='center', color='black')
        # Test 0
        plt.text(x[i] + bar_width / 2, test_0[i] / 2, f'{test_0[i]:.2f}', ha='center', va='center', color='black')
        # Test 1
        plt.text(x[i] + bar_width / 2, test_0[i] + test_1[i] / 2, f'{test_1[i]:.2f}', ha='center', va='center', color='black')

    plt.xlabel('Fold')
    plt.ylabel('Ratio')
    plt.title(f'Class Distribution (0 and 1) in build_failed - {proj_name}')
    plt.xticks(x, [f'Fold {i + 1}' for i in folds])
    plt.legend()
    plt.tight_layout()


def plot_roc_curve(y_true, y_pred_probs, proj_name, fold_idx):
    """
    Vẽ đường cong ROC và tính AUC.

    Parameters:
    - y_true: Nhãn thực tế (ground truth).
    - y_pred_probs: Xác suất dự đoán cho lớp positive (class 1).
    - proj_name: Tên dự án (dùng để đặt tiêu đề).
    - fold_idx: Chỉ số fold (dùng để đặt tiêu đề).
    - save_path: Đường dẫn để lưu biểu đồ (nếu None thì không lưu).
    """
    # Tính FPR, TPR và ngưỡng
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    # Tạo biểu đồ ROC bằng Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(color='darkorange', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(color='navy', width=2, dash='dash')
    ))

    # Cập nhật layout
    fig.update_layout(
        title=f'ROC Curve - {proj_name} (Fold {fold_idx + 1})',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.05]),
        legend=dict(x=0.7, y=0.1),
        width=800,
        height=600
    )

    # Lưu biểu đồ dưới dạng HTML và log vào MLflow
    html_path = f"roc_curve_{proj_name}_fold_{fold_idx + 1}.html"
    fig.write_html(html_path)
    mlflow.log_artifact(html_path, artifact_path="plots/roc")
    os.remove(html_path)
    print(f"Logged ROC curve for {proj_name} (Fold {fold_idx + 1}) to MLflow")


def plot_training_history(history, proj_name, fold_idx):
    # Tạo subplot với 1 hàng, 2 cột
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))

    # Biểu đồ Loss
    epochs = list(range(1, len(history.history['loss']) + 1))
    fig.add_trace(
        go.Scatter(x=epochs, y=history.history['loss'], mode='lines', name='Train Loss', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history.history['val_loss'], mode='lines', name='Validation Loss',
                   line=dict(color='orange')),
        row=1, col=1
    )

    # Biểu đồ Accuracy
    fig.add_trace(
        go.Scatter(x=epochs, y=history.history['accuracy'], mode='lines', name='Train Accuracy',
                   line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy',
                   line=dict(color='orange')),
        row=1, col=2
    )

    # Cập nhật layout
    fig.update_layout(
        title=f'Training History - {proj_name} (Fold {fold_idx + 1})',
        width=1200,
        height=500
    )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    # Lưu biểu đồ dưới dạng HTML và log vào MLflow
    html_path = f"training_history_{proj_name}_fold_{fold_idx + 1}.html"
    fig.write_html(html_path)
    mlflow.log_artifact(html_path, artifact_path="plots/training_history")
    os.remove(html_path)
    print(f"Logged training history plot for {proj_name} (Fold {fold_idx + 1}) to MLflow")

def plot_metrics(train_entries, test_entries, title, COLUMNS_RES):
    train_df = pd.DataFrame(train_entries)[COLUMNS_RES]
    test_df = pd.DataFrame(test_entries)[COLUMNS_RES]

    print(f"\n{title} - Train Results:")
    print(train_df.groupby(['proj', 'exp']).mean(numeric_only=True))
    print(f"\n{title} - Test Results:")
    print(test_df.groupby(['proj', 'exp']).mean(numeric_only=True))

    # Tạo biểu đồ AUC
    fig_auc = go.Figure()
    for proj in test_df['proj'].unique():
        proj_data = test_df[test_df['proj'] == proj]
        fig_auc.add_trace(go.Box(
            y=proj_data['AUC'],
            name=proj,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    fig_auc.update_layout(
        title=f'AUC ({title})',
        xaxis_title='Project',
        yaxis_title='AUC'
    )

    # Tạo biểu đồ Accuracy
    fig_accuracy = go.Figure()
    for proj in test_df['proj'].unique():
        proj_data = test_df[test_df['proj'] == proj]
        fig_accuracy.add_trace(go.Box(
            y=proj_data['accuracy'],
            name=proj,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    fig_accuracy.update_layout(
        title=f'Accuracy ({title})',
        xaxis_title='Project',
        yaxis_title='Accuracy'
    )

    # Tạo biểu đồ F1
    fig_f1 = go.Figure()
    for proj in test_df['proj'].unique():
        proj_data = test_df[test_df['proj'] == proj]
        fig_f1.add_trace(go.Box(
            y=proj_data['F1'],
            name=proj,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    fig_f1.update_layout(
        title=f'F1 ({title})',
        xaxis_title='Project',
        yaxis_title='F1'
    )

    # Ghi các biểu đồ vào MLflow
    html_path_auc = f"{title.lower().replace(' ', '_')}_auc.html"
    html_path_accuracy = f"{title.lower().replace(' ', '_')}_accuracy.html"
    html_path_f1 = f"{title.lower().replace(' ', '_')}_f1.html"
    fig_auc.write_html(html_path_auc)
    fig_accuracy.write_html(html_path_accuracy)
    fig_f1.write_html(html_path_f1)
    mlflow.log_artifact(html_path_auc, artifact_path="plots/metrics")
    mlflow.log_artifact(html_path_accuracy, artifact_path="plots/metrics")
    mlflow.log_artifact(html_path_f1, artifact_path="plots/metrics")
    os.remove(html_path_auc)
    os.remove(html_path_accuracy)
    os.remove(html_path_f1)
    print(f"Logged interactive metrics plots for {title} to MLflow")

    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # for i, metric in enumerate(['AUC', 'accuracy', 'F1']):
    #     test_df.boxplot(column=metric, by='proj', ax=axes[i])
    #     axes[i].set_title(f"{metric} ({title})")
    #     axes[i].set_xlabel("Project")
    #     axes[i].set_ylabel(metric)
    #     axes[i].tick_params(axis='x', rotation=45)


    # plt.tight_layout()
    # # plt.show()