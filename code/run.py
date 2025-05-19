"""
Main entry poin to the project in order to train a MLPNAS sampling architectures tailored to a classification problem (either binary or multi-class doesn't make a difference) on a given dataset. Refer to the arguments at the bottom to select a different one:

python run.py --dataset=<dataset_name>

Refer also to constants.py for the main hyperparameters of the model.
"""

import pandas as pd
import argparse

from utils import get_top_n_architectures, get_accuracy_distribution
from datautils import load_dataset

# MLPNAS object class
from mlpnas import MLPNAS
import numpy as np
import pickle

from typing_extensions import Literal

# Pre-made MLP classifier to compare against
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def main(args):
    Xtr, ytr, Xte, yte, num_classes = load_dataset(args.dataset)

    # 以下部分为数据不平衡扩充，转换独热编码为标签（如果是独热的）
    if ytr.ndim > 1 and ytr.shape[1] > 1:
        y_train_original_labels = np.argmax(ytr, axis=1)
    else:
        y_train_original_labels = ytr.flatten() if ytr.ndim == 2 else ytr

    print(f"Original training dataset shape: X: {Xtr.shape}, y: {y_train_original_labels.shape}")
    print(f"Original training class distribution: {np.bincount(y_train_original_labels)}")

    # 计算类别数量
    class_counts = np.bincount(y_train_original_labels)
    majority_class = np.argmax(class_counts)
    minority_class = 1 - majority_class

    # 设置目标样本数（你可以改成其他更合适的值）
    target_samples = min(2000, class_counts[majority_class])

    # 定义组合采样策略：先下采样，再 SMOTE 过采样
    undersample = RandomUnderSampler(sampling_strategy={majority_class: target_samples}, random_state=42)
    oversample = SMOTE(sampling_strategy={minority_class: target_samples}, random_state=42)
    pipeline = Pipeline([('under', undersample), ('over', oversample)])

    try:
        # 组合采样
        X_train_resampled, y_train_resampled_original_labels = pipeline.fit_resample(Xtr, y_train_original_labels)

        print(f"Resampled dataset shape: X: {X_train_resampled.shape}, y: {y_train_resampled_original_labels.shape}")
        print(f"Resampled class distribution: {np.bincount(y_train_resampled_original_labels)}")

        # 处理独热编码
        if num_classes > 2:
            y_train_resampled_one_hot = to_categorical(y_train_resampled_original_labels, num_classes=num_classes)
        else:
            y_train_resampled_one_hot = y_train_resampled_original_labels.reshape(-1, 1)

        # 提供给 MLPNAS 的数据
        X_for_nas = X_train_resampled
        y_for_nas = y_train_resampled_one_hot

    except ValueError as e:
        print(f"SMOTE failed: {e}. This can happen if a class has too few samples (e.g., < k_neighbors for SMOTE).")
        print("Proceeding with original data for NAS.")
        X_for_nas = Xtr
        y_for_nas = ytr  # 或 y_train_original_labels
        
    nas = MLPNAS(X_for_nas, y_for_nas, Xte, yte,num_classes, task_name=args.dataset)
    
    
    # nas = MLPNAS(Xtr, ytr, Xte, yte,num_classes, task_name=args.dataset)

    # Traverse MLPNAS's search space and sample plausible architectures
    data = nas.search()
    
    # 加载之前保存的 search 数据
    # with open('/root/Desktop/mlp-nas/code/LOGS/event1739511885/nas_data.pkl', 'rb') as f:
    #     data = pickle.load(f)
    
    top_n_architectures = get_top_n_architectures(args.n, num_classes) # Display Top n architectures
    print(f"最优的前{args.n}个架构为：{top_n_architectures}")
    # Extract best architecture
    best_model = nas.extract_best_model(data)

    # Fine-tune the best architecture on the training set
    # Xtr for another 50 epochs (unless it stops early)
    #
    # Assign the return to a variable which will
    # hold training's history in case you need to
    #
    # i.e.: best_model_history
    _ = nas.finetune_model(
        best_model, Xtr, ytr, validation_split=0.2, 
        batch_size=64, shuffle=True, epochs=50, save=args.save
    )

    # from utils import plot_history
    # plot_history(best_model_history)
    
    # MLPNAS.shapley(best_model)
    tloss, tacc, f1_score, auc = MLPNAS.score(best_model, Xte, yte)
    print(f"NAS-made MLP test loss: {tloss}")
    print(f"NAS-made MLP test accuracy: {tacc}")
    print(f"NAS-made MLP test f1_score: {f1_score}")
    print(f"NAS-made MLP test auc: {auc}")
    
    MLPNAS.shap_hot(best_model)

    # Test against sklearn MLP classifier
    # mlp = MLPClassifier(activation="relu",
    #                     solver="adam", alpha=0.01).fit(Xtr, ytr)

    # print(f"\nScikit-learn MLP test accuracy: {mlp.score(Xte, yte)}")

    # Validation accuracies barplot
    get_accuracy_distribution(args.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO: Move hyperparameters from constants.py to here
    parser.add_argument('--save', default=True, action='store_true',
                         help='Whether to save the best architecture sampled by MLPNAS')
    parser.add_argument('--no-save', dest='save', action='store_false')

    parser.add_argument('--dataset', type=str, default='mnist',
                         choices=('mnist', 'crimes', '10speakers', 'wine'),
                         help='Name of the dataset to run MLPNAS against')

    parser.add_argument('--n', type=int, default=5,
                         help='How many architectures to show after MLPNAS sampling')

    # Parse arguments and run
    args = parser.parse_args()
    main(args)