import os
import shap
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Model configuration
model_configs = {
    "超声-急诊": ("RandomForest", {'n_estimators': 357, 'max_depth': 14, 'min_samples_leaf': 1, 'max_features': 0.4021767833180502, 'bootstrap': True}),
    "心功能室-门诊": ("RandomForest", {'n_estimators': 106, 'max_depth': 10, 'min_samples_leaf': 4, 'max_features': 0.31154400313415803, 'bootstrap': False}),
    "咽喉镜-门诊": ("SVR", {'C': 13.374107709675418, 'epsilon': 0.9837057502857758, 'gamma': 'auto'}),
    "心功能室-急诊": ("XGBoost", {'n_estimators': 267, 'max_depth': 12, 'learning_rate': 0.012034582723863858, 'subsample': 0.927244701816625, 'colsample_bytree': 0.6556575947392053, 'gamma': 0.4881915805341791, 'min_child_weight': 3, 'reg_alpha': 0.17975496372783134, 'reg_lambda': 0.532576559712103}),
    "2F咽拭子": ("DecisionTree", {'max_depth': 18, 'min_samples_split': 6, 'min_samples_leaf': 3, 'max_features': 0.6405074202323546}),
    "2F检验科-门诊末梢": ("RandomForest", {'n_estimators': 441, 'max_depth': 26, 'min_samples_leaf': 3, 'max_features': 0.3984879476706409, 'bootstrap': True}),
    "2F检验科-急诊末梢": ("SVR", {'C': 2.8110563835485087, 'epsilon': 0.5359881736867094, 'gamma': 'auto'}),
    "CT-门诊+急诊": ("XGBoost", {'n_estimators': 355, 'max_depth': 6, 'learning_rate': 0.01853665883041131, 'subsample': 0.8450569000800362, 'colsample_bytree': 0.7550732724828378, 'gamma': 0.2288611748518899, 'min_child_weight': 8, 'reg_alpha': 0.28331316993642636, 'reg_lambda': 0.670456025846887}),
    "超声-门诊": ("LightGBM", {'max_depth': 10, 'num_leaves': 32, 'learning_rate': 0.05376724985872417, 'feature_fraction': 0.7062344006015898, 'bagging_fraction': 0.5579990604600252}),
    "MRI-门诊+急诊": ("RandomForest", {'n_estimators': 434, 'max_depth': 29, 'min_samples_leaf': 5, 'max_features': 0.3001632304422872, 'bootstrap': False}),
    "发热楼检验科": ("KNN", {'n_neighbors': 10, 'weights': 'distance', 'p': 2}),
    "1F咽拭子": ("RandomForest", {'n_estimators': 357, 'max_depth': 14, 'min_samples_leaf': 1, 'max_features': 0.4021767833180502, 'bootstrap': True}),
    "X线-门诊+急诊": ("RandomForest", {'n_estimators': 167, 'max_depth': 15, 'min_samples_leaf': 4, 'max_features': 0.43521568067567795, 'bootstrap': False}),
}

feature_cols = ["Month", "Day", "Hour", "the day of week", "the number of queuing patient", "Arrival rate"]
target_col = "Queuing time"

# Data loading
data_folder = "/home/pumc/tangrui_zy/tfenv/WT prediction"
file_list = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')]

for file_name in file_list:
    try:
        file_path = os.path.join(data_folder, file_name)
        df = pd.read_excel(file_path)
        df = df.dropna(subset=feature_cols + [target_col])
        X = df[feature_cols]
        y = df[target_col]

        if len(X) < 10:
            print(f"⚠️ 数据过少，跳过文件：{file_name}")
            continue

        # Model match
        model_name, params = None, None
        for key in model_configs:
            if key in file_name:
                model_name, params = model_configs[key]
                break
        if not model_name:
            print(f"⚠️ 未匹配模型: {file_name}")
            continue

        # Model initialization
        if model_name == "RandomForest":
            model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
        elif model_name == "SVR":
            model = SVR(**params)
        elif model_name == "XGBoost":
            model = XGBRegressor(**params, tree_method='hist', random_state=42)
        elif model_name == "LightGBM":
            model = LGBMRegressor(**params, device='gpu', random_state=42)
        elif model_name == "DecisionTree":
            model = DecisionTreeRegressor(**params, random_state=42)
        elif model_name == "KNN":
            model = KNeighborsRegressor(**params)
        else:
            print(f"❌ 不支持的模型类型：{model_name}")
            continue

        model.fit(X, y)

        # Sampling strategy defination
        sample_sizes = [500, 1000, 2000] if len(X) >= 500 else [len(X)]

        print(f"\n📄 {file_name} | 模型: {model_name} | 总数据量: {len(X)}")
        for sample_size in sample_sizes:
            X_sample = X.sample(n=sample_size, random_state=42)
            if model_name in ["XGBoost", "LightGBM", "RandomForest", "DecisionTree"]:
                explainer = shap.Explainer(model, X_sample)
            else:
                background = shap.kmeans(X_sample, 500)
                explainer = shap.KernelExplainer(model.predict, background)

            shap_values = explainer(X_sample)
            mean_shap = np.abs(shap_values.values).mean(axis=0)

            print(f"  🔹 Sample Size: {sample_size}")
            for feature, val in zip(feature_cols, mean_shap):
                print(f"    {feature}: {val:.4f}")
        print("-" * 60)

    except Exception as e:
        print(f"❌ 处理文件 {file_name} 时出错：{e}")
