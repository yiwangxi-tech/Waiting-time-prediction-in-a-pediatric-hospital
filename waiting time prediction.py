import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from scipy.stats import uniform, randint
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# 读取 Excel 文件，以1F咽拭子场景为例
df = pd.read_excel('1F咽拭子.xlsx')

# 如果是影像检查，则将‘分诊签到时间’替换为‘签到/登记时间’
df_rd = df.drop_duplicates(subset=['病人ID', 'Visit times', '分诊签到时间'], keep='first')

# 提取时间信息并将该列转换为日期时间类型，如果是影像检查，则将‘分诊签到时间’替换为‘签到/登记时间’，将‘采样时间’替换为‘上机检查时间’
df_rd['分诊签到时间'] = pd.to_datetime(df_rd['分诊签到时间'])
df_rd['采样时间'] = pd.to_datetime(df_rd['采样时间'])

# 按照时间先后顺序整理Excel文件
df_rd_rank = df_rd.sort_values(by='分诊签到时间', ascending=True)

# 提取月、日、时信息并添加到新的列中
df_rd_rank['month'] = df_rd_rank[time_column].dt.month
df_rd_rank['day'] = df_rd_rank[time_column].dt.day
df_rd_rank['hour'] = df_rd_rank[time_column].dt.hour
df_rd_rank['week'] = df_rd_rank[time_column].dt.weekday

# 按照“month”、“day”、“hour”进行统计每小时达到的人数
grouped = df_rd_rank.groupby(['month', 'day', 'hour']).size().reset_index(name='arrival rate')

# 计算排队人数，如果是影像检查，则将‘分诊签到时间’替换为‘签到/登记时间’，将‘采样时间’替换为‘上机检查时间’
for i in range(len(df_rd_rank)):
    current_sign_in_time = df_rd_rank.loc[i, '分诊签到时间']
    start_index = max(0, i - 10)   # 需动态调整，在10-100之间，根据队列场景的拥挤程度
    count = (df_rd_rank.loc[start_index:i-1, '采样时间'] > current_sign_in_time).sum()
    df_rd_rank.loc[i, 'the number of queuing patient'] = count

# 预处理数据
y = df["waiting time"]
X = df[["month", "day", "hour", "week", "the number of queuing patient", "arrival rate"]]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 随机森林
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

# XGBoost
xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(X_train, y_train)
xgb_reg_predictions = xgb_reg.predict(X_test)

# KNN
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

# MLP
mlp = MLPRegressor()
mlp.fit(X_train, y_train)
mlp_predictions = mlp.predict(X_test)

# SVM
svm = SVR()
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)

# Linear
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
lr_reg_predictions = lr_reg.predict(X_test)

# 弹性网络回归
en_reg = ElasticNet()
en_reg.fit(X_train, y_train)
en_reg_predictions = en_reg.predict(X_test)

# CART
cart_reg = DecisionTreeRegressor()
cart_reg.fit(X_train, y_train)
cart_reg_predictions = cart_reg.predict(X_test)

# 袋装法回归（Bagging）
bagging_reg = BaggingRegressor()
bagging_reg.fit(X_train, y_train)
bagging_reg_predictions = bagging_reg.predict(X_test)

# LightGBM
lgb_reg = lgb.LGBMRegressor()
lgb_reg.fit(X_train, y_train)
lgb_reg_predictions = lgb_reg.predict(X_test)

# 评估模型
def evaluate_model(y_test, predictions, model_name):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {"Model": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# 以LightGBM为例，使用随机搜索寻找最优超参数
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 20),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'num_leaves': randint(20, 200),
    'min_child_samples': randint(5, 100),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
    'min_split_gain': uniform(0, 0.5),
    'bagging_freq': randint(1, 10),
    'feature_fraction': uniform(0.5, 0.5),
    'boosting_type': ['gbdt', 'dart', 'goss']
}

# 初始化 LightGBM 回归模型
lgb_reg = LGBMRegressor(random_state=0, verbose=-1)

# 使用随机搜索法进行超参数调整
random_search = RandomizedSearchCV(
    estimator=lgb_reg,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=0,
    n_jobs=-1
)

# 拟合数据
random_search.fit(X_train, y_train)

# 输出最佳超参数组合
print(f"Best parameters found: {random_search.best_params_}")

# 使用最佳参数的模型进行预测
best_lgb_reg = random_search.best_estimator_
lgb_predictions = best_lgb_reg.predict(X_test)
lgb_mae = mean_absolute_error(y_test, lgb_predictions)
lgb_mse = mean_squared_error(y_test, lgb_predictions)
lgb_rmse = root_mean_squared_error(y_test, lgb_predictions)
lgb_r2 = r2_score(y_test, lgb_predictions)
print(f"Mean Absolute Error (MAE): {lgb_mae}")
print(f"Mean Squared Error (MSE): {lgb_mse}")
print(f"Root Mean Squared Error (MAE): {lgb_rmse}")
print(f"R² Score: {lgb_r2}")