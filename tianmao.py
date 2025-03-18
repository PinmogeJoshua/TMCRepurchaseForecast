# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# 加载数据
user_log = pd.read_csv('user_log_format1.csv')
user_info = pd.read_csv('user_info_format1.csv')
train = pd.read_csv('train_format1.csv')
test = pd.read_csv('test_format1.csv')

# 分离特征和标签
X = train.drop(columns=['user_id', 'merchant_id', 'label'])  # 去除不必要列
y = train['label']

# 提取用户-商家交互特征
user_merchant_interactions = user_log.groupby(['user_id', 'seller_id']).agg(
    clicks=('action_type', lambda x: (x == 0).sum()),
    add_to_cart=('action_type', lambda x: (x == 1).sum()),
    purchases=('action_type', lambda x: (x == 2).sum()),
    favorites=('action_type', lambda x: (x == 3).sum()),
    total_interactions=('action_type', 'count'),
    days_active=('time_stamp', 'nunique')
).reset_index()

# 统计用户购买行为特征
user_stats = user_log.groupby('user_id').agg(
    total_purchases=('action_type', lambda x: (x == 2).sum()),
    total_clicks=('action_type', lambda x: (x == 0).sum()),
    total_add_to_cart=('action_type', lambda x: (x == 1).sum()),
    total_favorites=('action_type', lambda x: (x == 3).sum()),
    unique_merchants=('seller_id', 'nunique'),
    total_interactions=('action_type', 'count'),
    days_active=('time_stamp', 'nunique')
).reset_index()

# 提取商家特征
merchant_stats = user_log.groupby('seller_id').agg(
    merchant_total_purchases=('action_type', lambda x: (x == 2).sum()),
    merchant_total_clicks=('action_type', lambda x: (x == 0).sum()),
    merchant_total_add_to_cart=('action_type', lambda x: (x == 1).sum()),
    merchant_total_favorites=('action_type', lambda x: (x == 3).sum()),
    merchant_unique_users=('user_id', 'nunique'),
    merchant_total_interactions=('action_type', 'count'),
    merchant_days_active=('time_stamp', 'nunique')
).reset_index()

# 计算用户的点击转化率、加购转化率等
user_stats['click_purchase_ratio'] = user_stats['total_clicks'] / (user_stats['total_purchases'] + 1)
user_stats['add_to_cart_purchase_ratio'] = user_stats['total_add_to_cart'] / (user_stats['total_purchases'] + 1)
user_stats['click_favorite_ratio'] = user_stats['total_clicks'] / (user_stats['total_favorites'] + 1)
user_stats['interaction_per_merchant'] = user_stats['total_interactions'] / (user_stats['unique_merchants'] + 1)
user_stats['interaction_per_day'] = user_stats['total_interactions'] / (user_stats['days_active'] + 1)

# 计算商家的点击转化率、加购转化率等
merchant_stats['merchant_click_purchase_ratio'] = merchant_stats['merchant_total_clicks'] / (merchant_stats['merchant_total_purchases'] + 1)
merchant_stats['merchant_add_to_cart_purchase_ratio'] = merchant_stats['merchant_total_add_to_cart'] / (merchant_stats['merchant_total_purchases'] + 1)
merchant_stats['merchant_click_favorite_ratio'] = merchant_stats['merchant_total_clicks'] / (merchant_stats['merchant_total_favorites'] + 1)
merchant_stats['merchant_interaction_per_user'] = merchant_stats['merchant_total_interactions'] / (merchant_stats['merchant_unique_users'] + 1)
merchant_stats['merchant_interaction_per_day'] = merchant_stats['merchant_total_interactions'] / (merchant_stats['merchant_days_active'] + 1)

# 重命名列以匹配 train 和 test 数据框中的列名
user_merchant_interactions.rename(columns={'seller_id': 'merchant_id'}, inplace=True)

# 确保 train 和 test 数据框中有 merchant_id 列
if 'merchant_id' not in train.columns:
    train['merchant_id'] = train['merchant_id']

if 'merchant_id' not in test.columns:
    test['merchant_id'] = test['merchant_id']

# 合并用户画像数据
train_data = train.merge(user_info, on='user_id', how='left')
test_data = test.merge(user_info, on='user_id', how='left')

# 合并用户-商家交互特征
train_data = train_data.merge(user_merchant_interactions, on=['user_id', 'merchant_id'], how='left').fillna(0)
test_data = test_data.merge(user_merchant_interactions, on=['user_id', 'merchant_id'], how='left').fillna(0)

# 合并用户统计特征
train_data = train_data.merge(user_stats, on='user_id', how='left').fillna(0)
test_data = test_data.merge(user_stats, on='user_id', how='left').fillna(0)

# 将性别和年龄范围处理为数值
train_data['gender'] = train_data['gender'].fillna(2).astype(int)
train_data['age_range'] = train_data['age_range'].fillna(-1).astype(int)
test_data['gender'] = test_data['gender'].fillna(2).astype(int)
test_data['age_range'] = test_data['age_range'].fillna(-1).astype(int)

# 分离特征和标签
X = train_data.drop(columns=['user_id', 'merchant_id', 'label'])  # 去除不必要列
y = train_data['label']

# 拆分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置XGBoost参数
xgb_params = {
    'objective': 'binary:logistic',
    'max_depth': 8,
    'learning_rate': 0.01,    # 降低学习率
    'scale_pos_weight': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1,              # L2 正则化
    'alpha': 0.5,             # L1 正则化
    'eval_metric': 'auc',
    'random_state': 42
}

# 转换数据格式为XGBoost的DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# 训练XGBoost模型
watchlist = [(dtrain, 'train'), (dval, 'eval')]
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=2000, early_stopping_rounds=50, evals=watchlist, verbose_eval=10)

# 评估XGBoost模型
val_pred_xgb = xgb_model.predict(dval)
auc_score_xgb = roc_auc_score(y_val, val_pred_xgb)
print(f'XGBoost验证集AUC得分: {auc_score_xgb}')

# 定义随机森林参数
rf_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

# 训练随机森林模型
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_train, y_train)

# 评估随机森林模型
val_pred_rf = rf_model.predict_proba(X_val)[:, 1]
auc_score_rf = roc_auc_score(y_val, val_pred_rf)
print(f'随机森林验证集AUC得分: {auc_score_rf}')

# 训练逻辑回归模型
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# 评估逻辑回归模型
val_pred_lr = lr_model.predict_proba(X_val)[:, 1]
auc_score_lr = roc_auc_score(y_val, val_pred_lr)
print(f'逻辑回归验证集AUC得分: {auc_score_lr}')

# 使用投票分类器结合多个模型
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb.XGBClassifier(**xgb_params)),
        ('rf', RandomForestClassifier(**rf_params)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ],
    voting='soft'
)

# 训练投票分类器
voting_clf.fit(X_train, y_train)

# 评估投票分类器
val_pred_voting = voting_clf.predict_proba(X_val)[:, 1]
auc_score_voting = roc_auc_score(y_val, val_pred_voting)
print(f'投票分类器验证集AUC得分: {auc_score_voting}')

# 预测测试集
dtest = xgb.DMatrix(test_data.drop(columns=['user_id', 'merchant_id', 'prob'], errors='ignore'))
test_data = test_data.drop(columns=['prob'], errors='ignore')
test_data['prob'] = voting_clf.predict_proba(test_data.drop(columns=['user_id', 'merchant_id'], errors='ignore'))[:, 1]

# 保存结果
result = test_data[['user_id', 'merchant_id', 'prob']]
result.to_csv('prediction2.csv', index=False)
print("预测结果已保存至 prediction2.csv")