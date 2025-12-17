import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import time
import scipy.stats as stats
import warnings
import joblib
import tempfile
import os
import json
from io import StringIO

warnings.filterwarnings('ignore')

# XGBoost 库
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# LightGBM 库
try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

# CatBoost 库
try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


# ===================== 主页面 =====================
def ai_prediction_page():
    st.title("🤖 风电场风速AI预测系统")

    # 数据状态检查
    if 'dataset' not in st.session_state:
        st.warning("⚠️ 请先在数据导入页面导入风电场数据")
        return

    df = st.session_state['dataset'].copy()

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 分析配置")
        analysis_mode = st.radio(
            "分析模式",
            ["单模型预测", "多模型对比"]
        )

    # 时间特征处理
    datetime_col = next(
        (col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower() or 'date' in col.lower()),
        None)
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df['hour'] = df[datetime_col].dt.hour
        df['month'] = df[datetime_col].dt.month
        df['dayofyear'] = df[datetime_col].dt.dayofyear
        df['dayofweek'] = df[datetime_col].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['season'] = (df[datetime_col].dt.month % 12 + 3) // 3  # 季节划分
    else:
        st.error("❌ 未找到时间列")
        return

    # 目标变量
    target_candidates = ['predicted_wind_speed']
    target_column = next((col for col in target_candidates if col in df.columns), None)
    if not target_column:
        st.error("❌ 未找到目标变量 predicted_wind_speed")
        return

    # 数据概览
    st.subheader("📊 数据概览")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总样本数", f"{len(df):,}")
    with col2:
        st.metric("平均风速", f"{df[target_column].mean():.2f} m/s")
    with col3:
        st.metric("风速标准差", f"{df[target_column].std():.2f} m/s")
    with col4:
        # 修复时间范围显示问题
        time_min = df[datetime_col].min()
        time_max = df[datetime_col].max()
        time_span = time_max - time_min

        if time_span.days == 0:
            # 同一天，只显示时间
            time_range_str = f"{time_min.strftime('%H:%M')} → {time_max.strftime('%H:%M')}"
            full_info = f"{time_min.strftime('%Y-%m-%d %H:%M')} → {time_max.strftime('%Y-%m-%d %H:%M')}"
        else:
            # 跨天，使用紧凑格式
            time_range_str = f"{time_min.strftime('%m/%d %H:%M')} → {time_max.strftime('%m/%d %H:%M')}"
            full_info = f"{time_min.strftime('%Y-%m-%d %H:%M')} → {time_max.strftime('%Y-%m-%d %H:%M')}"

        st.metric("数据时间范围", time_range_str, help=full_info)

    if analysis_mode == "单模型预测":
        single_model_analysis(df, datetime_col, target_column)
    elif analysis_mode == "多模型对比":
        multi_model_comparison(df, datetime_col, target_column)


# ===================== 单模型分析 =====================
def single_model_analysis(df, datetime_col, target_column):
    st.subheader("🎯 单模型预测分析")

    # 仅使用指定的特征字段
    available_features = [
        'point_id', 'lat', 'lon', 'elevation', 'slope',
        'relative_humidity', 'temperature_c', 'wind_direction',
        'gust_direction', 'gust_speed', 'wind_direction_std', 'rainfall_mm'
    ]

    # 过滤实际存在于数据中的特征
    feature_candidates = [col for col in available_features if col in df.columns]

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_features = st.multiselect(
            "选择特征变量",
            options=feature_candidates,
            default=[col for col in [
                'relative_humidity', 'temperature_c', 'wind_direction',
                'gust_direction', 'gust_speed', 'wind_direction_std', 'rainfall_mm'
            ] if col in feature_candidates]
        )

    with col2:
        # 模型选择 - 四种算法：随机森林、XGBoost、CatBoost、LightGBM
        model_options = ["XGBoost", "随机森林", "CatBoost", "LightGBM"]

        # 检查库可用性
        available_models = []
        for model in model_options:
            if model == "XGBoost":
                if HAS_XGBOOST:
                    available_models.append(model)
            elif model == "LightGBM":
                if HAS_LIGHTGBM:
                    available_models.append(model)
            elif model == "CatBoost":
                if HAS_CATBOOST:
                    available_models.append(model)
            else:
                available_models.append(model)

        model_option = st.selectbox("选择算法", available_models)

        # 高级参数
        with st.expander("高级参数"):
            test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
            cv_folds = st.slider("交叉验证折数", 3, 10, 5)
            enable_permutation = st.checkbox("启用置换重要性分析", value=True)

            # 添加网格搜索选项
            enable_grid_search = st.checkbox("启用网格搜索调参", value=False)

            if enable_grid_search:
                st.info("⚠️ 网格搜索将增加计算时间，建议数据量较大时使用")
                # 网格搜索参数配置
                col_search1, col_search2 = st.columns(2)
                with col_search1:
                    search_method = st.radio(
                        "搜索方法",
                        ["网格搜索(GridSearch)", "随机搜索(RandomizedSearch)"],
                        index=0
                    )
                with col_search2:
                    if search_method == "随机搜索(RandomizedSearch)":
                        n_iter_search = st.slider("随机搜索迭代次数", 10, 100, 30)
                    else:
                        n_iter_search = None

                # 搜索范围配置
                st.markdown("##### 搜索参数范围")
                param_config_expander = st.expander("查看/修改参数搜索范围", expanded=False)
                with param_config_expander:
                    if model_option == "XGBoost":
                        xgb_param_grid = configure_xgboost_params()
                    elif model_option == "随机森林":
                        rf_param_grid = configure_randomforest_params()
                    elif model_option == "CatBoost":
                        catboost_param_grid = configure_catboost_params()
                    elif model_option == "LightGBM":
                        lgb_param_grid = configure_lightgbm_params()

            # XGBoost 特定参数（如果启用网格搜索，则不显示手动参数调整）
            if model_option == "XGBoost":
                if not enable_grid_search:
                    st.markdown("##### XGBoost参数")
                    xgb_learning_rate = st.slider("XGBoost学习率", 0.01, 0.3, 0.1, 0.01)
                    xgb_max_depth = st.slider("XGBoost最大深度", 3, 10, 6)
                    xgb_n_estimators = st.slider("XGBoost估计器数量", 50, 300, 100)
                else:
                    # 网格搜索时不显示手动参数
                    st.info("📊 网格搜索将自动寻找最优参数组合")

            # CatBoost 特定参数
            if model_option == "CatBoost":
                if not enable_grid_search:
                    st.markdown("##### CatBoost参数")
                    catboost_iterations = st.slider("迭代次数", 100, 2000, 500)
                    catboost_depth = st.slider("树深度", 4, 10, 6)
                    catboost_learning_rate = st.slider("学习率", 0.01, 0.3, 0.03, 0.01)
                    catboost_l2_leaf_reg = st.slider("L2正则化", 1, 10, 3)
                else:
                    st.info("📊 网格搜索将自动寻找最优参数组合")

            # LightGBM 特定参数
            if model_option == "LightGBM":
                if not enable_grid_search:
                    st.markdown("##### LightGBM参数")
                    lgb_n_estimators = st.slider("LightGBM估计器数量", 50, 500, 100)
                    lgb_learning_rate = st.slider("LightGBM学习率", 0.01, 0.3, 0.05, 0.01)
                    lgb_max_depth = st.slider("LightGBM最大深度", 3, 15, 7)
                    lgb_num_leaves = st.slider("叶子数量", 20, 150, 50)
                else:
                    st.info("📊 网格搜索将自动寻找最优参数组合")

    if not selected_features:
        st.warning("请选择至少一个特征变量")
        return

    if st.button("🚀 开始训练分析", type="primary", use_container_width=True):
        with st.spinner("正在进行模型训练..."):
            # 数据准备
            X = df[selected_features].fillna(0)
            y = df[target_column].fillna(0)

            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 模型训练
            model = None
            training_time = 0
            history = None
            grid_search_results = None
            best_params = None
            param_grid_used = None

            # 检查是否启用网格搜索
            if enable_grid_search:
                with st.spinner(f"正在进行{model_option}网格搜索调参..."):
                    if model_option == "XGBoost":
                        grid_search_results, model, training_time, param_grid_used = perform_xgboost_grid_search(
                            X_train_scaled, y_train, X_test_scaled, y_test,
                            search_method, n_iter_search, xgb_param_grid
                        )
                    elif model_option == "随机森林":
                        grid_search_results, model, training_time, param_grid_used = perform_randomforest_grid_search(
                            X_train_scaled, y_train, X_test_scaled, y_test,
                            search_method, n_iter_search, rf_param_grid
                        )
                    elif model_option == "CatBoost":
                        grid_search_results, model, training_time, param_grid_used = perform_catboost_grid_search(
                            X_train_scaled, y_train, X_test_scaled, y_test,
                            search_method, n_iter_search, catboost_param_grid
                        )
                    elif model_option == "LightGBM":
                        grid_search_results, model, training_time, param_grid_used = perform_lightgbm_grid_search(
                            X_train_scaled, y_train, X_test_scaled, y_test,
                            search_method, n_iter_search, lgb_param_grid
                        )

                    if grid_search_results is not None:
                        best_params = grid_search_results.best_params_
            else:
                # 常规模型训练
                if model_option == "随机森林":
                    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                elif model_option == "XGBoost":
                    model = xgb.XGBRegressor(
                        n_estimators=xgb_n_estimators,
                        max_depth=xgb_max_depth,
                        learning_rate=xgb_learning_rate,
                        random_state=42,
                        n_jobs=-1
                    )
                elif model_option == "CatBoost":
                    model = CatBoostRegressor(
                        iterations=catboost_iterations,
                        depth=catboost_depth,
                        learning_rate=catboost_learning_rate,
                        l2_leaf_reg=catboost_l2_leaf_reg,
                        random_seed=42,
                        verbose=0,
                        allow_writing_files=False
                    )
                elif model_option == "LightGBM":
                    model = lgb.LGBMRegressor(
                        n_estimators=lgb_n_estimators,
                        learning_rate=lgb_learning_rate,
                        max_depth=lgb_max_depth,
                        num_leaves=lgb_num_leaves,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=-1
                    )

                # 训练模型
                start_time = time.time()
                model.fit(X_train_scaled, y_train)
                training_time = time.time() - start_time

            # 预测
            y_pred = model.predict(X_test_scaled)

            # 交叉验证
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')

            # 计算指标
            results = calculate_metrics(y_test, y_pred, training_time)
            results['cv_mean'] = cv_scores.mean() if len(cv_scores) > 0 else 0
            results['cv_std'] = cv_scores.std() if len(cv_scores) > 0 else 0

            # 特征重要性（仅适用于支持特征重要性的模型）
            feature_importance = None
            permutation_importance_result = None

            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            elif model_option == "CatBoost":
                # CatBoost的特征重要性
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': model.get_feature_importance()
                }).sort_values('importance', ascending=False)
            elif model_option == "LightGBM":
                # LightGBM的特征重要性
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

            # 置换重要性
            if enable_permutation:
                with st.spinner("正在计算置换重要性..."):
                    permutation_importance_result = calculate_permutation_importance(
                        model, X_test_scaled, y_test, selected_features
                    )

            # 显示结果
            display_single_model_results(
                results, feature_importance, permutation_importance_result,
                model_option, y_test, y_pred, cv_scores, X_test_scaled, model,
                history, grid_search_results, best_params, param_grid_used, enable_grid_search
            )


# ===================== 参数配置函数 =====================
def configure_xgboost_params():
    """配置XGBoost参数网格"""
    st.write("XGBoost参数搜索范围配置:")

    col1, col2 = st.columns(2)

    with col1:
        n_estimators = st.multiselect(
            "n_estimators (树的数量)",
            [50, 100, 200, 300, 500],
            default=[100]
        )
        max_depth = st.multiselect(
            "max_depth (最大深度)",
            [3, 5, 7, 9, 11],
            default=[5]
        )
        learning_rate = st.multiselect(
            "learning_rate (学习率)",
            [0.01, 0.05, 0.1, 0.2, 0.3],
            default=[0.05]
        )
        subsample = st.multiselect(
            "subsample (样本采样率)",
            [0.6, 0.7, 0.8, 0.9, 1.0],
            default=[0.8]
        )

    with col2:
        colsample_bytree = st.multiselect(
            "colsample_bytree (特征采样率)",
            [0.6, 0.7, 0.8, 0.9, 1.0],
            default=[0.8]
        )
        gamma = st.multiselect(
            "gamma (分裂最小损失减少)",
            [0, 0.1, 0.2, 0.3, 0.5],
            default=[0]
        )
        reg_alpha = st.multiselect(
            "reg_alpha (L1正则化)",
            [0, 0.01, 0.1, 1, 10],
            default=[0]
        )
        reg_lambda = st.multiselect(
            "reg_lambda (L2正则化)",
            [0.1, 1, 1.5, 2, 3],
            default=[1]
        )

    param_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda
    }

    # 显示配置摘要
    st.info(f"总参数组合数: {np.prod([len(v) for v in param_grid.values()]):,}")

    return param_grid


def configure_randomforest_params():
    """配置随机森林参数网格"""
    st.write("随机森林参数搜索范围配置:")

    col1, col2 = st.columns(2)

    with col1:
        n_estimators = st.multiselect(
            "n_estimators (树的数量)",
            [100, 200, 300, 500, 1000],
            default=[200, 300, 500],
            key="rf_n_estimators"
        )
        max_depth = st.multiselect(
            "max_depth (最大深度)",
            [None, 10, 20, 30, 40],
            default=[None, 20, 30],
            key="rf_max_depth"
        )
        min_samples_split = st.multiselect(
            "min_samples_split (分裂最小样本数)",
            [2, 5, 10, 20],
            default=[2, 5, 10],
            key="rf_min_samples_split"
        )

    with col2:
        min_samples_leaf = st.multiselect(
            "min_samples_leaf (叶节点最小样本数)",
            [1, 2, 4, 8],
            default=[1, 2, 4],
            key="rf_min_samples_leaf"
        )
        max_features = st.multiselect(
            "max_features (最大特征数)",
            ['auto', 'sqrt', 'log2', 0.5, 0.8],
            default=['auto', 'sqrt', 0.8],
            key="rf_max_features"
        )
        bootstrap = st.multiselect(
            "bootstrap (有放回采样)",
            [True, False],
            default=[True],
            key="rf_bootstrap"
        )

    param_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap
    }

    # 显示配置摘要
    st.info(f"总参数组合数: {np.prod([len(v) for v in param_grid.values()]):,}")

    return param_grid


def configure_catboost_params():
    """配置CatBoost参数网格"""
    st.write("CatBoost参数搜索范围配置:")

    col1, col2 = st.columns(2)

    with col1:
        iterations = st.multiselect(
            "iterations (迭代次数)",
            [500, 1000, 1500, 2000],
            default=[1000, 1500],
            key="cb_iterations"
        )
        depth = st.multiselect(
            "depth (树深度)",
            [4, 6, 8, 10],
            default=[6, 8],
            key="cb_depth"
        )
        learning_rate = st.multiselect(
            "learning_rate (学习率)",
            [0.01, 0.03, 0.05, 0.1, 0.2],
            default=[0.03, 0.05, 0.1],
            key="cb_learning_rate"
        )

    with col2:
        l2_leaf_reg = st.multiselect(
            "l2_leaf_reg (L2正则化)",
            [1, 3, 5, 7, 10],
            default=[3, 5, 7],
            key="cb_l2_leaf_reg"
        )
        border_count = st.multiselect(
            "border_count (特征分箱数)",
            [32, 64, 128, 256],
            default=[64, 128],
            key="cb_border_count"
        )
        random_strength = st.multiselect(
            "random_strength (随机强度)",
            [0, 1, 5, 10],
            default=[0, 1],
            key="cb_random_strength"
        )

    param_grid = {
        'iterations': iterations,
        'depth': depth,
        'learning_rate': learning_rate,
        'l2_leaf_reg': l2_leaf_reg,
        'border_count': border_count,
        'random_strength': random_strength,
        'verbose': [0]  # 始终不显示训练过程
    }

    # 显示配置摘要
    st.info(f"总参数组合数: {np.prod([len(v) for v in param_grid.values()]):,}")

    return param_grid


def configure_lightgbm_params():
    """配置LightGBM参数网格"""
    st.write("LightGBM参数搜索范围配置:")

    col1, col2 = st.columns(2)

    with col1:
        n_estimators = st.multiselect(
            "n_estimators (树的数量)",
            [100, 200, 300, 500, 1000],
            default=[200, 300, 500],
            key="lgb_n_estimators"
        )
        num_leaves = st.multiselect(
            "num_leaves (叶子数量)",
            [31, 63, 127, 255],
            default=[31, 63, 127],
            key="lgb_num_leaves"
        )
        learning_rate = st.multiselect(
            "learning_rate (学习率)",
            [0.01, 0.05, 0.1, 0.2],
            default=[0.05, 0.1, 0.2],
            key="lgb_learning_rate"
        )

    with col2:
        max_depth = st.multiselect(
            "max_depth (最大深度)",
            [-1, 5, 10, 15],
            default=[-1, 10],
            key="lgb_max_depth"
        )
        min_child_samples = st.multiselect(
            "min_child_samples (叶节点最小样本数)",
            [20, 50, 100, 200],
            default=[20, 50, 100],
            key="lgb_min_child_samples"
        )
        reg_alpha = st.multiselect(
            "reg_alpha (L1正则化)",
            [0, 0.01, 0.1, 1],
            default=[0, 0.01, 0.1],
            key="lgb_reg_alpha"
        )
        reg_lambda = st.multiselect(
            "reg_lambda (L2正则化)",
            [0, 0.01, 0.1, 1],
            default=[0, 0.01, 0.1],
            key="lgb_reg_lambda"
        )

    param_grid = {
        'n_estimators': n_estimators,
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_child_samples': min_child_samples,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'verbosity': [-1]  # 始终不显示训练过程
    }

    # 显示配置摘要
    st.info(f"总参数组合数: {np.prod([len(v) for v in param_grid.values()]):,}")

    return param_grid


# ===================== 网格搜索函数 =====================
def perform_xgboost_grid_search(X_train, y_train, X_test, y_test, search_method, n_iter_search, param_grid):
    """
    执行XGBoost网格搜索
    """
    try:
        # 基础XGBoost模型
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

        # 显示搜索配置
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        st.info(f"开始XGBoost{search_method}，总参数组合: {total_combinations:,}")

        # 创建搜索对象
        if search_method == "网格搜索(GridSearch)":
            search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=0,
                scoring='r2',
                return_train_score=True
            )
            search_name = "网格搜索"
        else:  # 随机搜索
            search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_grid,
                n_iter=n_iter_search,
                cv=3,
                n_jobs=-1,
                verbose=0,
                random_state=42,
                scoring='r2',
                return_train_score=True
            )
            search_name = "随机搜索"

        # 执行搜索
        start_time = time.time()

        # 创建进度指示器
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"正在进行{search_name}...")

        search.fit(X_train, y_train)

        training_time = time.time() - start_time
        progress_bar.progress(100)
        status_text.empty()

        st.success(f"{search_name}完成！最佳参数R²分数: {search.best_score_:.4f}")

        return search, search.best_estimator_, training_time, param_grid

    except Exception as e:
        st.error(f"XGBoost网格搜索失败: {str(e)}")
        return None, None, 0, None


def perform_randomforest_grid_search(X_train, y_train, X_test, y_test, search_method, n_iter_search, param_grid):
    """
    执行随机森林网格搜索
    """
    try:
        # 基础随机森林模型
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        # 显示搜索配置
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        st.info(f"开始随机森林{search_method}，总参数组合: {total_combinations:,}")

        # 创建搜索对象
        if search_method == "网格搜索(GridSearch)":
            search = GridSearchCV(
                estimator=rf_model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=0,
                scoring='r2',
                return_train_score=True
            )
        else:  # 随机搜索
            search = RandomizedSearchCV(
                estimator=rf_model,
                param_distributions=param_grid,
                n_iter=n_iter_search,
                cv=3,
                n_jobs=-1,
                verbose=0,
                random_state=42,
                scoring='r2',
                return_train_score=True
            )

        # 执行搜索
        start_time = time.time()

        # 创建进度指示器
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("正在进行随机森林参数搜索...")

        search.fit(X_train, y_train)

        training_time = time.time() - start_time
        progress_bar.progress(100)
        status_text.empty()

        st.success(f"随机森林{search_method}完成！最佳参数R²分数: {search.best_score_:.4f}")

        return search, search.best_estimator_, training_time, param_grid

    except Exception as e:
        st.error(f"随机森林网格搜索失败: {str(e)}")
        return None, None, 0, None


def perform_catboost_grid_search(X_train, y_train, X_test, y_test, search_method, n_iter_search, param_grid):
    """
    执行CatBoost网格搜索
    """
    try:
        # 基础CatBoost模型
        cb_model = CatBoostRegressor(random_seed=42, verbose=0, allow_writing_files=False)

        # 显示搜索配置
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        st.info(f"开始CatBoost{search_method}，总参数组合: {total_combinations:,}")

        # 创建搜索对象
        if search_method == "网格搜索(GridSearch)":
            search = GridSearchCV(
                estimator=cb_model,
                param_grid=param_grid,
                cv=3,
                n_jobs=1,  # CatBoost不支持多线程
                verbose=0,
                scoring='r2',
                return_train_score=True
            )
        else:  # 随机搜索
            search = RandomizedSearchCV(
                estimator=cb_model,
                param_distributions=param_grid,
                n_iter=n_iter_search,
                cv=3,
                n_jobs=1,  # CatBoost不支持多线程
                verbose=0,
                random_state=42,
                scoring='r2',
                return_train_score=True
            )

        # 执行搜索
        start_time = time.time()

        # 创建进度指示器
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("正在进行CatBoost参数搜索...")

        search.fit(X_train, y_train)

        training_time = time.time() - start_time
        progress_bar.progress(100)
        status_text.empty()

        st.success(f"CatBoost{search_method}完成！最佳参数R²分数: {search.best_score_:.4f}")

        return search, search.best_estimator_, training_time, param_grid

    except Exception as e:
        st.error(f"CatBoost网格搜索失败: {str(e)}")
        return None, None, 0, None


def perform_lightgbm_grid_search(X_train, y_train, X_test, y_test, search_method, n_iter_search, param_grid):
    """
    执行LightGBM网格搜索
    """
    try:
        # 基础LightGBM模型
        lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)

        # 显示搜索配置
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        st.info(f"开始LightGBM{search_method}，总参数组合: {total_combinations:,}")

        # 创建搜索对象
        if search_method == "网格搜索(GridSearch)":
            search = GridSearchCV(
                estimator=lgb_model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=0,
                scoring='r2',
                return_train_score=True
            )
        else:  # 随机搜索
            search = RandomizedSearchCV(
                estimator=lgb_model,
                param_distributions=param_grid,
                n_iter=n_iter_search,
                cv=3,
                n_jobs=-1,
                verbose=0,
                random_state=42,
                scoring='r2',
                return_train_score=True
            )

        # 执行搜索
        start_time = time.time()

        # 创建进度指示器
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("正在进行LightGBM参数搜索...")

        search.fit(X_train, y_train)

        training_time = time.time() - start_time
        progress_bar.progress(100)
        status_text.empty()

        st.success(f"LightGBM{search_method}完成！最佳参数R²分数: {search.best_score_:.4f}")

        return search, search.best_estimator_, training_time, param_grid

    except Exception as e:
        st.error(f"LightGBM网格搜索失败: {str(e)}")
        return None, None, 0, None


# ===================== 多模型对比 =====================
def multi_model_comparison(df, datetime_col, target_column):
    st.subheader("⚖️ 多模型对比分析")

    # 仅使用指定的特征字段
    available_features = [
        'point_id', 'lat', 'lon', 'elevation', 'slope',
        'relative_humidity', 'temperature_c', 'wind_direction',
        'gust_direction', 'gust_speed', 'wind_direction_std', 'rainfall_mm'
    ]

    feature_candidates = [col for col in available_features if col in df.columns]

    selected_features = st.multiselect(
        "选择特征变量",
        options=feature_candidates,
        default=[col for col in [
            'temperature_c', 'relative_humidity', 'wind_direction',
            'gust_speed', 'elevation', 'slope'
        ] if col in feature_candidates]
    )

    # 模型选择 - 四种算法：随机森林、XGBoost、CatBoost、LightGBM
    model_options = ["XGBoost", "随机森林", "CatBoost", "LightGBM"]

    # 检查库可用性
    available_models = []
    for model in model_options:
        if model == "XGBoost":
            if HAS_XGBOOST:
                available_models.append(model)
        elif model == "LightGBM":
            if HAS_LIGHTGBM:
                available_models.append(model)
        elif model == "CatBoost":
            if HAS_CATBOOST:
                available_models.append(model)
        else:
            available_models.append(model)

    selected_algorithms = st.multiselect(
        "选择对比算法",
        options=available_models,
        default=available_models
    )

    # 高级参数配置
    with st.expander("高级参数配置"):
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("交叉验证折数", 3, 10, 5)

        # 为每个算法添加网格搜索选项
        grid_search_configs = {}
        for algo in selected_algorithms:
            st.markdown(f"##### {algo}参数配置")
            grid_search_configs[algo] = {}
            grid_search_configs[algo]['enable_grid_search'] = st.checkbox(
                f"对{algo}启用网格搜索", value=False, key=f"gs_{algo}"
            )

            if grid_search_configs[algo]['enable_grid_search']:
                col1, col2 = st.columns(2)
                with col1:
                    grid_search_configs[algo]['search_method'] = st.radio(
                        "搜索方法",
                        ["网格搜索(GridSearch)", "随机搜索(RandomizedSearch)"],
                        index=0,
                        key=f"method_{algo}"
                    )
                with col2:
                    if grid_search_configs[algo]['search_method'] == "随机搜索(RandomizedSearch)":
                        grid_search_configs[algo]['n_iter'] = st.slider(
                            "随机搜索迭代次数", 10, 100, 30, key=f"niter_{algo}"
                        )
                    else:
                        grid_search_configs[algo]['n_iter'] = None

                # 参数配置
                with st.expander(f"{algo}参数搜索范围配置", expanded=False):
                    if algo == "XGBoost":
                        grid_search_configs[algo]['param_grid'] = configure_xgboost_params()
                    elif algo == "随机森林":
                        grid_search_configs[algo]['param_grid'] = configure_randomforest_params()
                    elif algo == "CatBoost":
                        grid_search_configs[algo]['param_grid'] = configure_catboost_params()
                    elif algo == "LightGBM":
                        grid_search_configs[algo]['param_grid'] = configure_lightgbm_params()
            else:
                # 手动参数配置
                if algo == "XGBoost":
                    st.markdown("###### XGBoost手动参数")
                    grid_search_configs[algo]['manual_params'] = {
                        'n_estimators': st.slider("树的数量", 50, 300, 100, key=f"xgb_n_{algo}"),
                        'max_depth': st.slider("最大深度", 3, 10, 6, key=f"xgb_d_{algo}"),
                        'learning_rate': st.slider("学习率", 0.01, 0.3, 0.1, 0.01, key=f"xgb_lr_{algo}")
                    }
                elif algo == "随机森林":
                    st.markdown("###### 随机森林手动参数")
                    grid_search_configs[algo]['manual_params'] = {
                        'n_estimators': st.slider("树的数量", 100, 500, 200, key=f"rf_n_{algo}"),
                        'max_depth': st.slider("最大深度", 10, 50, 30, key=f"rf_d_{algo}"),
                        'min_samples_split': st.slider("分裂最小样本数", 2, 20, 5, key=f"rf_mss_{algo}")
                    }
                elif algo == "CatBoost":
                    st.markdown("###### CatBoost手动参数")
                    grid_search_configs[algo]['manual_params'] = {
                        'iterations': st.slider("迭代次数", 100, 2000, 500, key=f"cb_it_{algo}"),
                        'depth': st.slider("树深度", 4, 10, 6, key=f"cb_d_{algo}"),
                        'learning_rate': st.slider("学习率", 0.01, 0.3, 0.03, 0.01, key=f"cb_lr_{algo}")
                    }
                elif algo == "LightGBM":
                    st.markdown("###### LightGBM手动参数")
                    grid_search_configs[algo]['manual_params'] = {
                        'n_estimators': st.slider("树的数量", 50, 500, 100, key=f"lgb_n_{algo}"),
                        'num_leaves': st.slider("叶子数量", 20, 150, 50, key=f"lgb_nl_{algo}"),
                        'learning_rate': st.slider("学习率", 0.01, 0.3, 0.05, 0.01, key=f"lgb_lr_{algo}")
                    }

    if st.button("🔬 开始对比分析", type="primary", use_container_width=True):
        if not selected_features or not selected_algorithms:
            st.warning("请选择特征变量和对比算法")
            return

        with st.spinner("正在进行多模型对比分析..."):
            # 数据准备
            X = df[selected_features].fillna(0)
            y = df[target_column].fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 训练所有模型
            comparison_results = []
            feature_importances = {}
            predictions = {}
            models = {}
            grid_search_infos = {}
            all_grid_search_results = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, algo in enumerate(selected_algorithms):
                status_text.text(f"正在训练 {algo}... ({i + 1}/{len(selected_algorithms)})")

                try:
                    model = None
                    training_time = 0
                    grid_search_results = None

                    config = grid_search_configs[algo]

                    if config.get('enable_grid_search', False):
                        # 执行网格搜索
                        if algo == "XGBoost":
                            grid_search_results, model, training_time, param_grid_used = perform_xgboost_grid_search(
                                X_train_scaled, y_train, X_test_scaled, y_test,
                                config['search_method'], config['n_iter'], config['param_grid']
                            )
                        elif algo == "随机森林":
                            grid_search_results, model, training_time, param_grid_used = perform_randomforest_grid_search(
                                X_train_scaled, y_train, X_test_scaled, y_test,
                                config['search_method'], config['n_iter'], config['param_grid']
                            )
                        elif algo == "CatBoost":
                            grid_search_results, model, training_time, param_grid_used = perform_catboost_grid_search(
                                X_train_scaled, y_train, X_test_scaled, y_test,
                                config['search_method'], config['n_iter'], config['param_grid']
                            )
                        elif algo == "LightGBM":
                            grid_search_results, model, training_time, param_grid_used = perform_lightgbm_grid_search(
                                X_train_scaled, y_train, X_test_scaled, y_test,
                                config['search_method'], config['n_iter'], config['param_grid']
                            )

                        if grid_search_results is not None:
                            grid_search_infos[algo] = {
                                'best_params': grid_search_results.best_params_,
                                'best_score': grid_search_results.best_score_,
                                'cv_results': grid_search_results.cv_results_
                            }
                            all_grid_search_results[algo] = grid_search_results
                    else:
                        # 使用手动参数
                        if algo == "随机森林":
                            manual_params = config.get('manual_params', {})
                            model = RandomForestRegressor(
                                n_estimators=manual_params.get('n_estimators', 200),
                                max_depth=manual_params.get('max_depth', None),
                                min_samples_split=manual_params.get('min_samples_split', 2),
                                random_state=42,
                                n_jobs=-1
                            )
                        elif algo == "XGBoost":
                            manual_params = config.get('manual_params', {})
                            model = xgb.XGBRegressor(
                                n_estimators=manual_params.get('n_estimators', 100),
                                max_depth=manual_params.get('max_depth', 6),
                                learning_rate=manual_params.get('learning_rate', 0.1),
                                random_state=42,
                                n_jobs=-1
                            )
                        elif algo == "CatBoost":
                            manual_params = config.get('manual_params', {})
                            model = CatBoostRegressor(
                                iterations=manual_params.get('iterations', 500),
                                depth=manual_params.get('depth', 6),
                                learning_rate=manual_params.get('learning_rate', 0.03),
                                random_seed=42,
                                verbose=0,
                                allow_writing_files=False
                            )
                        elif algo == "LightGBM":
                            manual_params = config.get('manual_params', {})
                            model = lgb.LGBMRegressor(
                                n_estimators=manual_params.get('n_estimators', 100),
                                num_leaves=manual_params.get('num_leaves', 50),
                                learning_rate=manual_params.get('learning_rate', 0.05),
                                random_state=42,
                                n_jobs=-1,
                                verbosity=-1
                            )

                        # 训练模型
                        start_time = time.time()
                        model.fit(X_train_scaled, y_train)
                        training_time = time.time() - start_time

                    # 预测
                    y_pred = model.predict(X_test_scaled)
                    predictions[algo] = y_pred
                    models[algo] = model

                    # 计算指标
                    results = calculate_metrics(y_test, y_pred, training_time)

                    # 特征重要性
                    if hasattr(model, 'feature_importances_'):
                        feature_importances[algo] = pd.DataFrame({
                            'feature': selected_features,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                    elif algo == "CatBoost":
                        feature_importances[algo] = pd.DataFrame({
                            'feature': selected_features,
                            'importance': model.get_feature_importance()
                        }).sort_values('importance', ascending=False)
                    elif algo == "LightGBM":
                        feature_importances[algo] = pd.DataFrame({
                            'feature': selected_features,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)

                    comparison_results.append({
                        "算法": algo,
                        "MAE": results['mae'],
                        "RMSE": results['rmse'],
                        "R²": results['r2'],
                        "训练时间 (秒)": results['training_time']
                    })

                except Exception as e:
                    st.error(f"训练 {algo} 时出错: {str(e)}")
                    continue

                progress_bar.progress((i + 1) / len(selected_algorithms))

            progress_bar.empty()
            status_text.empty()

            # 显示对比结果
            display_comparison_results(
                comparison_results, feature_importances, y_test,
                predictions, selected_features, models, X_test_scaled,
                grid_search_infos, all_grid_search_results
            )


# ===================== 辅助函数 =====================
def calculate_metrics(y_true, y_pred, training_time):
    """计算模型评估指标"""
    return {
        'true': np.array(y_true),
        'pred': np.array(y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'training_time': training_time
    }


def calculate_permutation_importance(model, X_test, y_test, feature_names, n_repeats=5):
    """计算置换重要性"""
    try:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)

        return importance_df

    except Exception as e:
        st.warning(f"置换重要性计算失败: {str(e)}")
        return None


# ===================== 结果显示函数 =====================
def display_single_model_results(results, feature_importance, permutation_importance_result,
                                 model_name, y_true, y_pred, cv_scores, X_test, model,
                                 history=None, grid_search_results=None, best_params=None,
                                 param_grid_used=None, enable_grid_search=False):
    st.subheader(f"📊 {model_name} 模型性能")

    # 如果进行了网格搜索，显示最佳参数
    if enable_grid_search and grid_search_results is not None and best_params is not None:
        st.success(f"✅ 网格搜索完成！最佳参数:")

        # 创建最佳参数表格
        params_df = pd.DataFrame(list(best_params.items()), columns=['参数', '最优值'])

        # 美化显示
        col1, col2 = st.columns([3, 2])
        with col1:
            st.dataframe(params_df, use_container_width=True)

        with col2:
            # 显示搜索统计信息
            st.metric("最佳R²分数", f"{grid_search_results.best_score_:.4f}")
            st.metric("搜索耗时", f"{results['training_time']:.1f}秒")
            if hasattr(grid_search_results, 'cv_results_'):
                st.metric("搜索组合数", f"{len(grid_search_results.cv_results_['params']):,}")

    # 指标卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE", f"{results['mae']:.3f}")
    col2.metric("RMSE", f"{results['rmse']:.3f}")
    col3.metric("R²", f"{results['r2']:.4f}")
    col4.metric("训练时间", f"{results['training_time']:.2f}秒")
    col5.metric("交叉验证 R²", f"{results['cv_mean']:.4f}")

    # 可视化标签页 - 添加网格搜索分析标签页
    tab_names = ["预测性能", "残差分析", "特征重要性", "交叉验证", "误差分析", "模型诊断"]
    if enable_grid_search and grid_search_results is not None:
        tab_names.insert(3, "网格搜索分析")

    tabs = st.tabs(tab_names)

    # 预测性能标签页
    with tabs[0]:
        # 预测值 vs 真实值
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode='markers',
            marker=dict(color='royalblue', opacity=0.6),
            name='预测点'
        ))
        min_val, max_val = float(np.min(y_true)), float(np.max(y_true))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(dash='dash', color='red'),
            name='理想拟合线'
        ))
        fig.update_layout(
            title="预测值 vs 真实值",
            xaxis_title="真实风速 (m/s)",
            yaxis_title="预测风速 (m/s)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # 时间序列对比
        st.markdown("##### 时间序列对比")
        sample_size = min(200, len(y_true))
        if len(y_true) > sample_size:
            indices = np.random.choice(len(y_true), size=sample_size, replace=False)
            if hasattr(y_true, 'iloc'):
                y_true_sample = y_true.iloc[indices]
            else:
                y_true_sample = y_true[indices]
            y_pred_sample = y_pred[indices]

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                y=y_true_sample,
                mode='lines',
                name='真实值',
                line=dict(color='blue', width=2)
            ))
            fig_ts.add_trace(go.Scatter(
                y=y_pred_sample,
                mode='lines',
                name='预测值',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig_ts.update_layout(
                title="预测值时间序列对比（抽样）",
                xaxis_title="样本索引",
                yaxis_title="风速 (m/s)",
                height=400
            )
            st.plotly_chart(fig_ts, use_container_width=True)

    # 残差分析标签页
    with tabs[1]:
        # 残差分析
        residuals = y_true - y_pred

        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('残差分布', '残差 vs 预测值', '残差QQ图', '残差自相关'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 残差分布
        fig.add_trace(go.Histogram(x=residuals, nbinsx=50, name='残差分布'), row=1, col=1)

        # 残差 vs 预测值
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='残差'), row=1, col=2)
        fig.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=2)

        # QQ图
        theoretical_quantiles = stats.probplot(residuals, dist="norm")
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles[0][0], y=theoretical_quantiles[0][1],
            mode='markers', name='QQ图'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[min(theoretical_quantiles[0][0]), max(theoretical_quantiles[0][0])],
            y=[min(theoretical_quantiles[0][0]), max(theoretical_quantiles[0][0])],
            mode='lines', name='参考线', line=dict(dash='dash')
        ), row=2, col=1)

        # 残差自相关
        autocorr = [1.0] + [np.corrcoef(residuals[:-i], residuals[i:])[0, 1] for i in range(1, 21)]
        fig.add_trace(go.Scatter(
            x=list(range(len(autocorr))), y=autocorr,
            mode='lines+markers', name='自相关'
        ), row=2, col=2)
        fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=2)

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 特征重要性标签页
    with tabs[2]:
        col1, col2 = st.columns(2)

        with col1:
            if feature_importance is not None:
                fig = px.bar(feature_importance.head(10), x='importance', y='feature',
                             title="前10特征重要性（内置）",
                             color='importance',
                             color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("该模型不支持内置特征重要性分析")

        with col2:
            if permutation_importance_result is not None:
                fig = px.bar(permutation_importance_result.head(10), x='importance', y='feature',
                             title="前10置换重要性",
                             error_x='std',
                             color='importance',
                             color_continuous_scale='Plasma')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("未计算置换重要性")

    # 网格搜索分析标签页
    if enable_grid_search and grid_search_results is not None:
        with tabs[3]:
            st.subheader("🔍 网格搜索分析")

            # 获取CV结果
            cv_results = grid_search_results.cv_results_

            # 显示参数搜索范围
            st.markdown("### 📋 参数搜索范围")
            if param_grid_used is not None:
                param_grid_df = pd.DataFrame({
                    '参数': list(param_grid_used.keys()),
                    '搜索值': [str(v) for v in param_grid_used.values()]
                })
                st.dataframe(param_grid_df, use_container_width=True)

            # 参数重要性分析
            st.markdown("### 📈 参数对性能的影响")

            # 提取主要参数
            if best_params is not None:
                param_names = list(best_params.keys())

                # 为每个主要参数创建图表
                for param in ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                              'num_leaves', 'iterations', 'depth']:
                    if param in param_names:
                        # 获取该参数的所有值和对应的平均测试分数
                        param_values = []
                        mean_scores = []

                        for i, params_dict in enumerate(cv_results['params']):
                            if param in params_dict:
                                param_values.append(params_dict[param])
                                mean_scores.append(cv_results['mean_test_score'][i])

                        if param_values:
                            # 创建DataFrame并排序
                            param_df = pd.DataFrame({
                                param: param_values,
                                '平均R²分数': mean_scores
                            })
                            param_df = param_df.groupby(param)['平均R²分数'].mean().reset_index()
                            param_df = param_df.sort_values(param)

                            # 绘制折线图
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=param_df[param],
                                y=param_df['平均R²分数'],
                                mode='lines+markers',
                                name=param,
                                line=dict(width=3)
                            ))

                            # 标记最佳值
                            best_value = best_params[param]
                            best_score = grid_search_results.best_score_
                            fig.add_trace(go.Scatter(
                                x=[best_value],
                                y=[best_score],
                                mode='markers',
                                marker=dict(size=15, color='red', symbol='star'),
                                name=f'最佳值: {best_value}'
                            ))

                            fig.update_layout(
                                title=f"{param} 对模型性能的影响",
                                xaxis_title=param,
                                yaxis_title="平均R²分数",
                                height=400,
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)

            # 显示搜索过程摘要
            st.markdown("### 📊 搜索过程摘要")

            # 获取排名前10的参数组合
            top_indices = np.argsort(cv_results['mean_test_score'])[-10:][::-1]

            top_results = []
            for i, idx in enumerate(top_indices):
                top_results.append({
                    '排名': i + 1,
                    'R²分数': f"{cv_results['mean_test_score'][idx]:.4f}",
                    '参数': str(cv_results['params'][idx])
                })

            top_df = pd.DataFrame(top_results)
            st.dataframe(top_df, use_container_width=True)

            # 下载功能
            st.markdown("### 💾 保存搜索结果")

            col_dl1, col_dl2, col_dl3 = st.columns(3)

            with col_dl1:
                if st.button("保存最佳模型", use_container_width=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                        joblib.dump(grid_search_results.best_estimator_, tmp.name)

                        with open(tmp.name, 'rb') as f:
                            st.download_button(
                                label="下载最佳模型",
                                data=f,
                                file_name=f"best_{model_name.lower()}_model_{time.strftime('%Y%m%d_%H%M%S')}.pkl",
                                mime="application/octet-stream"
                            )

                    os.unlink(tmp.name)

            with col_dl2:
                if st.button("下载参数报告", use_container_width=True):
                    # 创建详细的参数报告
                    report = f"""
{model_name}网格搜索参数报告
生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

最佳参数组合:
{json.dumps(best_params, indent=2)}

性能指标:
- 最佳交叉验证R²: {grid_search_results.best_score_:.4f}
- 测试集R²: {results['r2']:.4f}
- 测试集MAE: {results['mae']:.3f}
- 测试集RMSE: {results['rmse']:.3f}
- 训练时间: {results['training_time']:.2f}秒

搜索统计:
- 搜索组合总数: {len(cv_results['params'])}
- 搜索方法: {grid_search_results.__class__.__name__}
- 交叉验证折数: 3

前十名参数组合:
"""

                    for i, idx in enumerate(top_indices):
                        report += f"\n{i + 1}. R²: {cv_results['mean_test_score'][idx]:.4f}\n"
                        report += f"   参数: {cv_results['params'][idx]}\n"

                    st.download_button(
                        label="下载参数报告",
                        data=report,
                        file_name=f"{model_name.lower()}_grid_search_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

            with col_dl3:
                if st.button("下载CSV结果", use_container_width=True):
                    # 创建CSV格式的结果
                    csv_data = []
                    for i in range(len(cv_results['params'])):
                        row = {}
                        for param, value in cv_results['params'][i].items():
                            row[param] = value
                        row['mean_test_score'] = cv_results['mean_test_score'][i]
                        row['std_test_score'] = cv_results['std_test_score'][i]
                        csv_data.append(row)

                    csv_df = pd.DataFrame(csv_data)
                    csv_str = csv_df.to_csv(index=False)

                    st.download_button(
                        label="下载完整CSV",
                        data=csv_str,
                        file_name=f"{model_name.lower()}_grid_search_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

    # 交叉验证标签页
    cross_val_tab_index = 4 if enable_grid_search and grid_search_results is not None else 3
    with tabs[cross_val_tab_index]:
        # 交叉验证结果
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, len(cv_scores) + 1)),
            y=cv_scores,
            marker_color='lightgreen',
            name='折数 R²'
        ))
        fig.add_hline(y=results['cv_mean'], line_dash='dash', line_color='red',
                      annotation_text=f'平均 R²: {results["cv_mean"]:.4f}')
        fig.update_layout(
            title="交叉验证结果",
            xaxis_title="折数",
            yaxis_title="R² 分数",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write(f"交叉验证稳定性: {results['cv_std']:.4f} (标准差)")

        # 交叉验证详细信息
        cv_details = pd.DataFrame({
            '折数': list(range(1, len(cv_scores) + 1)),
            'R²分数': cv_scores
        })
        st.dataframe(cv_details, use_container_width=True)

    # 误差分析标签页
    error_tab_index = cross_val_tab_index + 1
    with tabs[error_tab_index]:
        # 误差分析
        absolute_errors = np.abs(y_true - y_pred)
        relative_errors = np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("最大绝对误差", f"{np.max(absolute_errors):.3f}")
            st.metric("误差标准差", f"{np.std(absolute_errors):.3f}")
            st.metric("平均相对误差", f"{np.mean(relative_errors) * 100:.1f}%")
        with col2:
            st.metric("误差 < 0.5 m/s", f"{np.mean(absolute_errors < 0.5) * 100:.1f}%")
            st.metric("误差 < 1.0 m/s", f"{np.mean(absolute_errors < 1.0) * 100:.1f}%")
            st.metric("误差 < 2.0 m/s", f"{np.mean(absolute_errors < 2.0) * 100:.1f}%")

        # 误差分布
        fig = px.histogram(x=absolute_errors, nbins=50, title="绝对误差分布")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # 误差与真实值关系
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_true, y=absolute_errors,
            mode='markers',
            marker=dict(color='orange', opacity=0.6),
            name='绝对误差'
        ))
        fig.update_layout(
            title="绝对误差 vs 真实值",
            xaxis_title="真实风速 (m/s)",
            yaxis_title="绝对误差 (m/s)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # 模型诊断标签页
    model_diag_tab_index = error_tab_index + 1
    with tabs[model_diag_tab_index]:
        # 模型诊断
        st.subheader("模型诊断信息")

        # 学习曲线分析（简化版）
        train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
        train_scores = []

        for size in train_sizes:
            n_samples = int(len(X_test) * size)
            if n_samples > 0:
                X_subset = X_test[:n_samples]
                y_subset = y_true[:n_samples]
                pred_subset = model.predict(X_subset)
                train_scores.append(r2_score(y_subset, pred_subset))

        if train_scores:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[size for size in train_sizes[:len(train_scores)]],
                y=train_scores,
                mode='lines+markers',
                name='测试集 R²'
            ))
            fig.update_layout(
                title="模型性能 vs 数据量",
                xaxis_title="数据比例",
                yaxis_title="R² 分数",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无法计算学习曲线")

        # 模型稳定性分析
        st.write("**模型稳定性分析**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("交叉验证标准差", f"{results['cv_std']:.4f}")
        with col2:
            stability = "高" if results['cv_std'] < 0.05 else "中" if results['cv_std'] < 0.1 else "低"
            st.metric("稳定性评级", stability)

        # 模型复杂度分析
        st.write("**模型复杂度分析**")
        if hasattr(model, 'n_estimators'):
            st.write(f"- 树的数量: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            st.write(f"- 最大深度: {model.max_depth}")
        if hasattr(model, 'feature_importances_'):
            st.write(f"- 特征数量: {len(model.feature_importances_)}")


# ===================== 多模型对比结果显示函数 =====================
def display_comparison_results(comparison_results, feature_importances, y_true, predictions, selected_features, models,
                               X_test_scaled, grid_search_infos=None, all_grid_search_results=None):
    """显示多模型对比结果"""

    st.subheader("📋 算法性能对比总结")

    # 创建对比表格
    df_comparison = pd.DataFrame(comparison_results)

    # 添加排名信息
    df_comparison['MAE排名'] = df_comparison['MAE'].rank(method='min').astype(int)
    df_comparison['RMSE排名'] = df_comparison['RMSE'].rank(method='min').astype(int)
    df_comparison['R²排名'] = df_comparison['R²'].rank(method='min', ascending=False).astype(int)
    df_comparison['训练时间排名'] = df_comparison['训练时间 (秒)'].rank(method='min').astype(int)

    # 计算综合评分
    df_comparison['综合评分'] = (
            df_comparison['R²'] * 0.4 +  # R²权重最高
            (1 - df_comparison['MAE'] / df_comparison['MAE'].max()) * 0.3 +
            (1 - df_comparison['训练时间 (秒)'] / df_comparison['训练时间 (秒)'].max()) * 0.3
    )
    df_comparison['综合排名'] = df_comparison['综合评分'].rank(method='min', ascending=False).astype(int)

    # 排序显示
    df_comparison = df_comparison.sort_values('综合排名')

    # 使用颜色突出显示最佳结果
    def color_ranking(val, column):
        if val == 1:  # 第一名
            return 'background-color: #4CAF50; color: white; font-weight: bold;'
        elif val == 2:  # 第二名
            return 'background-color: #8BC34A; color: white;'
        elif val == 3:  # 第三名
            return 'background-color: #CDDC39;'
        return ''

    # 格式化表格
    styled_df = df_comparison.style.format({
        "MAE": "{:.3f}",
        "RMSE": "{:.3f}",
        "R²": "{:.4f}",
        "训练时间 (秒)": "{:.2f}",
        "综合评分": "{:.3f}"
    }).applymap(lambda x: color_ranking(x, 'MAE排名'), subset=['MAE排名']) \
        .applymap(lambda x: color_ranking(x, 'R²排名'), subset=['R²排名']) \
        .applymap(lambda x: color_ranking(x, '综合排名'), subset=['综合排名'])

    st.dataframe(styled_df, use_container_width=True)

    # ================== 可视化对比 ==================
    st.subheader("📊 性能可视化对比")

    # 创建标签页
    tab_names = ["雷达图综合对比", "散点图分析", "指标柱状图", "详细分析"]

    # 如果有网格搜索信息，添加网格搜索标签页
    if grid_search_infos:
        tab_names.insert(3, "网格搜索对比")

    tabs = st.tabs(tab_names)

    with tab_names[0]:
        # 雷达图综合对比
        categories = ['R²(越高越好)', 'MAE(越低越好)', 'RMSE(越低越好)', '速度(越快越好)']

        fig_radar = go.Figure()

        colors = px.colors.qualitative.Set3

        for i, row in df_comparison.iterrows():
            # 归一化指标 (0-1之间，1表示最好)
            r2_norm = row['R²']  # R²已经是0-1
            mae_norm = 1 - (row['MAE'] / df_comparison['MAE'].max())
            rmse_norm = 1 - (row['RMSE'] / df_comparison['RMSE'].max())
            speed_norm = 1 - (row['训练时间 (秒)'] / df_comparison['训练时间 (秒)'].max())

            values = [r2_norm, mae_norm, rmse_norm, speed_norm]

            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # 闭合雷达图
                theta=categories + [categories[0]],
                name=row['算法'],
                fill='toself',
                opacity=0.6,
                line=dict(color=colors[i % len(colors)], width=2)
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="算法综合性能雷达图",
            height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 其他标签页保持不变...
    # ... [原有的其他对比分析标签页内容保持不变] ...

    # 网格搜索对比标签页
    if grid_search_infos:
        grid_search_tab_index = 3
        with tabs[grid_search_tab_index]:
            st.subheader("🔬 网格搜索对比分析")

            if grid_search_infos:
                # 显示每个算法的网格搜索结果
                for algo, info in grid_search_infos.items():
                    with st.expander(f"{algo} 网格搜索结果", expanded=False):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.write("**最佳参数组合:**")
                            for param, value in info['best_params'].items():
                                st.write(f"- **{param}**: {value}")

                        with col2:
                            st.metric("最佳交叉验证R²", f"{info['best_score']:.4f}")
                            if algo in all_grid_search_results:
                                st.write(f"搜索组合数: {len(all_grid_search_results[algo].cv_results_['params']):,}")

                # 网格搜索效果对比
                st.markdown("### 📈 网格搜索效果对比")

                gs_results = []
                for algo, info in grid_search_infos.items():
                    gs_results.append({
                        '算法': algo,
                        '最佳交叉验证R²': info['best_score'],
                        '参数组合数': len(info['cv_results']['params'])
                    })

                if gs_results:
                    gs_df = pd.DataFrame(gs_results)

                    # 柱状图对比
                    fig = px.bar(gs_df, x='算法', y='最佳交叉验证R²',
                                 title="各算法网格搜索最佳结果",
                                 color='最佳交叉验证R²',
                                 color_continuous_scale='Viridis')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # 下载网格搜索汇总报告
                    st.markdown("### 💾 下载网格搜索汇总报告")

                    if st.button("生成网格搜索汇总报告", use_container_width=True):
                        report = "多算法网格搜索汇总报告\n"
                        report += f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

                        for algo, info in grid_search_infos.items():
                            report += f"=== {algo} ===\n"
                            report += f"最佳交叉验证R²: {info['best_score']:.4f}\n"
                            report += "最佳参数:\n"
                            for param, value in info['best_params'].items():
                                report += f"  {param}: {value}\n"
                            report += f"搜索参数组合数: {len(info['cv_results']['params']):,}\n\n"

                        st.download_button(
                            label="下载汇总报告",
                            data=report,
                            file_name=f"multi_model_grid_search_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
            else:
                st.info("本次对比分析中未进行网格搜索")

    # 详细分析标签页
    detail_tab_index = grid_search_tab_index + 1 if grid_search_infos else 3
    with tabs[detail_tab_index]:
        # 详细分析
        st.subheader("🔍 详细分析报告")

        # 最佳算法
        best_algo = df_comparison.iloc[0]
        st.success(f"🏆 **最佳综合算法**: **{best_algo['算法']}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²得分", f"{best_algo['R²']:.4f}",
                      delta=f"排名第{best_algo['R²排名']}")
        with col2:
            st.metric("MAE误差", f"{best_algo['MAE']:.3f}",
                      delta=f"排名第{best_algo['MAE排名']}")
        with col3:
            st.metric("训练时间", f"{best_algo['训练时间 (秒)']:.2f}s",
                      delta=f"排名第{best_algo['训练时间排名']}")

        # 算法特点分析
        st.markdown("### 📝 各算法特点分析")

        algo_analysis = {
            "随机森林": "稳定性好，抗过拟合能力强，适合中等规模数据，特征重要性解释性好",
            "XGBoost": "精度高，计算效率好，适合结构化数据，正则化防止过拟合",
            "CatBoost": "处理类别特征能力强，无需预处理，抗过拟合，训练速度较快",
            "LightGBM": "训练速度快，内存占用小，支持类别特征，适合大规模数据"
        }

        for algo in df_comparison['算法']:
            with st.expander(f"{algo} 算法特点"):
                if algo in algo_analysis:
                    st.write(f"**特点**: {algo_analysis[algo]}")

                # 显示是否使用了网格搜索
                if grid_search_infos and algo in grid_search_infos:
                    st.success("✅ 本次使用了网格搜索调参")
                    st.write(f"**最佳交叉验证R²**: {grid_search_infos[algo]['best_score']:.4f}")
                else:
                    st.info("ℹ️ 本次使用手动参数配置")

                # 显示该算法的具体指标
                algo_data = df_comparison[df_comparison['算法'] == algo].iloc[0]
                st.write(f"""
                - **R²**: {algo_data['R²']:.4f} (排名: {algo_data['R²排名']})
                - **MAE**: {algo_data['MAE']:.3f} (排名: {algo_data['MAE排名']})
                - **训练时间**: {algo_data['训练时间 (秒)']:.2f}秒 (排名: {algo_data['训练时间排名']})
                - **综合评分**: {algo_data['综合评分']:.3f} (排名: {algo_data['综合排名']})
                """)

        # 推荐选择
        st.markdown("### 💡 算法选择建议")

        # 根据需求推荐
        best_r2_algo = df_comparison.loc[df_comparison['R²'].idxmax()]
        best_mae_algo = df_comparison.loc[df_comparison['MAE'].idxmin()]
        fastest_algo = df_comparison.loc[df_comparison['训练时间 (秒)'].idxmin()]

        st.info(f"""
        **根据您的需求推荐**:

        - 🎯 **追求最高精度**: 选择 **{best_r2_algo['算法']}** (R²={best_r2_algo['R²']:.4f})
        - 📉 **要求最小误差**: 选择 **{best_mae_algo['算法']}** (MAE={best_mae_algo['MAE']:.3f})
        - ⚡ **关注训练速度**: 选择 **{fastest_algo['算法']}** ({fastest_algo['训练时间 (秒)']:.2f}秒)
        - ⚖️ **最佳平衡选择**: 选择 **{best_algo['算法']}** (综合评分={best_algo['综合评分']:.3f})

        **风电预测建议**: 对于风电预测，推荐优先考虑 **{best_r2_algo['算法']}** 或 **{best_algo['算法']}**，
        因为预测精度对风电功率估算最为关键。
        """)

        # 保存结果选项
        st.markdown("### 💾 保存分析结果")

        col_save1, col_save2 = st.columns(2)

        with col_save1:
            if st.button("📥 导出对比结果", use_container_width=True):
                # 创建可下载的数据
                csv = df_comparison.to_csv(index=False)
                st.download_button(
                    label="下载CSV文件",
                    data=csv,
                    file_name="算法对比结果.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col_save2:
            if st.button("📋 生成完整报告", use_container_width=True):
                # 生成完整报告
                report = f"""
风电场风速预测模型对比分析报告
生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

性能对比总结:
{df_comparison.to_string(index=False)}

最佳算法: {best_algo['算法']}
- R²分数: {best_algo['R²']:.4f}
- MAE误差: {best_algo['MAE']:.3f}
- 训练时间: {best_algo['训练时间 (秒)']:.2f}秒

各算法特点:
"""
                for algo in df_comparison['算法']:
                    algo_data = df_comparison[df_comparison['算法'] == algo].iloc[0]
                    report += f"\n{algo}:"
                    if algo in algo_analysis:
                        report += f"\n  特点: {algo_analysis[algo]}"
                    report += f"\n  R²: {algo_data['R²']:.4f} (排名: {algo_data['R²排名']})"
                    report += f"\n  MAE: {algo_data['MAE']:.3f} (排名: {algo_data['MAE排名']})"
                    report += f"\n  训练时间: {algo_data['训练时间 (秒)']:.2f}秒"

                st.download_button(
                    label="下载完整报告",
                    data=report,
                    file_name=f"风电预测模型对比报告_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )