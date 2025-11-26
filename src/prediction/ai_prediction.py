import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import time
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

# XGBoost 库
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# 深度学习库
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except Exception:
    HAS_TENSORFLOW = False


# ===================== 深度学习模型构建函数 =====================
def create_lstm_model(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001):
    """创建LSTM模型"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True, dropout=dropout_rate),
        LSTM(units // 2, dropout=dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


def create_gru_model(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001):
    """创建GRU模型"""
    model = Sequential([
        Input(shape=input_shape),
        GRU(units, return_sequences=True, dropout=dropout_rate),
        GRU(units // 2, dropout=dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


def prepare_sequences_for_dl(X, y, time_steps=10):
    """为深度学习模型准备时间序列数据"""
    X_sequences = []
    y_sequences = []

    for i in range(time_steps, len(X)):
        X_sequences.append(X[i - time_steps:i])
        y_sequences.append(y[i])

    return np.array(X_sequences), np.array(y_sequences)


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
        # 模型选择 - 四种算法：随机森林、XGBoost、LSTM、GRU
        model_options = ["随机森林", "XGBoost", "LSTM", "GRU"]

        # 检查库可用性
        available_models = []
        for model in model_options:
            if model in ["LSTM", "GRU"]:
                if HAS_TENSORFLOW:
                    available_models.append(model)
            elif model == "XGBoost":
                if HAS_XGBOOST:
                    available_models.append(model)
            else:
                available_models.append(model)

        model_option = st.selectbox("选择算法", available_models)

        # 高级参数
        with st.expander("高级参数"):
            test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
            cv_folds = st.slider("交叉验证折数", 3, 10, 5)
            enable_permutation = st.checkbox("启用置换重要性分析", value=True)

            # XGBoost 特定参数
            if model_option == "XGBoost":
                xgb_learning_rate = st.slider("XGBoost学习率", 0.01, 0.3, 0.1, 0.01)
                xgb_max_depth = st.slider("XGBoost最大深度", 3, 10, 6)
                xgb_n_estimators = st.slider("XGBoost估计器数量", 50, 300, 100)

            # 深度学习特定参数
            if model_option in ["LSTM", "GRU"]:
                time_steps = st.slider("时间步长", 5, 50, 10,
                                       help="考虑的历史时间步数")
                lstm_units = st.slider("LSTM/GRU单元数", 16, 128, 50)
                epochs = st.slider("训练轮次", 10, 200, 50)
                batch_size = st.slider("批次大小", 16, 128, 32)
                learning_rate = st.slider("学习率", 0.0001, 0.01, 0.001, 0.0001)

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
            elif model_option == "LSTM":
                # 准备时间序列数据
                X_train_seq, y_train_seq = prepare_sequences_for_dl(X_train_scaled, y_train.values, time_steps)
                X_test_seq, y_test_seq = prepare_sequences_for_dl(X_test_scaled, y_test.values, time_steps)

                # 创建模型
                model = create_lstm_model(
                    input_shape=(time_steps, len(selected_features)),
                    units=lstm_units,
                    learning_rate=learning_rate
                )

                # 训练模型
                start_time = time.time()
                history = model.fit(
                    X_train_seq, y_train_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test_seq, y_test_seq),
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(patience=5, factor=0.5)
                    ]
                )
                training_time = time.time() - start_time

                # 预测
                y_pred = model.predict(X_test_seq).flatten()
                y_test = y_test_seq

            elif model_option == "GRU":
                # 准备时间序列数据
                X_train_seq, y_train_seq = prepare_sequences_for_dl(X_train_scaled, y_train.values, time_steps)
                X_test_seq, y_test_seq = prepare_sequences_for_dl(X_test_scaled, y_test.values, time_steps)

                # 创建模型
                model = create_gru_model(
                    input_shape=(time_steps, len(selected_features)),
                    units=lstm_units,
                    learning_rate=learning_rate
                )

                # 训练模型
                start_time = time.time()
                history = model.fit(
                    X_train_seq, y_train_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test_seq, y_test_seq),
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(patience=5, factor=0.5)
                    ]
                )
                training_time = time.time() - start_time

                # 预测
                y_pred = model.predict(X_test_seq).flatten()
                y_test = y_test_seq

            # 传统机器学习模型训练和预测
            if model_option not in ["LSTM", "GRU"]:
                start_time = time.time()
                model.fit(X_train_scaled, y_train)
                training_time = time.time() - start_time
                y_pred = model.predict(X_test_scaled)

            # 交叉验证（仅适用于传统模型）
            cv_scores = []
            if model_option not in ["LSTM", "GRU"]:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')

            # 计算指标
            results = calculate_metrics(y_test, y_pred, training_time)
            if model_option not in ["LSTM", "GRU"]:
                results['cv_mean'] = cv_scores.mean() if len(cv_scores) > 0 else 0
                results['cv_std'] = cv_scores.std() if len(cv_scores) > 0 else 0
            else:
                results['cv_mean'] = 0
                results['cv_std'] = 0

            # 特征重要性（仅适用于支持特征重要性的模型）
            feature_importance = None
            permutation_importance_result = None

            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

            # 置换重要性
            if enable_permutation and model_option not in ["LSTM", "GRU"]:
                with st.spinner("正在计算置换重要性..."):
                    permutation_importance_result = calculate_permutation_importance(
                        model, X_test_scaled, y_test, selected_features
                    )

            # 显示结果
            display_single_model_results(
                results, feature_importance, permutation_importance_result,
                model_option, y_test, y_pred, cv_scores, X_test_scaled, model, history
            )


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

    # 模型选择 - 四种算法：随机森林、XGBoost、LSTM、GRU
    model_options = ["随机森林", "XGBoost", "LSTM", "GRU"]

    # 检查库可用性
    available_models = []
    for model in model_options:
        if model in ["LSTM", "GRU"]:
            if HAS_TENSORFLOW:
                available_models.append(model)
        elif model == "XGBoost":
            if HAS_XGBOOST:
                available_models.append(model)
        else:
            available_models.append(model)

    selected_algorithms = st.multiselect(
        "选择对比算法",
        options=available_models,
        default=available_models  # 默认选择所有可用模型
    )

    # XGBoost 参数
    xgb_params = {}
    if "XGBoost" in selected_algorithms:
        with st.expander("XGBoost参数配置"):
            xgb_params['learning_rate'] = st.slider("XGBoost学习率", 0.01, 0.3, 0.1, 0.01)
            xgb_params['max_depth'] = st.slider("XGBoost最大深度", 3, 10, 6)
            xgb_params['n_estimators'] = st.slider("XGBoost估计器数量", 50, 300, 100)

    # 深度学习参数
    dl_params = {}
    if any(model in selected_algorithms for model in ["LSTM", "GRU"]):
        with st.expander("深度学习参数配置"):
            time_steps = st.slider("时间步长", 5, 50, 10)
            lstm_units = st.slider("LSTM/GRU单元数", 16, 128, 50)
            epochs = st.slider("训练轮次", 10, 100, 30)
            batch_size = st.slider("批次大小", 16, 128, 32)
            dl_params = {
                'time_steps': time_steps,
                'units': lstm_units,
                'epochs': epochs,
                'batch_size': batch_size
            }

    if st.button("🔬 开始对比分析", type="primary", use_container_width=True):
        if not selected_features or not selected_algorithms:
            st.warning("请选择特征变量和对比算法")
            return

        with st.spinner("正在进行多模型对比分析..."):
            # 数据准备
            X = df[selected_features].fillna(0)
            y = df[target_column].fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 训练所有模型
            comparison_results = []
            feature_importances = {}
            predictions = {}
            models = {}
            training_histories = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, algo in enumerate(selected_algorithms):
                status_text.text(f"正在训练 {algo}... ({i + 1}/{len(selected_algorithms)})")

                try:
                    model = None
                    history = None

                    if algo == "随机森林":
                        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                    elif algo == "XGBoost":
                        model = xgb.XGBRegressor(
                            n_estimators=xgb_params['n_estimators'],
                            max_depth=xgb_params['max_depth'],
                            learning_rate=xgb_params['learning_rate'],
                            random_state=42,
                            n_jobs=-1
                        )
                    elif algo == "LSTM":
                        # 准备时间序列数据
                        X_train_seq, y_train_seq = prepare_sequences_for_dl(
                            X_train_scaled, y_train.values, dl_params['time_steps'])
                        X_test_seq, y_test_seq = prepare_sequences_for_dl(
                            X_test_scaled, y_test.values, dl_params['time_steps'])

                        model = create_lstm_model(
                            input_shape=(dl_params['time_steps'], len(selected_features)),
                            units=dl_params['units']
                        )

                        start_time = time.time()
                        history = model.fit(
                            X_train_seq, y_train_seq,
                            epochs=dl_params['epochs'],
                            batch_size=dl_params['batch_size'],
                            validation_data=(X_test_seq, y_test_seq),
                            verbose=0
                        )
                        training_time = time.time() - start_time

                        y_pred = model.predict(X_test_seq).flatten()
                        y_test_used = y_test_seq

                    elif algo == "GRU":
                        # 准备时间序列数据
                        X_train_seq, y_train_seq = prepare_sequences_for_dl(
                            X_train_scaled, y_train.values, dl_params['time_steps'])
                        X_test_seq, y_test_seq = prepare_sequences_for_dl(
                            X_test_scaled, y_test.values, dl_params['time_steps'])

                        model = create_gru_model(
                            input_shape=(dl_params['time_steps'], len(selected_features)),
                            units=dl_params['units']
                        )

                        start_time = time.time()
                        history = model.fit(
                            X_train_seq, y_train_seq,
                            epochs=dl_params['epochs'],
                            batch_size=dl_params['batch_size'],
                            validation_data=(X_test_seq, y_test_seq),
                            verbose=0
                        )
                        training_time = time.time() - start_time

                        y_pred = model.predict(X_test_seq).flatten()
                        y_test_used = y_test_seq

                    # 传统模型训练
                    if algo not in ["LSTM", "GRU"]:
                        start_time = time.time()
                        model.fit(X_train_scaled, y_train)
                        training_time = time.time() - start_time
                        y_pred = model.predict(X_test_scaled)
                        y_test_used = y_test

                    predictions[algo] = y_pred
                    models[algo] = model
                    if history:
                        training_histories[algo] = history

                    # 计算指标
                    results = calculate_metrics(y_test_used, y_pred, training_time)

                    # 特征重要性
                    if hasattr(model, 'feature_importances_'):
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
                predictions, selected_features, models, X_test_scaled, training_histories
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
        # 使用sklearn的permutation_importance
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        # 创建重要性DataFrame
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
                                 model_name, y_true, y_pred, cv_scores, X_test, model, history=None):
    st.subheader(f"📊 {model_name} 模型性能")

    # 指标卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE", f"{results['mae']:.3f}")
    col2.metric("RMSE", f"{results['rmse']:.3f}")
    col3.metric("R²", f"{results['r2']:.4f}")
    col4.metric("训练时间", f"{results['training_time']:.2f}秒")

    if model_name not in ["LSTM", "GRU"]:
        col5.metric("交叉验证 R²", f"{results['cv_mean']:.4f}")
    else:
        col5.metric("验证损失", f"{history.history['val_loss'][-1]:.4f}" if history else "N/A")

    # 可视化标签页
    tab_names = ["预测性能", "残差分析", "特征重要性", "交叉验证", "误差分析", "模型诊断"]
    if model_name in ["LSTM", "GRU"]:
        tab_names.insert(3, "训练过程")

    tabs = st.tabs(tab_names)

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
            yaxis_title="预测风速 (m/s)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 预测时间序列（如果数据有序）
        sample_size = min(200, len(y_true))
        if len(y_true) > sample_size:
            try:
                indices = np.random.choice(len(y_true), size=sample_size, replace=False)
                if hasattr(y_true, 'iloc'):
                    y_true_sample = y_true.iloc[indices]
                else:
                    y_true_sample = y_true[indices]
                y_pred_sample = y_pred[indices]

                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    y=y_true_sample,
                    mode='lines+markers', name='真实值'
                ))
                fig_ts.add_trace(go.Scatter(
                    y=y_pred_sample, mode='lines+markers', name='预测值'
                ))
                fig_ts.update_layout(title="预测值时间序列对比（抽样）")
                st.plotly_chart(fig_ts, use_container_width=True)
            except Exception as e:
                st.warning(f"时间序列抽样失败: {str(e)}")

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

    with tabs[2]:
        # 特征重要性
        col1, col2 = st.columns(2)

        with col1:
            if feature_importance is not None:
                fig = px.bar(feature_importance.head(10), x='importance', y='feature',
                             title="前10特征重要性（内置）")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("该模型不支持内置特征重要性分析")

        with col2:
            if permutation_importance_result is not None:
                fig = px.bar(permutation_importance_result.head(10), x='importance', y='feature',
                             title="前10置换重要性")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("未计算置换重要性")

    # 深度学习训练过程可视化
    if model_name in ["LSTM", "GRU"] and history:
        with tabs[3]:
            st.subheader("📈 训练过程监控")

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history.history['loss'],
                mode='lines',
                name='训练损失'
            ))
            fig_loss.add_trace(go.Scatter(
                y=history.history['val_loss'],
                mode='lines',
                name='验证损失'
            ))
            fig_loss.update_layout(
                title="训练和验证损失曲线",
                xaxis_title="训练轮次",
                yaxis_title="损失值"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            # 显示模型结构信息
            st.subheader("🛠️ 模型结构信息")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text_area("模型结构", "\n".join(model_summary), height=200)

    # 调整后续标签页索引
    offset = 1 if model_name in ["LSTM", "GRU"] else 0

    with tabs[3 + offset]:
        # 交叉验证结果
        if model_name not in ["LSTM", "GRU"]:
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
                yaxis_title="R² 分数"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write(f"交叉验证稳定性: {results['cv_std']:.4f} (标准差)")
        else:
            st.info("深度学习模型使用验证集进行性能评估")

    with tabs[4 + offset]:
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
        st.plotly_chart(fig, use_container_width=True)

    with tabs[5 + offset]:
        # 模型诊断
        st.subheader("模型诊断信息")

        if model_name not in ["LSTM", "GRU"]:
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
                    yaxis_title="R² 分数"
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
        else:
            st.info("深度学习模型诊断信息在训练过程标签页中显示")


# ===================== 多模型对比结果显示函数 =====================
def display_comparison_results(comparison_results, feature_importances, y_true, predictions, selected_features, models,
                               X_test_scaled, training_histories=None):
    """显示多模型对比结果"""
    st.subheader("📋 性能对比表")
    df_comparison = pd.DataFrame(comparison_results)
    st.dataframe(df_comparison.style.format({
        "MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.4f}", "训练时间 (秒)": "{:.2f}"
    }), use_container_width=True)

    # 深度学习训练历史可视化
    if training_histories:
        st.subheader("📈 深度学习模型训练过程")
        dl_algorithms = [algo for algo in training_histories.keys() if algo in ["LSTM", "GRU"]]

        if dl_algorithms:
            fig_history = go.Figure()
            for algo in dl_algorithms:
                history = training_histories[algo]
                fig_history.add_trace(go.Scatter(
                    y=history.history['loss'],
                    mode='lines',
                    name=f'{algo} - 训练损失'
                ))
                fig_history.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    mode='lines',
                    name=f'{algo} - 验证损失',
                    line=dict(dash='dash')
                ))

            fig_history.update_layout(
                title="深度学习模型训练损失曲线",
                xaxis_title="训练轮次",
                yaxis_title="损失值",
                height=400
            )
            st.plotly_chart(fig_history, use_container_width=True)

    # 特征重要性对比
    if feature_importances:
        st.subheader("🎯 特征重要性对比")
        algorithms = list(feature_importances.keys())

        # 选择前5个特征进行对比
        top_features = set()
        for algo in algorithms:
            top_features.update(feature_importances[algo].head(5)['feature'].tolist())
        top_features = list(top_features)[:8]  # 最多8个特征

        fig = go.Figure()
        for algo in algorithms:
            algo_importance = []
            for feature in top_features:
                feature_row = feature_importances[algo][feature_importances[algo]['feature'] == feature]
                if len(feature_row) > 0:
                    algo_importance.append(feature_row['importance'].values[0])
                else:
                    algo_importance.append(0)

            fig.add_trace(go.Bar(
                name=algo,
                x=top_features,
                y=algo_importance,
                text=[f'{x:.3f}' for x in algo_importance],
                textposition='auto'
            ))

        fig.update_layout(
            barmode='group',
            title="前几特征重要性对比",
            xaxis_title="特征",
            yaxis_title="重要性分数",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================== 预测值 vs 真实值散点图对比 ==================
    st.markdown("### 📊 预测性能对比图")

    # 创建散点图
    fig_scatter = go.Figure()

    # 颜色列表
    colors = px.colors.qualitative.Set3

    # 为每个算法添加散点
    for i, (algo, y_pred) in enumerate(predictions.items()):
        # 降采样避免过于密集
        sample_size = min(1000, len(y_true))
        if len(y_true) > sample_size:
            try:
                indices = np.random.choice(len(y_true), size=sample_size, replace=False)
                if hasattr(y_true, 'iloc'):
                    y_true_sample = y_true.iloc[indices]
                else:
                    y_true_sample = y_true[indices]
                y_pred_sample = y_pred[indices]
            except Exception as e:
                st.warning(f"抽样失败: {str(e)}，使用全部数据")
                y_true_sample = y_true
                y_pred_sample = y_pred
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred

        # 计算该算法的R²
        algo_r2 = r2_score(y_true_sample, y_pred_sample)

        fig_scatter.add_trace(go.Scatter(
            x=y_true_sample,
            y=y_pred_sample,
            mode='markers',
            name=f'{algo} (R²={algo_r2:.3f})',
            marker=dict(
                color=colors[i % len(colors)],
                opacity=0.6,
                size=6
            ),
            hovertemplate='<b>真实值</b>: %{x:.2f}<br><b>预测值</b>: %{y:.2f}<br><b>算法</b>: ' + algo + '<extra></extra>'
        ))

    # 添加理想拟合线
    min_val = min(min(y_true), min([min(pred) for pred in predictions.values()]))
    max_val = max(max(y_true), max([max(pred) for pred in predictions.values()]))
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='理想拟合线 (y=x)',
        line=dict(dash='dash', color='black', width=2),
        hovertemplate='理想拟合线<extra></extra>'
    ))

    fig_scatter.update_layout(
        title="预测值 vs 真实值散点图对比",
        xaxis_title="真实风速 (m/s)",
        yaxis_title="预测风速 (m/s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ================== 性能指标柱状图对比 ==================
    st.markdown("### 📈 性能指标可视化对比")

    tab1, tab2, tab3 = st.tabs(["MAE & RMSE", "R² 决定系数", "训练时间"])

    with tab1:
        fig_metrics = go.Figure()
        algorithms = [result["算法"] for result in comparison_results]
        mae_values = [result["MAE"] for result in comparison_results]
        rmse_values = [result["RMSE"] for result in comparison_results]

        fig_metrics.add_trace(go.Bar(
            name='MAE',
            x=algorithms,
            y=mae_values,
            marker_color='#FF6B6B',
            text=[f'{x:.3f}' for x in mae_values],
            textposition='auto'
        ))
        fig_metrics.add_trace(go.Bar(
            name='RMSE',
            x=algorithms,
            y=rmse_values,
            marker_color='#4ECDC4',
            text=[f'{x:.3f}' for x in rmse_values],
            textposition='auto'
        ))

        fig_metrics.update_layout(
            title="MAE 和 RMSE 对比（越低越好）",
            barmode='group',
            xaxis_title="算法",
            yaxis_title="误差值"
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    with tab2:
        fig_r2 = go.Figure()
        r2_values = [result["R²"] for result in comparison_results]

        # 根据R²值设置颜色（越高越好）
        colors_r2 = ['#FF6B6B' if x < 0.5 else '#4ECDC4' if x < 0.8 else '#1A936F' for x in r2_values]

        fig_r2.add_trace(go.Bar(
            x=algorithms,
            y=r2_values,
            marker_color=colors_r2,
            text=[f'{x:.4f}' for x in r2_values],
            textposition='auto'
        ))

        fig_r2.update_layout(
            title="R² 决定系数对比（越接近1越好）",
            xaxis_title="算法",
            yaxis_title="R² 值",
            yaxis_range=[0, 1]
        )
        # 添加参考线
        fig_r2.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="平均水平")
        fig_r2.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="优秀水平")

        st.plotly_chart(fig_r2, use_container_width=True)

    with tab3:
        fig_time = go.Figure()
        time_values = [result["训练时间 (秒)"] for result in comparison_results]

        fig_time.add_trace(go.Bar(
            x=algorithms,
            y=time_values,
            marker_color='#6A0572',
            text=[f'{x:.2f}秒' for x in time_values],
            textposition='auto'
        ))

        fig_time.update_layout(
            title="训练时间对比（秒）",
            xaxis_title="算法",
            yaxis_title="训练时间 (秒)"
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # ================== 算法排名和推荐 ==================
    st.markdown("### 🏆 算法性能排名")

    # 按R²排名
    ranked_by_r2 = sorted(comparison_results, key=lambda x: x['R²'], reverse=True)
    # 按MAE排名
    ranked_by_mae = sorted(comparison_results, key=lambda x: x['MAE'])
    # 按训练时间排名
    ranked_by_time = sorted(comparison_results, key=lambda x: x['训练时间 (秒)'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🥇 精度排名 (R²)**")
        for i, result in enumerate(ranked_by_r2):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i + 1}."
            st.write(f"{medal} {result['算法']}: {result['R²']:.4f}")

    with col2:
        st.markdown("**🎯 误差排名 (MAE)**")
        for i, result in enumerate(ranked_by_mae):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i + 1}."
            st.write(f"{medal} {result['算法']}: {result['MAE']:.3f}")

    with col3:
        st.markdown("**⚡ 速度排名**")
        for i, result in enumerate(ranked_by_time):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i + 1}."
            st.write(f"{medal} {result['算法']}: {result['训练时间 (秒)']:.2f}秒")

    # 总结推荐
    st.markdown("### 💡 算法选择建议")
    best_model = ranked_by_r2[0]['算法']
    best_r2 = ranked_by_r2[0]['R²']
    fastest_model = ranked_by_time[0]['算法']

    st.info(f"""
    **推荐算法**: **{best_model}** (R² = {best_r2:.4f})

    - 🎯 **追求最高精度**: 选择 **{best_model}**
    - ⚡ **关注训练速度**: 选择 **{fastest_model}**
    - ⚖️ **要求平衡**: 推荐尝试 **{ranked_by_r2[1]['算法'] if len(ranked_by_r2) > 1 else best_model}**
    - 📊 **综合考虑**: 查看所有指标，选择最适合您业务场景的算法
    """)