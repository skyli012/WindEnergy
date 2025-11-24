import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ========== 缓存计算函数 ==========
@st.cache_data
def compute_monthly_avg(df, datetime_col):
    """计算月平均风速"""
    df['month'] = df[datetime_col].dt.to_period('M').dt.to_timestamp()
    return df.groupby('month')['predicted_wind_speed'].mean().reset_index()


@st.cache_data
def compute_correlation(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None
    return df[numeric_cols].corr()


# ========== 主页面 ==========
def data_analysis_page():
    st.title("🌬️ 风电场数据分析中心")

    if 'dataset' not in st.session_state:
        st.warning("⚠️ 请先在数据导入页面导入风电场数据")
        return

    df = st.session_state['dataset'].copy()

    # 自动识别时间列
    datetime_col = next(
        (col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower() or 'date' in col.lower()),
        None)
    if not datetime_col:
        st.error("❌ 未检测到时间列")
        return

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    df = df.dropna(subset=[datetime_col]).sort_values(by=datetime_col)

    # 数据概览卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("数据总量", f"{len(df):,} 条")
    with col2:
        st.metric("时间跨度", f"{(df[datetime_col].max() - df[datetime_col].min()).days} 天")
    with col3:
        if 'predicted_wind_speed' in df.columns:
            st.metric("平均风速", f"{df['predicted_wind_speed'].mean():.2f} m/s")
    with col4:
        st.metric("数据点数量", f"{df['point_id'].nunique()}" if 'point_id' in df.columns else "N/A")

    st.markdown("---")

    # 使用标签页组织分析内容
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 时间趋势",
        "🌪️ 风速分析",
        "🧭 风向分析",
        "🗺️ 空间分析",
        "🔗 相关性",
        "📋 数据概览"
    ])

    with tab1:
        temporal_analysis_enhanced(df, datetime_col)
    with tab2:
        windspeed_analysis_enhanced(df)
    with tab3:
        wind_direction_analysis_enhanced(df)
    with tab4:
        spatial_analysis_enhanced(df)
    with tab5:
        correlation_analysis_enhanced(df)
    with tab6:
        data_overview(df, datetime_col)


# ================= 增强的分析模块 ====================
def temporal_analysis_enhanced(df, datetime_col):
    st.subheader("📊 时间序列分析")

    if 'predicted_wind_speed' not in df.columns:
        st.error("未找到风速数据")
        return

    # 只使用小时粒度
    df['time_period'] = df[datetime_col].dt.floor('H')
    title = "小时平均风速趋势"

    # 多变量趋势图
    cols_to_plot = [col for col in ['predicted_wind_speed', 'temperature_c', 'relative_humidity', 'gust_speed'] if
                    col in df.columns]

    if len(cols_to_plot) > 1:
        fig = make_subplots(rows=len(cols_to_plot), cols=1,
                            subplot_titles=cols_to_plot,
                            vertical_spacing=0.05)

        for i, col in enumerate(cols_to_plot, 1):
            # 对于聚合数据，使用小时平均值
            period_avg = df.groupby('time_period')[col].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=period_avg['time_period'], y=period_avg[col],
                           name=col, line=dict(width=2)),
                row=i, col=1
            )

        fig.update_layout(height=300 * len(cols_to_plot), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        period_avg = df.groupby('time_period')['predicted_wind_speed'].mean().reset_index()
        fig = px.line(period_avg, x='time_period', y='predicted_wind_speed',
                      title=title, line_shape='spline')
        st.plotly_chart(fig, use_container_width=True)

    # 日变化分析
    st.subheader("🌤️ 日变化分析")
    df['hour'] = df[datetime_col].dt.hour

    # 小时粒度分析
    time_stats = df.groupby('hour')['predicted_wind_speed'].agg(['mean', 'std']).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_stats['hour'], y=time_stats['mean'],
                             name='平均值', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=time_stats['hour'], y=time_stats['mean'] + time_stats['std'],
                             name='+1标准差', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=time_stats['hour'], y=time_stats['mean'] - time_stats['std'],
                             name='-1标准差', line=dict(color='red', dash='dash'),
                             fill='tonexty'))

    fig.update_layout(
        title="风速小时变化趋势",
        xaxis_title="小时",
        yaxis_title="风速 (m/s)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 短期波动分析
    st.subheader("📈 短期波动分析")

    # 计算滚动平均值
    df_sorted = df.sort_values(datetime_col)
    window_sizes = {
        "1小时": 6,  # 6个10分钟 = 1小时
        "3小时": 18,  # 18个10分钟 = 3小时
        "6小时": 36  # 36个10分钟 = 6小时
    }

    selected_window = st.selectbox("选择滚动窗口:", list(window_sizes.keys()))
    window_size = window_sizes[selected_window]

    # 计算滚动统计
    df_sorted['rolling_mean'] = df_sorted['predicted_wind_speed'].rolling(window=window_size, center=True).mean()
    df_sorted['rolling_std'] = df_sorted['predicted_wind_speed'].rolling(window=window_size, center=True).std()

    # 显示短期趋势
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted[datetime_col], y=df_sorted['predicted_wind_speed'],
                             name='原始风速', line=dict(color='lightblue', width=1), opacity=0.6))
    fig.add_trace(go.Scatter(x=df_sorted[datetime_col], y=df_sorted['rolling_mean'],
                             name=f'{selected_window}滚动平均', line=dict(color='red', width=2)))

    fig.update_layout(
        title=f"风速短期趋势分析 ({selected_window}滚动平均)",
        xaxis_title="时间",
        yaxis_title="风速 (m/s)"
    )
    st.plotly_chart(fig, use_container_width=True)


def windspeed_analysis_enhanced(df):
    st.subheader("🌪️ 风速统计分析")

    if 'predicted_wind_speed' not in df.columns:
        st.error("未找到风速数据")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        # 分布直方图 + 密度曲线
        fig = px.histogram(df, x='predicted_wind_speed', nbins=30,
                           marginal="box", opacity=0.7,
                           title="风速分布直方图")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 统计指标卡片
        stats = df['predicted_wind_speed'].describe()

        st.metric("平均值", f"{stats['mean']:.2f} m/s")
        st.metric("标准差", f"{stats['std']:.2f} m/s")
        st.metric("最大值", f"{stats['max']:.2f} m/s")
        st.metric("中位数", f"{stats['50%']:.2f} m/s")
        st.metric("25%分位数", f"{stats['25%']:.2f} m/s")
        st.metric("75%分位数", f"{stats['75%']:.2f} m/s")

    # 风速等级分析
    st.subheader("📊 风速等级分布")
    wind_bins = [0, 3, 6, 9, 12, 15, float('inf')]
    wind_labels = ['微风(0-3)', '轻风(3-6)', '中风(6-9)', '强风(9-12)', '大风(12-15)', '暴风(15+)']

    df['wind_level'] = pd.cut(df['predicted_wind_speed'], bins=wind_bins, labels=wind_labels)
    wind_level_count = df['wind_level'].value_counts().sort_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=wind_level_count.values, names=wind_level_count.index,
                     title="风速等级分布饼图")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(x=wind_level_count.index, y=wind_level_count.values,
                     title="风速等级频次分布", labels={'x': '风速等级', 'y': '频次'})
        st.plotly_chart(fig, use_container_width=True)


def wind_direction_analysis_enhanced(df):
    st.subheader("🧭 风向综合分析")

    if 'wind_direction' not in df.columns or 'predicted_wind_speed' not in df.columns:
        st.error("未找到风向或风速数据")
        return

    col1, col2 = st.columns(2)

    with col1:
        # 风玫瑰图
        df_sample = df.sample(min(5000, len(df)))
        fig = px.bar_polar(df_sample, r="predicted_wind_speed", theta="wind_direction",
                           color="predicted_wind_speed", template="plotly_dark",
                           color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 风向频率分布
        direction_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        direction_labels = ['北', '东北', '东', '东南', '南', '西南', '西', '西北']

        df['wind_direction_cat'] = pd.cut(df['wind_direction'], bins=direction_bins, labels=direction_labels)
        direction_count = df['wind_direction_cat'].value_counts()

        fig = px.bar(x=direction_count.index, y=direction_count.values,
                     title="风向频率分布", labels={'x': '风向', 'y': '频次'})
        st.plotly_chart(fig, use_container_width=True)

    # 风向稳定性分析
    if 'wind_direction_std' in df.columns:
        st.subheader("📏 风向稳定性分析")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("平均风向标准差", f"{df['wind_direction_std'].mean():.1f}°")
            st.metric("最大风向变化", f"{df['wind_direction_std'].max():.1f}°")

        with col2:
            fig = px.histogram(df, x='wind_direction_std', nbins=30,
                               title="风向标准差分布")
            st.plotly_chart(fig, use_container_width=True)


def spatial_analysis_enhanced(df):
    st.subheader("🗺️ 空间分布分析")

    if 'lat' not in df.columns or 'lon' not in df.columns:
        st.error("未找到地理位置数据")
        return

    # 选择颜色映射变量
    color_options = ['predicted_wind_speed', 'elevation', 'temperature_c', 'relative_humidity']
    available_color_options = [col for col in color_options if col in df.columns]

    color_by = st.selectbox("按颜色显示:", available_color_options)

    # 对大数据集进行采样
    if len(df) > 1000:
        sample_df = df.sample(n=1000, random_state=42)
    else:
        sample_df = df

    # 创建散点图
    fig = px.scatter_mapbox(sample_df,
                            lat='lat',
                            lon='lon',
                            color=color_by,
                            size='predicted_wind_speed' if 'predicted_wind_speed' in df.columns else None,
                            hover_data=['elevation', 'slope'] if all(
                                col in df.columns for col in ['elevation', 'slope']) else None,
                            color_continuous_scale='viridis',
                            zoom=10,
                            title=f"风电场空间分布 - 按{color_by}着色")

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    # 地形分析
    if 'elevation' in df.columns and 'slope' in df.columns:
        st.subheader("🏔️ 地形特征分析")

        col1, col2 = st.columns(2)

        with col1:
            # 移除trendline参数，使用普通散点图
            fig = px.scatter(df, x='elevation', y='predicted_wind_speed',
                             title="海拔与风速关系")
            # 手动添加趋势线（使用numpy计算）
            try:
                # 计算线性趋势
                z = np.polyfit(df['elevation'], df['predicted_wind_speed'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df['elevation'].min(), df['elevation'].max(), 100)
                y_trend = p(x_trend)

                fig.add_trace(go.Scatter(x=x_trend, y=y_trend,
                                         mode='lines',
                                         name='趋势线',
                                         line=dict(color='red', dash='dash')))
            except:
                pass  # 如果计算趋势线失败，继续显示散点图

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 移除trendline参数，使用普通散点图
            fig = px.scatter(df, x='slope', y='predicted_wind_speed',
                             title="坡度与风速关系")
            # 手动添加趋势线
            try:
                z = np.polyfit(df['slope'], df['predicted_wind_speed'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df['slope'].min(), df['slope'].max(), 100)
                y_trend = p(x_trend)

                fig.add_trace(go.Scatter(x=x_trend, y=y_trend,
                                         mode='lines',
                                         name='趋势线',
                                         line=dict(color='red', dash='dash')))
            except:
                pass

            st.plotly_chart(fig, use_container_width=True)


def correlation_analysis_enhanced(df):
    st.subheader("🔗 多变量相关性分析")

    corr = compute_correlation(df)
    if corr is None:
        st.warning("数据不足，无法进行相关性分析")
        return

    # 交互式相关性矩阵
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="风电场变量相关性热力图",
                    width=800, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 散点图矩阵
    st.subheader("📈 散点图矩阵")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 优先选择气象相关变量
    priority_cols = ['predicted_wind_speed', 'temperature_c', 'relative_humidity', 'gust_speed', 'elevation']
    default_cols = [col for col in priority_cols if col in numeric_cols][:4]

    selected_cols = st.multiselect("选择要分析的变量:", numeric_cols,
                                   default=default_cols)

    if len(selected_cols) >= 2:
        # 移除trendline参数
        fig = px.scatter_matrix(df[selected_cols], height=800)
        st.plotly_chart(fig, use_container_width=True)

    # 重点相关性分析
    if 'predicted_wind_speed' in corr.columns:
        st.subheader("🎯 与风速的相关性分析")

        wind_corr = corr['predicted_wind_speed'].sort_values(ascending=False)
        # 排除自身相关性
        strong_corr = wind_corr[(abs(wind_corr) > 0.1) & (wind_corr != 1.0)]

        # 使用进度条展示相关性强度
        for var, corr_val in strong_corr.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{var}**")
                st.progress(abs(corr_val), text=f"相关性强度: {corr_val:.3f}")
            with col2:
                if corr_val > 0:
                    st.metric("方向", "正相关", delta=f"{corr_val:.3f}")
                else:
                    st.metric("方向", "负相关", delta=f"{corr_val:.3f}")


def data_overview(df, datetime_col):
    st.subheader("📋 数据概览")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**数据基本信息:**")
        info_dict = {
            "总记录数": len(df),
            "时间范围": f"{df[datetime_col].min()} 至 {df[datetime_col].max()}",
            "数据点数量": df['point_id'].nunique() if 'point_id' in df.columns else "N/A",
            "时间分辨率": f"{(df[datetime_col].iloc[1] - df[datetime_col].iloc[0]).total_seconds() / 60:.0f} 分钟"
        }
        st.json(info_dict)

    with col2:
        st.write("**缺失值统计:**")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            for col, count in missing_data[missing_data > 0].items():
                st.write(f"- {col}: {count} 个缺失值 ({count / len(df) * 100:.1f}%)")
        else:
            st.success("✅ 无缺失值")

    # 数据预览
    st.subheader("🔍 数据预览")
    st.dataframe(df.head(100), use_container_width=True)

    # 变量分布概览
    st.subheader("📊 数值变量分布")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)