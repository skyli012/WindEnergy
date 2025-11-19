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
    return df.groupby('month')['wind_speed_ms'].mean().reset_index()


@st.cache_data
def compute_correlation(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None
    return df[numeric_cols].corr()


# ========== 主页面 ==========
def data_analysis_page():
    st.title("🌬️ 气象数据分析中心")

    if 'dataset' not in st.session_state:
        st.warning("⚠️ 请先在数据导入页面导入气象数据")
        return

    df = st.session_state['dataset'].copy()

    # 自动识别时间列
    datetime_col = next(
        (col for col in df.columns if 'time' in col.lower() or 'datatime' in col.lower() or 'date' in col.lower()),
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
        if 'wind_speed_ms' in df.columns:
            st.metric("平均风速", f"{df['wind_speed_ms'].mean():.1f} m/s")
    with col4:
        st.metric("数据维度", f"{len(df.columns)} 个字段")

    st.markdown("---")

    # 使用标签页组织分析内容
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 时间趋势",
        "🌪️ 风速分析",
        "🧭 风向分析",
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
        correlation_analysis_enhanced(df)
    with tab5:
        data_overview(df, datetime_col)


# ================= 增强的分析模块 ====================
def temporal_analysis_enhanced(df, datetime_col):
    st.subheader("📊 时间序列分析")

    if 'wind_speed_ms' not in df.columns:
        st.error("未找到风速数据")
        return

    # 选择时间粒度
    time_granularity = st.radio("时间粒度:", ["月", "周", "日"], horizontal=True)

    if time_granularity == "月":
        df['time_period'] = df[datetime_col].dt.to_period('M').dt.to_timestamp()
        title = "月平均风速趋势"
    elif time_granularity == "周":
        df['time_period'] = df[datetime_col].dt.to_period('W').dt.to_timestamp()
        title = "周平均风速趋势"
    else:
        df['time_period'] = df[datetime_col].dt.date
        title = "日平均风速趋势"

    # 多变量趋势图
    cols_to_plot = [col for col in ['wind_speed_ms', 'temperature_c', 'humidity', 'pressure_millibars'] if
                    col in df.columns]

    if len(cols_to_plot) > 1:
        fig = make_subplots(rows=len(cols_to_plot), cols=1,
                            subplot_titles=cols_to_plot,
                            vertical_spacing=0.05)

        for i, col in enumerate(cols_to_plot, 1):
            period_avg = df.groupby('time_period')[col].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=period_avg['time_period'], y=period_avg[col],
                           name=col, line=dict(width=2)),
                row=i, col=1
            )

        fig.update_layout(height=300 * len(cols_to_plot), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        period_avg = df.groupby('time_period')['wind_speed_ms'].mean().reset_index()
        fig = px.line(period_avg, x='time_period', y='wind_speed_ms',
                      title=title, line_shape='spline')
        st.plotly_chart(fig, use_container_width=True)

    # 季节性分析
    st.subheader("🌤️ 季节性分析")
    df['month'] = df[datetime_col].dt.month
    monthly_stats = df.groupby('month')['wind_speed_ms'].agg(['mean', 'std', 'min', 'max']).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_stats['month'], y=monthly_stats['mean'],
                             name='平均值', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=monthly_stats['month'], y=monthly_stats['mean'] + monthly_stats['std'],
                             name='+1标准差', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=monthly_stats['month'], y=monthly_stats['mean'] - monthly_stats['std'],
                             name='-1标准差', line=dict(color='red', dash='dash')))

    fig.update_layout(title="月度风速统计", xaxis_title="月份", yaxis_title="风速 (m/s)")
    st.plotly_chart(fig, use_container_width=True)


def windspeed_analysis_enhanced(df):
    st.subheader("🌪️ 风速统计分析")

    if 'wind_speed_ms' not in df.columns:
        st.error("未找到风速数据")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        # 分布直方图 + 密度曲线
        fig = px.histogram(df, x='wind_speed_ms', nbins=30,
                           marginal="box", opacity=0.7,
                           title="风速分布直方图")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 统计指标卡片
        stats = df['wind_speed_ms'].describe()

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

    df['wind_level'] = pd.cut(df['wind_speed_ms'], bins=wind_bins, labels=wind_labels)
    wind_level_count = df['wind_level'].value_counts().sort_index()

    fig = px.pie(values=wind_level_count.values, names=wind_level_count.index,
                 title="风速等级分布饼图")
    st.plotly_chart(fig, use_container_width=True)



def wind_direction_analysis_enhanced(df):
    st.subheader("🧭 风向综合分析")

    if 'wind_bearing_degrees' not in df.columns or 'wind_speed_ms' not in df.columns:
        st.error("未找到风向或风速数据")
        return

    col1, col2 = st.columns(2)

    with col1:
        # 风玫瑰图
        df_sample = df.sample(min(5000, len(df)))
        fig = px.bar_polar(df_sample, r="wind_speed_ms", theta="wind_bearing_degrees",
                           color="wind_speed_ms", template="plotly_dark",
                           color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 风向频率分布
        direction_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        direction_labels = ['北', '东北', '东', '东南', '南', '西南', '西', '西北']

        df['wind_direction'] = pd.cut(df['wind_bearing_degrees'], bins=direction_bins, labels=direction_labels)
        direction_count = df['wind_direction'].value_counts()

        fig = px.bar(x=direction_count.index, y=direction_count.values,
                     title="风向频率分布", labels={'x': '风向', 'y': '频次'})
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
                    title="气象变量相关性热力图",
                    width=800, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 散点图矩阵
    st.subheader("📈 散点图矩阵")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect("选择要分析的变量:", numeric_cols,
                                   default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols)

    if len(selected_cols) >= 2:
        fig = px.scatter_matrix(df[selected_cols], height=800)
        st.plotly_chart(fig, use_container_width=True)

    # 重点相关性分析
    if 'wind_speed_ms' in corr.columns:
        st.subheader("🎯 与风速的相关性分析")

        wind_corr = corr['wind_speed_ms'].sort_values(ascending=False)
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
        st.json({
            "总记录数": len(df),
            "时间范围": f"{df[datetime_col].min()} 至 {df[datetime_col].max()}",
            "数据类型分布": df.dtypes.value_counts().to_dict()
        })

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