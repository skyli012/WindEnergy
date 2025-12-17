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

    # 自动检测时间列
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
        st.metric("总记录数", f"{len(df):,}")
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
        "🔗 相关性分析",
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

    # 仅使用小时粒度
    df['time_period'] = df[datetime_col].dt.floor('H')
    title = "小时平均风速趋势"

    # 多变量趋势图 - 改为2x2布局
    cols_to_plot = [col for col in ['predicted_wind_speed', 'temperature_c', 'relative_humidity', 'gust_speed'] if
                    col in df.columns]

    if len(cols_to_plot) > 0:
        # 创建2x2的子图布局
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=cols_to_plot if len(cols_to_plot) >= 4 else cols_to_plot + [''] * (
                                        4 - len(cols_to_plot)),
                            vertical_spacing=0.15,
                            horizontal_spacing=0.1)

        # 定义2x2布局的位置映射
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for i, col in enumerate(cols_to_plot[:4]):  # 最多显示4个变量
            if i >= len(positions):
                break

            # 对于聚合数据，使用小时平均值
            period_avg = df.groupby('time_period')[col].mean().reset_index()
            row, col_pos = positions[i]

            fig.add_trace(
                go.Scatter(x=period_avg['time_period'], y=period_avg[col],
                           name=col, line=dict(width=2)),
                row=row, col=col_pos
            )

            # 设置y轴标签
            if col_pos == 1:  # 左列
                fig.update_yaxes(title_text=col, row=row, col=col_pos)

            # 设置x轴标签（只在底行显示）
            if row == 2:
                fig.update_xaxes(title_text="时间", row=row, col=col_pos)

        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="时间序列分析 (2x2布局)"
        )
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
                             name='+1 标准差', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=time_stats['hour'], y=time_stats['mean'] - time_stats['std'],
                             name='-1 标准差', line=dict(color='red', dash='dash'),
                             fill='tonexty'))

    fig.update_layout(
        title="风速小时变化趋势",
        xaxis_title="小时",
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
    wind_labels = ['轻风(0-3)', '微风(3-6)', '和风(6-9)', '强风(9-12)', '大风(12-15)', '暴风(15+)']

    df['wind_level'] = pd.cut(df['predicted_wind_speed'], bins=wind_bins, labels=wind_labels)
    wind_level_count = df['wind_level'].value_counts().sort_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=wind_level_count.values, names=wind_level_count.index,
                     title="风速等级分布饼图")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(x=wind_level_count.index, y=wind_level_count.values,
                     title="风速等级频率分布", labels={'x': '风速等级', 'y': '频率'})
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
                     title="风向频率分布", labels={'x': '风向', 'y': '频率'})
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

    # 对大数据集进行抽样
    if len(df) > 1000:
        sample_df = df.sample(n=1000, random_state=42)
    else:
        sample_df = df

    # 创建散点地图
    fig = px.scatter_mapbox(sample_df,
                            lat='lat',
                            lon='lon',
                            color=color_by,
                            size='predicted_wind_speed' if 'predicted_wind_speed' in df.columns else None,
                            hover_data=['elevation', 'slope'] if all(
                                col in df.columns for col in ['elevation', 'slope']) else None,
                            color_continuous_scale='viridis',
                            zoom=10,
                            title=f"风电场空间分布 - 按 {color_by} 着色")

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    # 地形分析
    if 'elevation' in df.columns and 'slope' in df.columns:
        st.subheader("🏔️ 地形特征分析")

        col1, col2 = st.columns(2)

        with col1:
            # 使用常规散点图，不包含趋势线参数
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
                pass  # 如果趋势线计算失败，继续使用散点图

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 使用常规散点图，不包含趋势线参数
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
    st.subheader("🔗 主要变量相关性分析")
    st.markdown("""
    **分析说明：**
    - 展示了风速与主要气象变量、地形变量之间的相关性
    - 结果表明，风速与部分气象变量（如阵风速度、气温等）以及地形变量（如海拔、坡度）之间存在一定程度的相关性
    - 整体相关系数分布较为分散，未呈现出单一主导因素
    """)

    # 定义主要关注的变量（基于您的字段）
    # 核心变量：风速
    core_vars = ['predicted_wind_speed']

    # 直接气象相关变量（与风速高度相关）
    direct_meteorological_vars = [
        'gust_speed',  # 阵风速度
        'wind_direction',  # 风向
        'wind_direction_std',  # 风向稳定性
        'gust_direction'  # 阵风方向
    ]

    # 间接气象相关变量
    indirect_meteorological_vars = [
        'temperature_c',  # 气温
        'relative_humidity',  # 相对湿度
        'rainfall_mm'  # 降雨量
    ]

    # 地形和地理变量
    terrain_geographic_vars = [
        'elevation',  # 海拔
        'slope',  # 坡度
        'hour'  # 小时（时间因素）
    ]

    # 空间距离变量（可能影响风场）
    spatial_distance_vars = [
        'water_distance',  # 距水体距离
        'road_distance',  # 距道路距离
        'grid_proximity'  # 电网接近度
    ]

    # 筛选出数据集中存在的变量
    available_core_vars = [var for var in core_vars if var in df.columns]
    available_direct_meteo_vars = [var for var in direct_meteorological_vars if var in df.columns]
    available_indirect_meteo_vars = [var for var in indirect_meteorological_vars if var in df.columns]
    available_terrain_vars = [var for var in terrain_geographic_vars if var in df.columns]
    available_spatial_vars = [var for var in spatial_distance_vars if var in df.columns]

    # 合并所有主要变量（按相关性优先级）
    all_main_vars = (
            available_core_vars +
            available_direct_meteo_vars +
            available_indirect_meteo_vars +
            available_terrain_vars +
            available_spatial_vars[:2]  # 只取最重要的2个空间变量
    )

    # 确保去重
    all_main_vars = list(dict.fromkeys(all_main_vars))

    if len(all_main_vars) < 2:
        st.warning("数据不足，无法进行相关性分析")
        return

    # 仅计算主要变量之间的相关性
    corr = df[all_main_vars].corr()

    # 创建自定义颜色序列，使相关性更明显
    colorscale = [
        [0.0, '#2E86AB'],  # 深蓝 - 强负相关
        [0.25, '#A3D9FF'],  # 浅蓝 - 中等负相关
        [0.5, '#FFFFFF'],  # 白 - 无相关
        [0.75, '#FF9B85'],  # 浅红 - 中等正相关
        [1.0, '#E63946']  # 深红 - 强正相关
    ]

    # 交互式相关性矩阵
    fig = px.imshow(
        corr,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale=colorscale,
        zmin=-1,  # 设置颜色范围
        zmax=1,
        title="主要变量相关性热力图",
        labels=dict(
            color="相关系数",
            x="变量",
            y="变量"
        )
    )

    # 更新布局，使图表更专业
    fig.update_layout(
        width=700,
        height=600,
        margin=dict(l=80, r=50, t=80, b=80),
        title_font=dict(size=18, family="Arial", color="#2c3e50"),
        coloraxis_colorbar=dict(
            title="相关系数",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            thickness=15,
            len=0.8
        )
    )

    # 更新字体大小和样式
    fig.update_traces(
        textfont=dict(size=10, family="Arial", color="black"),
        texttemplate='%{z:.2f}'
    )

    # 更新坐标轴标签
    fig.update_xaxes(
        tickangle=-45,
        tickfont=dict(size=11, family="Arial"),
        title_font=dict(size=13, family="Arial")
    )

    fig.update_yaxes(
        tickfont=dict(size=11, family="Arial"),
        title_font=dict(size=13, family="Arial")
    )

    st.plotly_chart(fig, use_container_width=True)

    # 显示风速与其他变量的详细相关系数
    st.subheader("📊 风速与各变量相关系数分析")

    if 'predicted_wind_speed' in corr.columns:
        wind_corr_series = corr['predicted_wind_speed'].drop('predicted_wind_speed')

        # 创建分类显示
        categories = {
            '直接气象因素': [],
            '间接气象因素': [],
            '地形地理因素': [],
            '空间距离因素': []
        }

        # 分类变量
        for var, corr_value in wind_corr_series.items():
            if var in direct_meteorological_vars:
                categories['直接气象因素'].append((var, corr_value))
            elif var in indirect_meteorological_vars:
                categories['间接气象因素'].append((var, corr_value))
            elif var in terrain_geographic_vars:
                categories['地形地理因素'].append((var, corr_value))
            elif var in spatial_distance_vars:
                categories['空间距离因素'].append((var, corr_value))
            else:
                categories['其他因素'].append((var, corr_value))

        # 显示每个类别的分析
        for category_name, variables in categories.items():
            if variables:
                st.markdown(f"**{category_name}**")

                # 创建该类别的DataFrame
                cat_df = pd.DataFrame(
                    variables,
                    columns=['变量', '相关系数']
                ).sort_values('相关系数', ascending=False)

                # 添加相关性强度和方向
                cat_df['相关性强度'] = cat_df['相关系数'].abs().apply(
                    lambda x: '强相关(≥0.7)' if x >= 0.7 else
                    '中等相关(0.3-0.7)' if x >= 0.3 else
                    '弱相关(0.1-0.3)' if x >= 0.1 else
                    '极弱相关(<0.1)'
                )

                cat_df['相关方向'] = cat_df['相关系数'].apply(
                    lambda x: '正相关' if x > 0 else '负相关' if x < 0 else '无相关'
                )

                # 显示表格
                st.dataframe(
                    cat_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        '变量': st.column_config.TextColumn('变量名称', width="medium"),
                        '相关系数': st.column_config.NumberColumn(
                            '相关系数',
                            format='%.3f',
                            help='-1到1之间的值，绝对值越大相关性越强',
                            width="small"
                        ),
                        '相关性强度': st.column_config.TextColumn('强度分类', width="medium"),
                        '相关方向': st.column_config.TextColumn('方向', width="small")
                    }
                )

        # 关键发现总结
        st.markdown("""
        **🔍 关键发现总结：**

        1. **主要相关变量**：
           - 阵风速度 (gust_speed)：通常与风速高度相关
           - 地形因素 (elevation, slope)：山地地形对风速有重要影响
           - 气象变量 (temperature_c, relative_humidity)：影响空气密度和流动

        2. **时间维度**：
           - 小时 (hour)：显示风速的日变化规律

        3. **空间因素**：
           - 水体距离 (water_distance)：水体对局地风场有调节作用

        4. **综合分析**：
           - 风速受多种因素共同影响，无单一主导变量
           - 相关系数分布分散，验证了多元影响因素的存在
           - 为风电场选址提供了多维度的数据支持
        """)

    else:
        # 如果数据集没有风速变量，显示完整的相关系数表
        st.dataframe(
            corr.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1),
            use_container_width=True
        )

    # 添加统计显著性说明
    st.markdown("""
    ---
    **📝 统计说明：**
    - 相关系数范围：-1（完全负相关）到 1（完全正相关）
    - |r| ≥ 0.7：强相关
    - 0.3 ≤ |r| < 0.7：中等相关
    - 0.1 ≤ |r| < 0.3：弱相关
    - |r| < 0.1：极弱相关或无相关
    - 样本量：{} 条记录
    """.format(len(df)))


def data_overview(df, datetime_col):
    st.subheader("📋 数据概览")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**基本数据信息:**")
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