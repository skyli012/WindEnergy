import streamlit as st
import pandas as pd
import numpy as np


def data_import_page():
    st.title("🌬️ 风电场风速预测数据导入")

    st.markdown("""
    上传风电场风速预测数据集（CSV格式）。  
    系统将识别时间、地理位置、气象条件和风速等字段。
    """)

    uploaded_file = st.file_uploader("📁 选择CSV文件上传", type="csv")

    if uploaded_file is not None:
        try:
            # ====== 读取数据 ======
            df = pd.read_csv(uploaded_file)

            # ====== 标准化列名 ======
            df.columns = [c.strip().lower() for c in df.columns]

            # ====== 检查关键列 ======
            required_cols = [
                "timestamp", "predicted_wind_speed", "lat", "lon"
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"❌ 缺少必要列: {', '.join(missing)}")
                st.write("检测到的列名：", list(df.columns))
                return

            # ====== 时间解析 ======
            # 尝试多种时间格式
            time_parsed = False
            for time_col in ['timestamp', 'time', 'datetime', 'date']:
                if time_col in df.columns:
                    try:
                        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                        df = df.dropna(subset=[time_col])
                        # 重命名为统一的timestamp
                        if time_col != 'timestamp':
                            df = df.rename(columns={time_col: 'timestamp'})
                        time_parsed = True
                        st.success("🕒 时间列已成功解析为日期时间格式。")
                        break
                    except Exception as e:
                        continue

            if not time_parsed:
                st.error("❌ 无法解析时间列，请确保时间格式正确")
                return

            # ====== 数据清洗和类型转换 ======
            # 确保数值列的正确类型
            numeric_columns = [
                'predicted_wind_speed', 'lat', 'lon', 'elevation', 'slope',
                'relative_humidity', 'temperature_c', 'wind_direction',
                'gust_direction', 'gust_speed', 'wind_direction_std', 'rainfall_mm',
                'grid_proximity', 'road_distance', 'residential_distance',
                'heritage_distance', 'geology_distance', 'water_distance', 'cost'
            ]

            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # ====== 处理重复和时间排序 ======
            df = df.sort_values('timestamp')
            df = df.reset_index(drop=True)

            # ====== 保存到会话状态 ======
            st.session_state["dataset"] = df

            # ====== 数据摘要 ======
            st.subheader("📊 数据摘要")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("记录数", f"{len(df):,}")
            with col2:
                st.metric("字段数", len(df.columns))
            with col3:
                st.metric("时间范围",
                          f"{df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} → {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
            with col4:
                st.metric("数据点数量", f"{df['point_id'].nunique()}" if 'point_id' in df.columns else "N/A")

            # ====== 数据预览 ======
            st.subheader("🔍 数据预览")
            st.dataframe(df.head(10), use_container_width=True)

            # ====== 字段信息 ======
            st.subheader("🧭 字段信息")
            info_df = pd.DataFrame({
                "字段名": df.columns,
                "数据类型": df.dtypes.values,
                "非空数量": df.count().values,
                "缺失率": (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(info_df, use_container_width=True)

            # ====== 基本统计 ======
            st.subheader("📈 基本统计（数值字段）")
            numeric_df = df.select_dtypes(include=[np.number])
            st.dataframe(numeric_df.describe(), use_container_width=True)

            # ====== 风速分析 ======
            st.subheader("🌪️ 风速分析")
            if 'predicted_wind_speed' in df.columns:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("平均风速", f"{df['predicted_wind_speed'].mean():.2f} m/s")
                with col2:
                    st.metric("最大风速", f"{df['predicted_wind_speed'].max():.2f} m/s")
                with col3:
                    st.metric("最小风速", f"{df['predicted_wind_speed'].min():.2f} m/s")
                with col4:
                    st.metric("风速标准差", f"{df['predicted_wind_speed'].std():.2f} m/s")

                # 风速分布
                import plotly.express as px
                fig = px.histogram(df, x='predicted_wind_speed',
                                   title="风速分布直方图",
                                   nbins=30)
                st.plotly_chart(fig, use_container_width=True)

            # ====== 地理位置分析 ======
            st.subheader("🗺️ 地理位置分析")
            if all(col in df.columns for col in ['lat', 'lon']):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("纬度范围", f"{df['lat'].min():.4f} - {df['lat'].max():.4f}")
                with col2:
                    st.metric("经度范围", f"{df['lon'].min():.4f} - {df['lon'].max():.4f}")

                # 简单的地理分布散点图
                if len(df) > 1000:
                    # 对大数据集进行采样
                    sample_df = df.sample(n=1000, random_state=42)
                else:
                    sample_df = df

                fig = px.scatter(sample_df, x='lon', y='lat',
                                 color='predicted_wind_speed' if 'predicted_wind_speed' in df.columns else None,
                                 title="地理位置分布",
                                 color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)

            # ====== 时间序列分析 ======
            st.subheader("⏰ 时间序列分析")
            if 'timestamp' in df.columns and 'predicted_wind_speed' in df.columns:
                # 按小时聚合查看日变化
                df_hourly = df.copy()
                df_hourly['hour'] = df_hourly['timestamp'].dt.hour
                hourly_avg = df_hourly.groupby('hour')['predicted_wind_speed'].mean().reset_index()

                fig = px.line(hourly_avg, x='hour', y='predicted_wind_speed',
                              title="风速日变化趋势",
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)

            # ====== 气象条件相关性 ======
            st.subheader("🔗 气象条件相关性")
            weather_cols = ['temperature_c', 'relative_humidity', 'wind_direction', 'gust_speed', 'rainfall_mm']
            available_weather_cols = [col for col in weather_cols if col in df.columns]

            if available_weather_cols and 'predicted_wind_speed' in df.columns:
                # 计算相关性矩阵
                corr_matrix = df[['predicted_wind_speed'] + available_weather_cols].corr()
                # 只保留与风速的相关性
                wind_corr = corr_matrix['predicted_wind_speed'].drop('predicted_wind_speed').sort_values(key=abs,
                                                                                                         ascending=False)

                fig = px.bar(x=wind_corr.values, y=wind_corr.index,
                             orientation='h',
                             title="各气象因素与风速的相关性",
                             labels={'x': '相关系数', 'y': '气象因素'})
                st.plotly_chart(fig, use_container_width=True)

            st.success("✅ 数据导入成功！已准备好进行风速预测分析。")

        except Exception as e:
            st.error(f"⚠️ 文件解析出错: {str(e)}")
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")

    else:
        st.info("👆 请上传风电场风速预测数据 CSV 文件。")
        st.subheader("📄 预期数据格式示例")
        st.markdown("""
        | 字段名 | 示例值 | 说明 |
        |--------|--------|------|
        | point_id | 0 | 点位ID |
        | timestamp | 2024/1/1 0:00 | 时间戳 |
        | lat | 32.48787 | 纬度 |
        | lon | 35.43678 | 经度 |
        | elevation | 424.1 | 海拔(m) |
        | slope | 16.5 | 坡度(°) |
        | predicted_wind_speed | 5.65 | 预测风速(m/s) |
        | relative_humidity | 88.4 | 相对湿度(%) |
        | temperature_c | 9.2 | 温度(°C) |
        | wind_direction | 131 | 风向(°) |
        | gust_direction | 137.9 | 阵风风向(°) |
        | gust_speed | 7.53 | 阵风速度(m/s) |
        | wind_direction_std | 10 | 风向标准差 |
        | rainfall_mm | 0 | 降雨量(mm) |
        | grid_proximity | 0.52 | 电网接近度 |
        | road_distance | 659.7 | 道路距离(m) |
        | residential_distance | 471.3 | 居民区距离(m) |
        | heritage_distance | 178.4 | 遗产距离(m) |
        | geology_distance | 401.1 | 地质距离(m) |
        | water_distance | 780.3 | 水域距离(m) |
        | cost | 1200.5 | 成本 |
        """)