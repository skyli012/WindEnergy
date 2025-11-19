import streamlit as st
import pandas as pd
import numpy as np

def data_import_page():
    st.title("🌬️ Szeged（匈牙利）2006–2016 气象数据导入")

    st.markdown("""
    上传已清洗好的 **Szeged（匈牙利）2006–2016 气象数据集**（CSV格式）。  
    系统将直接识别时间、温度、湿度、风速、能见度等字段。
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
                "datatime", "temperature_c", "apparent_temperature_c", "humidity",
                "pressure_millibars", "wind_speed_ms", "visibility_ms",
                "summary_code", "precip_type_code"
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"❌ 缺少必要列: {', '.join(missing)}")
                st.write("检测到的列名：", list(df.columns))
                return

            # ====== 时间解析 ======
            df['datatime'] = pd.to_datetime(df['datatime'], errors='coerce')
            df = df.dropna(subset=['datatime'])
            st.success("🕒 时间列已成功解析为日期时间格式。")

            # ====== 单位标准化 ======
            # 湿度：如果是 0~1 的比例，转换为百分比
            if df['humidity'].max() <= 1:
                df['humidity'] = df['humidity'] * 100

            # ====== 保存到会话状态 ======
            st.session_state["dataset"] = df

            # ====== 数据摘要 ======
            st.subheader("📊 数据摘要")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("记录数", f"{len(df):,}")
            with col2:
                st.metric("字段数", len(df.columns))
            with col3:
                st.metric("时间范围", f"{df['datatime'].min().strftime('%Y-%m-%d')} → {df['datatime'].max().strftime('%Y-%m-%d')}")

            # ====== 数据预览 ======
            st.subheader("🔍 数据预览")
            st.dataframe(df.head(10), use_container_width=True)

            # ====== 字段信息 ======
            st.subheader("🧭 字段信息")
            info_df = pd.DataFrame({
                "字段名": df.columns,
                "数据类型": df.dtypes.values,
                "非空数量": df.count().values
            })
            st.dataframe(info_df, use_container_width=True)


            # ====== 基本统计 ======
            st.subheader("📈 基本统计（数值字段）")
            st.dataframe(df.select_dtypes(include=[np.number]).describe(), use_container_width=True)

            st.success("✅ 数据导入成功！已准备好进行风速预测或分析。")

        except Exception as e:
            st.error(f"⚠️ 文件解析出错: {str(e)}")

    else:
        st.info("👆 请上传已清洗好的 Szeged 气象数据 CSV 文件。")
        st.subheader("📄 预期数据格式示例")
        st.markdown("""
        | 字段名 | 示例值 | 说明 |
        |--------|--------|------|
        | datatime | 2006/3/31 22:00 | 时间戳 (年月日小时) |
        | temperature_c | 9.47 | 温度(°C) |
        | apparent_temperature_c | 7.38 | 体感温度(°C) |
        | humidity | 89 | 湿度(%) |
        | wind_speed_ms | 3.92 | 风速(m/s) |
        | wind_bearing_degrees | 251 | 风向(°) |
        | pressure_millibars | 1015.13 | 气压(hPa) |
        | visibility_ms | 4.39 | 能见度(m/s) |
        | summary_code | 3 | 天气编码 |
        | precip_type_code | 1 | 降水类型编码 |
        """)
