import streamlit as st
import pandas as pd
# ======================================================
# 🔍 数据质量检查函数
# ======================================================
def check_data_quality(df):
    """检查风速数据质量"""
    st.markdown("#### 🔍 数据质量检查")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "predicted_wind_speed" in df.columns:
            wind_speed = df["predicted_wind_speed"]
            st.metric("风速范围", f"{wind_speed.min():.1f} - {wind_speed.max():.1f} m/s")
            if wind_speed.std() < 0.5:
                st.error("❌ 风速数据变化太小")

    with col2:
        if "valid" in df.columns:
            valid_count = df["valid"].sum()
            total_count = len(df)
            valid_ratio = valid_count / total_count * 100
            st.metric("有效点位", f"{valid_count}/{total_count} ({valid_ratio:.1f}%)")
            if valid_ratio < 10:
                st.error("❌ 有效点位过少")

    with col3:
        if "wind_power_density" in df.columns:
            power_density = df["wind_power_density"]
            st.metric("风能密度", f"{power_density.mean():.0f} W/m²")