import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# 配置：你的 4 个预测模型 + 4 个优化算法
# -------------------------------
PREDICT_MODELS = ["RandomForest", "LinearRegression", "LSTM", "GRU"]
OPT_ALGORITHMS = ["GA", "SA", "PSO", "Greedy"]


def prediction_optimization_comparison_page():

    st.title("📊 预测模型 × 优化算法 全组合对比分析")

    st.markdown("""
    此模块会对比 4 个预测模型（RF / LR / LSTM / GRU）
    与 4 个优化算法（GA / SA / PSO / Greedy）的所有 16 种组合，
    计算关键指标并推荐最优组合。
    """)

    st.markdown("---")

    # -----------------------------
    # 1. 用户上传 16 个 CSV 文件
    # -----------------------------
    uploaded_files = st.file_uploader(
        "上传所有组合的 CSV 结果（16 个，每个文件包含 wind_power, grid_power, battery_power, storage_soc）",
        type=["csv"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("请上传 16 个预测+优化组合的结果文件")
        return

    # 放入 dict
    results = {}
    for file in uploaded_files:
        df = pd.read_csv(file)
        name = file.name.replace(".csv", "")
        results[name] = df

    st.success(f"已成功加载 {len(results)} 个结果文件")

    st.markdown("---")

    # -----------------------------
    # 2. 计算关键指标
    # -----------------------------
    st.subheader("📈 综合性能指标")

    records = []

    for name, df in results.items():

        # 时间步长计算
        delta_h = (pd.to_datetime(df["timestamp"].iloc[1]) -
                   pd.to_datetime(df["timestamp"].iloc[0])).seconds / 3600

        total_wind = df["wind_power"].sum() * delta_h
        total_grid = df["grid_power"].sum() * delta_h
        curtailment = total_wind - total_grid
        curtail_rate = curtailment / total_wind * 100 if total_wind > 0 else 0

        soc_range = df["storage_soc"].max() - df["storage_soc"].min()

        max_grid = df["grid_power"].max()

        records.append({
            "组合": name,
            "总并网能量(kWh)": total_grid,
            "弃风率(%)": curtail_rate,
            "SOC波动": soc_range,
            "最大并网(kW)": max_grid
        })

    metrics_df = pd.DataFrame(records)
    st.dataframe(metrics_df)

    st.markdown("---")

    # -----------------------------
    # 3. 雷达图可视化
    # -----------------------------
    st.subheader("📊 雷达图对比")

    categories = ["总并网能量(kWh)", "弃风率(%)", "SOC波动", "最大并网(kW)"]

    fig = go.Figure()

    for _, row in metrics_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[
                row["总并网能量(kWh)"],
                100 - row["弃风率(%)"],  # 越低越好 → 反转
                1 - row["SOC波动"],       # 越稳定越好 → 反转
                row["最大并网(kW)"],
            ],
            theta=categories,
            fill='toself',
            name=row["组合"]
        ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), height=600)
    st.plotly_chart(fig)

    # -----------------------------
    # 4. 计算综合得分，给出最终推荐
    # -----------------------------
    st.subheader("🏆 最佳组合推荐")

    # 归一化评分
    metrics_df["综合得分"] = (
        (metrics_df["总并网能量(kWh)"] / metrics_df["总并网能量(kWh)"].max()) * 0.4 +
        ((100 - metrics_df["弃风率(%)"]) / (100 - metrics_df["弃风率(%)"]).max()) * 0.3 +
        ((1 - metrics_df["SOC波动"]) / (1 - metrics_df["SOC波动"]).max()) * 0.2 +
        (metrics_df["最大并网(kW)"] / metrics_df["最大并网(kW)"].max()) * 0.1
    )

    best = metrics_df.loc[metrics_df["综合得分"].idxmax()]

    st.success(f"""
    ### 🏅 最优组合：**{best['组合']}**
    综合得分：**{best['综合得分']:.4f}**

    - 总并网能量：{best['总并网能量(kWh)']:.1f} kWh  
    - 弃风率：{best['弃风率(%)']:.2f}%  
    - SOC波动：{best['SOC波动']:.3f}  
    - 最大并网功率：{best['最大并网(kW)']:.1f} kW  
    """)

    st.markdown("---")
