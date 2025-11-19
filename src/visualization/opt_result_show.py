import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point

from src.utils.create_map import create_fengjie_base_map


# ======================================================
# 🚀 显示优化结果 - 数据分析部分
# ======================================================
def display_optimization_result(result, df):
    st.subheader(f"🚀 最优风机布局与算法收敛分析（{result['algorithm']}）")

    sol = result["solution"]
    if not sol:
        st.error("❌ 没有找到有效的解决方案")
        return

    turbines = df.loc[sol].copy().reset_index(drop=True)
    turbines["turbine_id"] = [f"T{i + 1}" for i in range(len(turbines))]

    # 保留奉节县内风机
    base_map = create_fengjie_base_map()
    if base_map:
        turbines_fengjie = turbines[
            turbines.apply(lambda row: Point(row["lon"], row["lat"]).within(base_map['geometry']), axis=1)
        ]
    else:
        turbines_fengjie = turbines

    # 计算发电量 - 使用修正后的函数
    if not turbines_fengjie.empty:
        power_results = calculate_power_generation_corrected(turbines_fengjie)
    else:
        power_results = None

    # 收敛曲线
    st.markdown("#### 📈 算法收敛过程")
    fitness_history = result.get("fitness_history") or result.get("convergence_history") or []
    if fitness_history:
        fitness_smooth = pd.Series(fitness_history).rolling(5, min_periods=1).mean()
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            y=fitness_history,
            mode="lines",
            name="原始适应度",
            line=dict(color='lightblue', width=1)
        ))
        fig_conv.add_trace(go.Scatter(
            y=fitness_smooth,
            mode="lines",
            name="平滑趋势",
            line=dict(color="crimson", width=3)
        ))
        fig_conv.update_layout(
            height=400,
            template="plotly_white",
            title="算法收敛曲线",
            xaxis_title="迭代次数",
            yaxis_title="适应度值"
        )
        # 修复：添加唯一的key参数
        st.plotly_chart(fig_conv, use_container_width=True, key="convergence_chart")

    # 输出结果 - 增强版（包含发电量信息）
    st.markdown("#### 📊 优化结果与发电量分析")

    if power_results and not turbines_fengjie.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总装机容量", f"{power_results['total_capacity_mw']:.1f} MW")
        with col2:
            st.metric("年发电量", f"{power_results['total_annual_generation_gwh']:.1f} GWh")
        with col3:
            st.metric("平均容量因数", f"{power_results['average_capacity_factor']:.1%}")
        with col4:
            st.metric("等效满发小时", f"{power_results['equivalent_full_load_hours']:.0f} h")

        # 经济效益分析 - 使用更合理的参数
        st.markdown("#### 💰 经济效益估算")

        # 更合理的假设参数
        electricity_price = 0.4  # 元/kWh（风电上网电价）
        investment_per_kw = 6000  # 元/kW（单位千瓦投资）
        om_cost_per_kw = 150  # 元/kW/年（运维成本）

        total_investment = power_results['total_capacity_kw'] * investment_per_kw / 1e8  # 亿元
        annual_revenue = power_results['total_annual_generation_kwh'] * electricity_price / 1e8  # 亿元
        annual_om_cost = power_results['total_capacity_kw'] * om_cost_per_kw / 1e8  # 亿元
        annual_profit = annual_revenue - annual_om_cost

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总投资", f"{total_investment:.2f} 亿元")
        with col2:
            st.metric("年发电收入", f"{annual_revenue:.2f} 亿元")
        with col3:
            st.metric("年运维成本", f"{annual_om_cost:.2f} 亿元")
        with col4:
            profit_color = "normal" if annual_profit >= 0 else "inverse"
            st.metric("年净利润", f"{annual_profit:.2f} 亿元", delta_color=profit_color)

        # 投资回收期
        if annual_profit > 0:
            payback_period = total_investment / annual_profit
            st.metric("投资回收期", f"{payback_period:.1f} 年")
        else:
            st.metric("投资回收期", "无法回收", delta="亏损运营", delta_color="inverse")

        # 显示发电量分布
        st.markdown("#### 📈 发电量分布分析")
        col1, col2 = st.columns(2)
        with col1:
            # 容量因数分布
            if power_results['capacity_factors']:
                fig_cf = go.Figure()
                fig_cf.add_trace(go.Histogram(
                    x=power_results['capacity_factors'],
                    nbinsx=20,
                    name="容量因数分布"
                ))
                fig_cf.update_layout(
                    title="风机容量因数分布",
                    xaxis_title="容量因数",
                    yaxis_title="风机数量",
                    template="plotly_white"
                )
                # 修复：添加唯一的key参数
                st.plotly_chart(fig_cf, use_container_width=True, key="capacity_factor_histogram")

        with col2:
            # 风速与发电量关系
            if power_results['annual_generation_per_turbine']:
                fig_wind = go.Figure()
                fig_wind.add_trace(go.Scatter(
                    x=turbines_fengjie["predicted_wind_speed"],
                    y=[gen / 1e6 for gen in power_results['annual_generation_per_turbine']],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=power_results['capacity_factors'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="容量因数")
                    ),
                    text=[f"T{i + 1}" for i in range(len(turbines_fengjie))],
                    name="风机"
                ))
                fig_wind.update_layout(
                    title="风速与年发电量关系",
                    xaxis_title="风速 (m/s)",
                    yaxis_title="年发电量 (GWh)",
                    template="plotly_white"
                )
                # 修复：添加唯一的key参数
                st.plotly_chart(fig_wind, use_container_width=True, key="wind_generation_scatter")

    else:
        # 原始指标（如果没有发电量数据）
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最优适应度值", f"{result['fitness']:.0f}")
        with col2:
            st.metric("风机数量", len(turbines_fengjie))
        with col3:
            if len(turbines_fengjie) > 0:
                avg_wind_speed = turbines_fengjie["predicted_wind_speed"].mean()
                st.metric("平均风速", f"{avg_wind_speed:.1f} m/s")
        with col4:
            if len(turbines_fengjie) > 0:
                # 创建功率密度数据表格
                power_data = {
                    "指标": ["总功率密度", "平均功率密度", "最大功率密度", "最小功率密度"],
                    "数值(W/m²)": [
                        f"{turbines_fengjie['wind_power_density'].sum():.0f}",
                        f"{turbines_fengjie['wind_power_density'].mean():.0f}",
                        f"{turbines_fengjie['wind_power_density'].max():.0f}",
                        f"{turbines_fengjie['wind_power_density'].min():.0f}"
                    ]
                }
                power_df = pd.DataFrame(power_data)
                st.dataframe(power_df, hide_index=True, use_container_width=True, key="power_density_table")

    # 显示详细数据
    st.markdown("#### 📋 风机详细信息")
    if not turbines_fengjie.empty:
        display_df = turbines_fengjie[
            ["turbine_id", "lat", "lon", "predicted_wind_speed", "wind_power_density", "cost"]].copy()
        display_df["lat"] = display_df["lat"].round(4)
        display_df["lon"] = display_df["lon"].round(4)
        display_df["predicted_wind_speed"] = display_df["predicted_wind_speed"].round(2)
        display_df["wind_power_density"] = display_df["wind_power_density"].round(0)
        display_df["cost"] = display_df["cost"].round(0)

        # 添加发电量信息
        if power_results and len(power_results['annual_generation_per_turbine']) == len(turbines_fengjie):
            display_df["年发电量(GWh)"] = [f"{x / 1e6:.2f}" for x in power_results['annual_generation_per_turbine']]
            display_df["容量因数"] = [f"{x:.1%}" for x in power_results['capacity_factors']]

        st.dataframe(display_df, use_container_width=True, key="turbine_details_table")

        # 风机配置说明
        if power_results:
            st.markdown("#### 🌀 风机配置说明")
            config = power_results['turbine_config']
            st.write(f"""
            - **风机型号**: {config['model']}
            - **单机容量**: {config['rated_power'] / 1000} MW
            - **风轮直径**: {config['rotor_diameter']} 米
            - **轮毂高度**: {config['hub_height']} 米
            - **工作风速**: {config['cut_in_speed']}-{config['rated_speed']}-{config['cut_out_speed']} m/s
            - **综合效率**: {config['efficiency']:.0%}（考虑尾流、可用率等损失）
            - **计算方法**: 基于威布尔分布和典型功率曲线
            """)
    else:
        st.info("没有在奉节县范围内找到有效的风机位置")


# ======================================================
# 🔍 数据质量检查函数
# ======================================================
def check_data_quality_for_power_calculation(turbines_df):
    """检查发电量计算所需的数据质量"""
    if turbines_df.empty:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        wind_speeds = turbines_df["predicted_wind_speed"]
        avg_wind_speed = wind_speeds.mean()
        st.metric("平均风速", f"{avg_wind_speed:.1f} m/s")
        if avg_wind_speed < 5.0:
            st.error("❌ 风速偏低")
        elif avg_wind_speed > 12.0:
            st.warning("⚠️ 风速偏高")

    with col2:
        wind_std = wind_speeds.std()
        st.metric("风速标准差", f"{wind_std:.1f} m/s")
        if wind_std < 0.5:
            st.warning("⚠️ 风速变化较小")

    with col3:
        valid_ratio = (wind_speeds >= 3.0).mean() * 100
        st.metric("有效风速比例", f"{valid_ratio:.1f}%")
        if valid_ratio < 80:
            st.warning("⚠️ 部分点位风速过低")


# ======================================================
# ⚡ 修正的发电量计算模块
# ======================================================
def calculate_power_generation_corrected(turbines_df):
    """
    修正的风电场发电量计算
    基于更合理的风速分布和功率曲线模型
    """
    # 更合理的风机配置参数（以金风科技GW-140/2500为例）
    TURBINE_CONFIG = {
        'model': '金风科技 GW-140/2500',
        'rated_power': 2500,  # 额定功率 (kW)
        'rotor_diameter': 140,  # 风轮直径 (m)
        'hub_height': 90,  # 轮毂高度 (m)
        'cut_in_speed': 3.0,  # 切入风速 (m/s)
        'rated_speed': 11.0,  # 额定风速 (m/s)
        'cut_out_speed': 25.0,  # 切出风速 (m/s)
        'efficiency': 0.92,  # 风机效率（不含尾流）
        'availability': 0.98,  # 可用率
        'array_efficiency': 0.92,  # 阵列效率（尾流损失）
    }

    def detailed_power_curve(wind_speed):
        """
        更详细的功率曲线函数
        基于实际风机功率特性
        """
        if wind_speed < TURBINE_CONFIG['cut_in_speed']:
            return 0
        elif wind_speed < TURBINE_CONFIG['rated_speed']:
            # 使用更精确的功率曲线近似
            normalized_speed = (wind_speed - TURBINE_CONFIG['cut_in_speed']) / \
                               (TURBINE_CONFIG['rated_speed'] - TURBINE_CONFIG['cut_in_speed'])
            return TURBINE_CONFIG['rated_power'] * (normalized_speed ** 3)
        elif wind_speed <= TURBINE_CONFIG['cut_out_speed']:
            return TURBINE_CONFIG['rated_power']
        else:
            return 0

    def weibull_wind_distribution(avg_wind_speed, k=2.0, points=12):
        """
        生成威布尔分布的风速频率分布
        k: 形状参数，通常1.5-2.5
        """
        # 修正：使用标准math库的gamma函数
        from scipy.special import gamma  # 使用scipy的gamma函数更准确

        # 尺度参数计算
        c = avg_wind_speed / gamma(1 + 1 / k)

        # 生成风速区间
        wind_bins = np.linspace(0.5, 25.5, points + 1)
        wind_speeds = (wind_bins[:-1] + wind_bins[1:]) / 2
        wind_speeds = np.clip(wind_speeds, 0, 25)

        # 计算频率
        frequencies = (weibull_cdf(wind_bins[1:], c, k) -
                       weibull_cdf(wind_bins[:-1], c, k))
        frequencies = frequencies / frequencies.sum()  # 归一化

        return wind_speeds, frequencies

    def weibull_cdf(x, c, k):
        """威布尔分布累积分布函数"""
        return 1 - np.exp(-(x / c) ** k)

    # 计算每台风机年发电量
    annual_generation_per_turbine = []
    capacity_factors = []

    for _, turbine in turbines_df.iterrows():
        avg_wind_speed = turbine['predicted_wind_speed']

        try:
            # 尝试使用威布尔分布
            from scipy.special import gamma
            wind_speeds, frequencies = weibull_wind_distribution(avg_wind_speed)
        except ImportError:
            # 如果scipy不可用，使用简化方法
            st.warning("⚠️ scipy未安装，使用简化发电量计算")
            # 简化方法：直接使用平均风速
            theoretical_power = detailed_power_curve(avg_wind_speed)
            actual_power = (theoretical_power *
                            TURBINE_CONFIG['efficiency'] *
                            TURBINE_CONFIG['availability'] *
                            TURBINE_CONFIG['array_efficiency'])
            annual_energy = actual_power * 8760
        else:
            # 基于风速分布计算年发电量
            annual_energy = 0
            for speed, freq in zip(wind_speeds, frequencies):
                power_output = detailed_power_curve(speed)
                # 考虑各种效率损失
                actual_power = (power_output *
                                TURBINE_CONFIG['efficiency'] *
                                TURBINE_CONFIG['availability'] *
                                TURBINE_CONFIG['array_efficiency'])
                annual_energy += actual_power * 8760 * freq

        annual_generation_per_turbine.append(annual_energy)

        # 容量因数
        capacity_factor = annual_energy / (TURBINE_CONFIG['rated_power'] * 8760)
        capacity_factors.append(capacity_factor)

    # 汇总统计
    total_annual_generation = sum(annual_generation_per_turbine)
    avg_capacity_factor = np.mean(capacity_factors)
    total_capacity = len(turbines_df) * TURBINE_CONFIG['rated_power']
    equivalent_full_load_hours = total_annual_generation / total_capacity

    return {
        'total_annual_generation_kwh': total_annual_generation,
        'total_annual_generation_mwh': total_annual_generation / 1000,
        'total_annual_generation_gwh': total_annual_generation / 1e6,
        'total_capacity_kw': total_capacity,
        'total_capacity_mw': total_capacity / 1000,
        'average_capacity_factor': avg_capacity_factor,
        'equivalent_full_load_hours': equivalent_full_load_hours,
        'turbine_count': len(turbines_df),
        'annual_generation_per_turbine': annual_generation_per_turbine,
        'capacity_factors': capacity_factors,
        'turbine_config': TURBINE_CONFIG
    }


# ======================================================
# ⚡ 简化版发电量计算（备用）
# ======================================================
def calculate_power_generation_simple(turbines_df):
    """
    简化版发电量计算，不依赖scipy
    """
    TURBINE_CONFIG = {
        'model': '金风科技 GW-140/2500',
        'rated_power': 2500,  # kW
        'cut_in_speed': 3.0,
        'rated_speed': 11.0,
        'cut_out_speed': 25.0,
        'overall_efficiency': 0.35,  # 综合效率
    }

    def power_curve(wind_speed):
        if wind_speed < TURBINE_CONFIG['cut_in_speed']:
            return 0
        elif wind_speed < TURBINE_CONFIG['rated_speed']:
            return TURBINE_CONFIG['rated_power'] * (
                    (wind_speed - TURBINE_CONFIG['cut_in_speed']) /
                    (TURBINE_CONFIG['rated_speed'] - TURBINE_CONFIG['cut_in_speed'])
            ) ** 3
        elif wind_speed <= TURBINE_CONFIG['cut_out_speed']:
            return TURBINE_CONFIG['rated_power']
        else:
            return 0

    annual_generation_per_turbine = []
    capacity_factors = []

    for _, turbine in turbines_df.iterrows():
        wind_speed = turbine['predicted_wind_speed']
        power_output = power_curve(wind_speed)
        annual_energy = power_output * 8760 * TURBINE_CONFIG['overall_efficiency']

        annual_generation_per_turbine.append(annual_energy)
        capacity_factor = annual_energy / (TURBINE_CONFIG['rated_power'] * 8760)
        capacity_factors.append(capacity_factor)

    total_annual_generation = sum(annual_generation_per_turbine)
    total_capacity = len(turbines_df) * TURBINE_CONFIG['rated_power']

    return {
        'total_annual_generation_kwh': total_annual_generation,
        'total_annual_generation_gwh': total_annual_generation / 1e6,
        'total_capacity_mw': total_capacity / 1000,
        'average_capacity_factor': np.mean(capacity_factors),
        'equivalent_full_load_hours': total_annual_generation / total_capacity,
        'annual_generation_per_turbine': annual_generation_per_turbine,
        'capacity_factors': capacity_factors,
        'turbine_config': TURBINE_CONFIG
    }


# ======================================================
# ⚡ 保留原始函数（兼容性）
# ======================================================
def calculate_power_generation(turbines_df):
    """原始函数，用于兼容性"""
    try:
        return calculate_power_generation_corrected(turbines_df)
    except Exception as e:
        st.warning(f"使用简化发电量计算: {e}")
        return calculate_power_generation_simple(turbines_df)