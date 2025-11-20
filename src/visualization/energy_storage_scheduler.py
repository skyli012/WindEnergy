import geopandas as gpd
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Point
import plotly.graph_objects as go
import altair as alt

from src.optimization.algorithm_convergence_curve import call_optimize_function
from src.utils.check_data import check_data_quality
from src.utils.create_map import display_fengjie_standalone_map, display_environment, display_optimization_map, \
    create_fengjie_base_map
from src.visualization.opt_result_show import display_optimization_result

# ======================================================
# 🔋 储能调度策略核心算法
# ======================================================

class EnergyStorageScheduler:
    """储能充放电调度策略"""

    def __init__(self, capacity_kwh, max_power_kw, efficiency=0.92):
        """
        初始化储能系统
        capacity_kwh: 储能容量 (kWh)
        max_power_kw: 最大充放电功率 (kW)
        efficiency: 充放电效率
        """
        self.capacity = capacity_kwh
        self.max_power = max_power_kw
        self.efficiency = efficiency

    def smoothing_strategy(self, wind_power, time_window=6):
        """
        出力平滑策略 - 滑动窗口方法
        wind_power: 风电功率序列 (kW)
        time_window: 滑动窗口大小 (小时)
        """
        n = len(wind_power)
        smoothed_power = np.zeros(n)
        battery_soc = np.zeros(n)  # 电池SOC (0-1)
        charge_discharge = np.zeros(n)  # 充放电功率 (+放电, -充电)

        # 初始SOC设为50%
        soc = 0.5

        for i in range(n):
            # 滑动窗口平滑
            start_idx = max(0, i - time_window // 2)
            end_idx = min(n, i + time_window // 2 + 1)
            target_power = np.mean(wind_power[start_idx:end_idx])

            # 计算需要的调节功率
            power_diff = target_power - wind_power[i]

            # 考虑储能系统限制
            if power_diff > 0:  # 需要放电
                max_discharge = min(
                    self.max_power,
                    soc * self.capacity,  # 当前可用能量
                    power_diff
                )
                actual_discharge = max_discharge
                soc -= actual_discharge / self.capacity
                charge_discharge[i] = actual_discharge

            elif power_diff < 0:  # 需要充电
                max_charge = min(
                    self.max_power,
                    (1 - soc) * self.capacity / self.efficiency,  # 剩余充电空间
                    -power_diff
                )
                actual_charge = max_charge
                soc += actual_charge * self.efficiency / self.capacity
                charge_discharge[i] = -actual_charge
            else:
                charge_discharge[i] = 0

            # 确保SOC在合理范围内
            soc = max(0.1, min(0.9, soc))
            battery_soc[i] = soc
            smoothed_power[i] = wind_power[i] + charge_discharge[i]

        return smoothed_power, battery_soc, charge_discharge

    def curtailment_reduction_strategy(self, wind_power, grid_capacity):
        """
        弃风消减策略
        grid_capacity: 电网接收容量 (kW)
        """
        n = len(wind_power)
        delivered_power = np.zeros(n)
        curtailed_power = np.zeros(n)
        battery_soc = np.zeros(n)
        charge_discharge = np.zeros(n)

        soc = 0.5

        for i in range(n):
            current_wind = wind_power[i]

            if current_wind > grid_capacity:  # 弃风情况
                # 计算弃风量
                curtailment = current_wind - grid_capacity

                # 尝试充电消纳弃风
                available_charge = min(
                    self.max_power,
                    (1 - soc) * self.capacity / self.efficiency,
                    curtailment
                )

                if available_charge > 0:
                    # 充电消纳部分弃风
                    charge_power = available_charge
                    soc += charge_power * self.efficiency / self.capacity
                    charge_discharge[i] = -charge_power
                    curtailed_power[i] = curtailment - charge_power
                    delivered_power[i] = grid_capacity
                else:
                    # 无法充电，全部弃风
                    curtailed_power[i] = curtailment
                    delivered_power[i] = grid_capacity

            elif current_wind < grid_capacity:  # 发电不足
                # 检查是否需要放电补充
                power_deficit = grid_capacity - current_wind

                if power_deficit > 0 and soc > 0.1:
                    # 计算可放电量
                    available_discharge = min(
                        self.max_power,
                        (soc - 0.1) * self.capacity,  # 保留10%电量
                        power_deficit
                    )

                    if available_discharge > 0:
                        soc -= available_discharge / self.capacity
                        charge_discharge[i] = available_discharge
                        delivered_power[i] = current_wind + available_discharge
                    else:
                        delivered_power[i] = current_wind
                else:
                    delivered_power[i] = current_wind

                curtailed_power[i] = 0
            else:
                delivered_power[i] = current_wind
                curtailed_power[i] = 0

            battery_soc[i] = soc

        return delivered_power, curtailed_power, battery_soc, charge_discharge


def calculate_wind_power_from_speed(wind_speed, turbine_capacity=2000):
    """根据风速计算风机发电功率"""
    # 简化的风机功率曲线
    cut_in = 3.0  # 切入风速 m/s
    rated = 12.0  # 额定风速 m/s
    cut_out = 25.0  # 切出风速 m/s

    power = np.zeros_like(wind_speed)

    for i, speed in enumerate(wind_speed):
        if speed < cut_in or speed > cut_out:
            power[i] = 0
        elif speed >= cut_in and speed < rated:
            # 线性增长区间
            power[i] = turbine_capacity * ((speed - cut_in) / (rated - cut_in)) ** 3
        else:  # rated to cut_out
            power[i] = turbine_capacity

    return power


def calculate_metrics(original_power, delivered_power, curtailed_power):
    """计算关键性能指标"""
    total_generation = np.sum(original_power)
    total_delivered = np.sum(delivered_power)
    total_curtailed = np.sum(curtailed_power)

    curtailment_rate = total_curtailed / total_generation * 100 if total_generation > 0 else 0
    utilization_improvement = ((total_delivered - total_generation + total_curtailed) /
                               total_generation * 100) if total_generation > 0 else 0

    # 计算波动性
    original_fluctuation = np.std(np.diff(original_power))
    delivered_fluctuation = np.std(np.diff(delivered_power))
    fluctuation_reduction = (original_fluctuation - delivered_fluctuation) / original_fluctuation * 100

    return {
        'total_generation_mwh': total_generation / 1000,  # 转换为MWh
        'total_delivered_mwh': total_delivered / 1000,
        'total_curtailed_mwh': total_curtailed / 1000,
        'curtailment_rate_percent': curtailment_rate,
        'utilization_improvement_percent': utilization_improvement,
        'fluctuation_reduction_percent': fluctuation_reduction,
        'original_fluctuation': original_fluctuation,
        'delivered_fluctuation': delivered_fluctuation
    }


def create_single_turbine_assessment(wind_power, delivered_power, curtailed_power, battery_soc, hours):
    """创建单个风机评估图表"""

    # 使用Streamlit columns布局
    # st.markdown("### 🌬️ 单个风机评估")

    # 关键指标卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_gen = np.sum(wind_power) / 1000  # MWh
        st.metric("总发电量", f"{total_gen:.1f} MWh")

    with col2:
        total_delivered = np.sum(delivered_power) / 1000  # MWh
        st.metric("可消纳电量", f"{total_delivered:.1f} MWh")

    with col3:
        curtailment_rate = (np.sum(curtailed_power) / np.sum(wind_power)) * 100 if np.sum(wind_power) > 0 else 0
        st.metric("弃风率", f"{curtailment_rate:.1f}%",
                  delta=f"-{curtailment_rate:.1f}%" if curtailment_rate > 0 else None)

    with col4:
        avg_soc = np.mean(battery_soc) * 100
        st.metric("平均SOC", f"{avg_soc:.1f}%")

    # 功率曲线图
    st.markdown("#### 📈 功率曲线分析")
    power_data = pd.DataFrame({
        '小时': hours,
        '原始功率': wind_power,
        '平滑后功率': delivered_power,
        '弃风功率': curtailed_power
    })

    power_chart = alt.Chart(power_data.melt('小时', var_name='类型', value_name='功率')).mark_line().encode(
        x='小时:Q',
        y='功率:Q',
        color='类型:N',
        strokeDash=alt.condition(
            alt.datum.类型 == '原始功率',
            alt.value([5, 5]),  # 虚线
            alt.value([0, 0])  # 实线
        )
    ).properties(height=300)

    st.altair_chart(power_chart, use_container_width=True)

    # SOC曲线图
    st.markdown("#### 🔋 电池SOC曲线")
    soc_data = pd.DataFrame({
        '小时': hours,
        'SOC': battery_soc * 100  # 转换为百分比
    })

    soc_chart = alt.Chart(soc_data).mark_area(
        line={'color': 'orange'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='white', offset=0),
                   alt.GradientStop(color='orange', offset=1)],
            x1=0, x2=0, y1=1, y2=0
        )
    ).encode(
        x='小时:Q',
        y=alt.Y('SOC:Q', title='SOC (%)', scale=alt.Scale(domain=[0, 100]))
    ).properties(height=250)

    st.altair_chart(soc_chart, use_container_width=True)

    # 功率分布直方图
    st.markdown("#### 📊 功率分布统计")
    col_left, col_right = st.columns(2)

    with col_left:
        # 原始功率分布
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(wind_power, bins=20, alpha=0.7, color='blue', label='原始功率')
        ax.set_xlabel('功率 (kW)')
        ax.set_ylabel('频率')
        ax.set_title('原始功率分布')
        ax.legend()
        st.pyplot(fig)

    with col_right:
        # 平滑后功率分布
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(delivered_power, bins=20, alpha=0.7, color='green', label='平滑后功率')
        ax.set_xlabel('功率 (kW)')
        ax.set_ylabel('频率')
        ax.set_title('平滑后功率分布')
        ax.legend()
        st.pyplot(fig)


def create_wind_farm_assessment(metrics, storage_capacity, max_power, n_turbines):
    """创建整体风场评估"""

    st.markdown("### 🏭 整体风场评估")

    # 整体性能指标
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        utilization_improvement = metrics['utilization_improvement_percent']
        st.metric("电能利用率提升", f"{utilization_improvement:.1f}%",
                  delta=f"+{utilization_improvement:.1f}%")

    with col2:
        fluctuation_reduction = metrics['fluctuation_reduction_percent']
        st.metric("功率波动降低", f"{fluctuation_reduction:.1f}%",
                  delta=f"+{fluctuation_reduction:.1f}%")

    with col3:
        st.metric("储能容量需求", f"{storage_capacity / 1000:.1f} MWh")

    with col4:
        st.metric("储能功率需求", f"{max_power / 1000:.1f} MW")

    # 电能分配饼图
    st.markdown("#### 🥧 电能分配分析")

    energy_data = pd.DataFrame({
        '类型': ['可消纳电量', '弃风电量', '未利用电量'],
        '数值': [
            metrics['total_delivered_mwh'],
            metrics['total_curtailed_mwh'],
            max(0, metrics['total_generation_mwh'] - metrics['total_delivered_mwh'] - metrics['total_curtailed_mwh'])
        ]
    })

    pie_chart = alt.Chart(energy_data).mark_arc().encode(
        theta='数值:Q',
        color=alt.Color('类型:N', scale=alt.Scale(
            domain=['可消纳电量', '弃风电量', '未利用电量'],
            range=['#28a745', '#dc3545', '#ffc107']
        )),
        tooltip=['类型', '数值']
    ).properties(height=300)

    st.altair_chart(pie_chart, use_container_width=True)

    # 电网接入条件改善分析
    st.markdown("#### ⚡ 电网接入条件改善")

    grid_data = pd.DataFrame({
        '指标': ['功率波动性', '可调度性', '电能质量', '备用容量'],
        '改善程度': [
            metrics['fluctuation_reduction_percent'],
            metrics['utilization_improvement_percent'],
            metrics['fluctuation_reduction_percent'] * 0.8,
            metrics['utilization_improvement_percent'] * 0.6
        ]
    })

    bar_chart = alt.Chart(grid_data).mark_bar().encode(
        x='指标:N',
        y='改善程度:Q',
        color=alt.Color('改善程度:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['指标', '改善程度']
    ).properties(height=300)

    st.altair_chart(bar_chart, use_container_width=True)

    # 储能配置建议
    st.markdown("#### 💡 储能配置建议")

    # 计算理论最优配置
    theoretical_capacity = metrics['total_curtailed_mwh'] * 1000 * 0.8  # 考虑80%的弃风可消纳
    theoretical_power = theoretical_capacity / 4  # 4小时放电率

    suggestion_col1, suggestion_col2 = st.columns(2)

    with suggestion_col1:
        st.info(f"""
        **当前配置分析:**
        - 容量: {storage_capacity / 1000:.1f} MWh
        - 功率: {max_power / 1000:.1f} MW
        - 容量利用率: {metrics['total_curtailed_mwh'] / (storage_capacity / 1000) * 100:.1f}%
        """)

    with suggestion_col2:
        st.success(f"""
        **理论最优配置:**
        - 建议容量: {theoretical_capacity / 1000:.1f} MWh
        - 建议功率: {theoretical_power / 1000:.1f} MW
        - 预计弃风率: < 5%
        """)