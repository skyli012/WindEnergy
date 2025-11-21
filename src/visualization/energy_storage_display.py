import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def display_energy_storage_performance(optimization_result, windfarm_data):
    """
    展示储能调度效果与发电表现 - 只显示最重要的功率时序图
    """
    st.markdown("---")
    st.markdown("#### 🔋 储能调度效果分析")

    # 检查是否有储能调度数据
    if not has_storage_data(optimization_result):
        st.warning("⚠️ 优化结果中未找到储能调度数据")
        return

    # 直接显示功率时序分析图
    display_power_time_series(optimization_result)


def has_storage_data(optimization_result):
    """检查优化结果中是否包含储能调度数据"""
    storage_keys = [
        'storage_schedule', 'battery_power', 'grid_power', 'wind_power',
        'storage_soc', 'energy_storage_data', 'time_series_data'
    ]

    for key in storage_keys:
        if key in optimization_result and optimization_result[key] is not None:
            return True
    return False


def display_power_time_series(optimization_result):
    """
    展示风机原始出力 vs 并网出力 vs 储能充放电（时间序列图）
    """
    st.markdown("##### 📊 功率时序分析：风机原始出力 vs 并网出力 vs 储能充放电")

    # 获取时间序列数据
    time_data = get_time_series_data(optimization_result)

    if time_data is None:
        st.error("❌ 无法获取时间序列数据")
        return

    # 创建单图，显示三条曲线
    fig = go.Figure()

    # 添加三条功率曲线
    if 'wind_power' in time_data:
        fig.add_trace(
            go.Scatter(
                x=time_data.index,
                y=time_data['wind_power'],
                name='风机原始出力 P_wind(t)',
                line=dict(color='blue', width=3),
                opacity=0.9
            )
        )

    if 'grid_power' in time_data:
        fig.add_trace(
            go.Scatter(
                x=time_data.index,
                y=time_data['grid_power'],
                name='并网出力 P_grid(t)',
                line=dict(color='green', width=3),
                opacity=0.9
            )
        )

    # 添加储能功率 - 使用填充效果区分充放电
    if 'battery_power' in time_data:
        # 充电部分（负值）
        charge_mask = time_data['battery_power'] < 0
        if charge_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=time_data.index[charge_mask],
                    y=time_data['battery_power'][charge_mask],
                    name='储能充电 P_batt(t)',
                    line=dict(color='red', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.3)',
                    mode='lines'
                )
            )

        # 放电部分（正值）
        discharge_mask = time_data['battery_power'] > 0
        if discharge_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=time_data.index[discharge_mask],
                    y=time_data['battery_power'][discharge_mask],
                    name='储能放电 P_batt(t)',
                    line=dict(color='orange', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255,165,0,0.3)',
                    mode='lines'
                )
            )

    # 添加功率限制线（20 MW）- 修改为山地项目电网约束
    grid_limit = 20  # MW
    fig.add_hline(
        y=grid_limit,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"电网限制 {grid_limit} MW",
        annotation_position="top left"
    )

    # 添加零线
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="black",
        line_width=1
    )

    # 更新布局
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="储能调度功率时序分析 - 山地风电项目",
        xaxis_title="时间",
        yaxis_title="功率 (MW)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # 显示关键统计信息
    display_power_statistics(time_data, grid_limit)


def display_power_statistics(time_data, grid_limit):
    """显示功率统计信息"""
    if 'wind_power' not in time_data or 'grid_power' not in time_data:
        return

    st.markdown("##### 📈 关键功率统计")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        max_wind = time_data['wind_power'].max()
        st.metric("最大原始出力", f"{max_wind:.1f} MW")

    with col2:
        max_grid = time_data['grid_power'].max()
        over_limit = max_grid > grid_limit
        st.metric(
            "最大并网出力",
            f"{max_grid:.1f} MW",
            delta="超限" if over_limit else "正常",
            delta_color="inverse" if over_limit else "normal"
        )

    with col3:
        if 'battery_power' in time_data:
            charge_data = time_data[time_data['battery_power'] < 0]['battery_power']
            discharge_data = time_data[time_data['battery_power'] > 0]['battery_power']
            max_charge = abs(charge_data.min()) if len(charge_data) > 0 else 0
            max_discharge = discharge_data.max() if len(discharge_data) > 0 else 0
            st.metric("最大充/放电", f"{max_charge:.1f}/{max_discharge:.1f} MW")

    with col4:
        if 'battery_power' in time_data:
            charge_data = time_data[time_data['battery_power'] < 0]['battery_power']
            discharge_data = time_data[time_data['battery_power'] > 0]['battery_power']
            # 计算总能量（考虑时间间隔为15分钟=0.25小时）
            total_charge_energy = abs(charge_data.sum()) * 0.25 if len(charge_data) > 0 else 0
            total_discharge_energy = discharge_data.sum() * 0.25 if len(discharge_data) > 0 else 0
            # 修正效率计算：放电能量/充电能量
            efficiency = (total_discharge_energy / total_charge_energy * 100) if total_charge_energy > 0 else 0
            # 效率不能超过100%
            efficiency = min(efficiency, 100)
            st.metric("充放电效率", f"{efficiency:.1f}%")

    # 计算弃风率 - 修正计算逻辑
    wind_energy = time_data['wind_power'].sum() * 0.25  # 转换为能量(MWh)
    grid_energy = time_data['grid_power'].sum() * 0.25  # 转换为能量(MWh)

    # 弃风能量 = 风电能量 - 并网能量
    curtailment_energy = wind_energy - grid_energy
    curtailment_rate = (curtailment_energy / wind_energy * 100) if wind_energy > 0 else 0
    # 弃风率不能为负数
    curtailment_rate = max(curtailment_rate, 0)

    st.markdown("##### 🎯 调度效果评估")
    col5, col6, col7 = st.columns(3)

    with col5:
        st.metric("弃风率", f"{curtailment_rate:.1f}%")

    with col6:
        # 计算削峰效果
        peak_shaving = max_wind - max_grid
        st.metric("削峰效果", f"{peak_shaving:.1f} MW")

    with col7:
        # 计算填谷效果
        min_wind = time_data['wind_power'].min()
        min_grid = time_data['grid_power'].min()
        valley_filling = max(0, min_grid - min_wind)
        st.metric("填谷效果", f"{valley_filling:.1f} MW")


def get_time_series_data(optimization_result):
    """
    从优化结果中提取时间序列数据
    如果没有真实数据，生成模拟数据
    """
    # 首先尝试从优化结果中获取真实数据
    possible_keys = [
        'time_series_data', 'storage_schedule', 'power_data',
        'wind_power', 'grid_power', 'battery_power', 'storage_soc'
    ]

    for key in possible_keys:
        if key in optimization_result and optimization_result[key] is not None:
            data = optimization_result[key]

            # 如果是DataFrame格式，直接返回
            if isinstance(data, pd.DataFrame):
                return data

            # 如果是字典格式，转换为DataFrame
            elif isinstance(data, dict):
                return pd.DataFrame(data)

    # 如果没有找到真实数据，生成模拟数据
    return create_realistic_sample_data()


def create_realistic_sample_data():
    """
    创建山地风电项目的模拟数据 - 电网约束20MW
    """
    np.random.seed(42)  # 保证结果可重现

    periods = 96  # 24小时 * 4（15分钟间隔）
    index = pd.date_range('2024-01-01 00:00', periods=periods, freq='15T')

    # 创建山地风电出力模式 - 考虑山地风电特点
    t = np.linspace(0, 4 * np.pi, periods)

    # 基础模式：山地风电波动较大
    daily_pattern = 0.6 + 0.3 * np.sin(t - np.pi / 2)  # 基础功率适中

    # 山地风电特点：阵风明显，波动大
    gust_wind_1 = 0.4 * np.exp(-((t - 1.5 * np.pi) ** 2) / 0.4)  # 上午阵风
    gust_wind_2 = 0.5 * np.exp(-((t - 2.5 * np.pi) ** 2) / 0.3)  # 下午阵风
    gust_wind_3 = 0.3 * np.exp(-((t - 3.5 * np.pi) ** 2) / 0.5)  # 夜间阵风

    # 山地风电随机波动较大
    random_waves = (0.3 * np.sin(5 * t) +
                    0.25 * np.sin(10 * t) +
                    0.2 * np.sin(18 * t))

    # 噪声 - 山地风电噪声较大
    noise = 0.2 * np.random.normal(size=periods)

    # 组合生成风电出力 - 山地风电规模较小
    wind_power = 25 * daily_pattern + 15 * (gust_wind_1 + gust_wind_2 + gust_wind_3) + 12 * random_waves + 8 * noise
    wind_power = np.clip(wind_power, 5, 45)  # 限制在5-45MW之间，符合山地项目规模

    # 模拟储能调度策略 - 山地项目储能配置较小
    grid_limit = 20  # MW - 山地电网约束较小
    battery_capacity = 30  # MWh - 山地项目储能容量较小
    max_charge_power = 8   # MW - 充放电功率较小
    max_discharge_power = 8  # MW

    # 初始化数组
    grid_power = np.zeros(periods)
    battery_power = np.zeros(periods)
    soc = np.zeros(periods)
    soc[0] = 50  # 初始SOC

    # 记录弃风情况
    curtailment_periods = 0

    for i in range(periods):
        # 计算风电功率与电网限制的差值
        power_diff = wind_power[i] - grid_limit

        if power_diff > 0:  # 风电功率超过限制，需要削峰
            # 储能充电能力有限
            available_charge_capacity = min(
                (100 - soc[i]) * battery_capacity / 100 * 4,  # SOC限制
                max_charge_power  # 功率限制
            )

            charge_power = min(power_diff, available_charge_capacity)
            battery_power[i] = -charge_power

            # 剩余的超限功率需要弃风
            remaining_excess = power_diff - charge_power
            if remaining_excess > 0:
                # 弃风策略：限制并网功率到20MW
                grid_power[i] = grid_limit
                curtailment_periods += 1
            else:
                grid_power[i] = wind_power[i] - charge_power

        elif wind_power[i] < 8:  # 风电功率较低，需要填谷（山地项目填谷门槛较低）
            # 计算需要提升的功率
            needed_power = 8 - wind_power[i]

            # 检查储能是否有足够能量放电
            available_discharge_capacity = min(
                (soc[i] - 20) * battery_capacity / 100 * 4,  # SOC限制
                max_discharge_power,
                needed_power
            )

            if available_discharge_capacity > 0:
                battery_power[i] = available_discharge_capacity
                grid_power[i] = wind_power[i] + available_discharge_capacity
            else:
                battery_power[i] = 0
                grid_power[i] = wind_power[i]

        else:  # 风电功率在正常范围内
            battery_power[i] = 0
            grid_power[i] = wind_power[i]

        # 更新SOC (15分钟间隔，功率单位MW，容量单位MWh)
        if i < periods - 1:
            soc_change = -battery_power[i] * 0.25 / battery_capacity * 100
            soc[i + 1] = max(20, min(100, soc[i] + soc_change))

    # 最终确保并网功率不超过限制
    grid_power = np.clip(grid_power, 0, grid_limit)

    # 计算实际弃风情况
    wind_energy = wind_power.sum() * 0.25
    grid_energy = grid_power.sum() * 0.25
    curtailment_energy = wind_energy - grid_energy
    curtailment_rate = (curtailment_energy / wind_energy * 100) if wind_energy > 0 else 0

    print(f"山地风电项目数据:")
    print(f"风电总能量: {wind_energy:.1f} MWh")
    print(f"并网总能量: {grid_energy:.1f} MWh")
    print(f"弃风能量: {curtailment_energy:.1f} MWh")
    print(f"弃风率: {curtailment_rate:.2f}%")
    print(f"弃风时段: {curtailment_periods}/{periods}")
    print(f"最大风电功率: {wind_power.max():.1f} MW")
    print(f"超限时段数量: {np.sum(wind_power > grid_limit)}/{periods}")

    data = pd.DataFrame({
        'wind_power': wind_power,
        'grid_power': grid_power,
        'battery_power': battery_power,
        'storage_soc': soc
    }, index=index)

    return data


def main():
    """
    独立运行时的演示函数
    """
    st.set_page_config(page_title="储能调度效果分析", layout="wide")
    st.title("🔋 山地风电项目储能调度效果分析")

    # 生成模拟数据
    optimization_result = {'time_series_data': create_realistic_sample_data()}

    # 显示储能调度效果
    display_energy_storage_performance(optimization_result, {})
