import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def display_energy_storage_performance(optimization_result, windfarm_data):
    """
    展示多风场储能调度效果与发电表现 - 支持选择特定风场查看
    """
    st.markdown("#### 🔋 多风场储能调度效果分析")

    # 检查是否有储能调度数据
    if not has_storage_data(optimization_result):
        st.warning("⚠️ 优化结果中未找到储能调度数据")
        return

    # 获取风场选择信息
    wind_farms = create_wind_farm_selector(optimization_result, windfarm_data)

    if not wind_farms:
        st.error("❌ 无法获取风场选择信息")
        return

    # 风场选择器
    selected_farm_data = display_wind_farm_selector(wind_farms)

    if not selected_farm_data:
        return

    # 使用可展开的下拉框来组织所有图表
    with st.expander(f"📊 风场 {selected_farm_data['name']} 储能调度详细分析", expanded=True):
        # 显示风场概况
        display_farm_overview(selected_farm_data)

        # 功率时序分析
        st.markdown("##### 📊 功率时序分析：风机原始出力 vs 并网出力 vs 储能充放电")
        display_power_time_series(selected_farm_data['storage_data'])

        # 爬坡速率对比分析
        st.markdown("##### 📈 爬坡速率对比分析：无储能 vs 有储能")
        display_ramp_rate_comparison(selected_farm_data['storage_data'])

        # 弃风功率分析
        st.markdown("##### 🗑️ 弃风功率分析：储能吸收 vs 实际弃风")
        display_curtailment_analysis(selected_farm_data['storage_data'])

        # SOC时间序列分析
        st.markdown("##### 🔋 储能SOC时间序列分析")
        display_soc_time_series(selected_farm_data['storage_data'])


def create_wind_farm_selector(optimization_result, windfarm_data):
    """
    创建风场选择器数据 - 修正版本：只显示实际选中的风场
    """
    wind_farms = []

    # 方法1: 从 farm_locations 获取（这是优化算法实际选中的位置）
    if 'farm_locations' in optimization_result:
        farm_locations = optimization_result['farm_locations']

        # 调试信息
        st.write(f"🔍 调试信息: 找到 {len(farm_locations)} 个风场位置")

        for i, location in enumerate(farm_locations):
            farm_id = f"farm_{i + 1}"
            farm_name = f"风场 {i + 1}"

            # 获取该风场的地理位置
            lat, lon = location[0], location[1] if len(location) >= 2 else (None, None)

            # 查找该风场对应的储能数据 - 关键修改
            storage_data = get_actual_farm_storage_data(optimization_result, farm_id, i, lat, lon, windfarm_data)

            # 计算风场性能指标
            performance_stats = calculate_farm_performance(storage_data, windfarm_data, location)

            wind_farms.append({
                'id': farm_id,
                'name': farm_name,
                'location': location,
                'lat': lat,
                'lon': lon,
                'storage_data': storage_data,
                'performance': performance_stats
            })

    # 方法2: 如果 farm_locations 不存在，从 best_positions_data 中提取前n个
    elif 'best_positions_data' in optimization_result:
        best_positions = optimization_result['best_positions_data']
        if isinstance(best_positions, pd.DataFrame) and not best_positions.empty:
            # 获取实际选择的风场数量
            n_farms = len(best_positions) if len(best_positions) <= 8 else 8  # 限制最多8个

            st.write(f"🔍 调试信息: 从 best_positions_data 中提取前 {n_farms} 个风场")

            for i, (idx, row) in enumerate(best_positions.head(n_farms).iterrows()):
                farm_id = f"farm_{i + 1}"
                farm_name = f"风场 {i + 1}"

                lat = row.get('lat', None)
                lon = row.get('lon', None)

                storage_data = get_actual_farm_storage_data(optimization_result, farm_id, i, lat, lon, windfarm_data)
                performance_stats = calculate_farm_performance(storage_data, windfarm_data, (lat, lon))

                wind_farms.append({
                    'id': farm_id,
                    'name': farm_name,
                    'location': (lat, lon),
                    'lat': lat,
                    'lon': lon,
                    'storage_data': storage_data,
                    'performance': performance_stats
                })

    # 如果还是没有找到风场，使用默认的样本数据
    if not wind_farms:
        st.warning("⚠️ 未找到风场位置数据，使用样本数据演示")
        # 创建样本风场数据
        n_sample_farms = 2  # 默认2个样本风场
        for i in range(n_sample_farms):
            farm_id = f"farm_{i + 1}"
            farm_name = f"样本风场 {i + 1}"

            # 创建样本位置（奉节县区域）
            base_lat, base_lon = 31.0451, 109.5167
            lat = base_lat + (i * 0.02)  # 每个风场间隔约2km
            lon = base_lon + (i * 0.02)

            storage_data = create_sample_farm_data(i)
            performance_stats = calculate_farm_performance(storage_data, windfarm_data, (lat, lon))

            wind_farms.append({
                'id': farm_id,
                'name': farm_name,
                'location': (lat, lon),
                'lat': lat,
                'lon': lon,
                'storage_data': storage_data,
                'performance': performance_stats
            })

    st.write(f"✅ 最终创建了 {len(wind_farms)} 个风场选择")
    return wind_farms


def get_actual_farm_storage_data(optimization_result, farm_id, farm_index, lat, lon, windfarm_data):
    """
    获取实际风场的储能数据 - 修正版本
    """
    # 首先尝试从优化结果中获取该风场的真实数据
    farm_keys = [
        f'farm_{farm_index}_data',
        f'wind_farm_{farm_index}',
        f'farm_specific_{farm_index}',
        'time_series_data'
    ]

    for key in farm_keys:
        if key in optimization_result and optimization_result[key] is not None:
            data = optimization_result[key]
            if isinstance(data, (pd.DataFrame, dict)):
                st.write(f"✅ 找到风场 {farm_id} 的真实数据")
                return data

    # 如果没有单独的风场数据，尝试从总体数据中提取该风场的数据
    if 'time_series_data' in optimization_result:
        overall_data = optimization_result['time_series_data']
        if isinstance(overall_data, pd.DataFrame) and not overall_data.empty:
            # 尝试根据位置信息匹配数据
            farm_data = extract_farm_data_by_location(overall_data, lat, lon, windfarm_data)
            if farm_data is not None:
                st.write(f"✅ 根据位置提取风场 {farm_id} 的数据")
                return farm_data

    # 最后使用样本数据
    st.write(f"⚠️ 风场 {farm_id} 使用样本数据")
    return create_sample_farm_data(farm_index)


def extract_farm_data_by_location(overall_data, target_lat, target_lon, windfarm_data):
    """
    根据位置信息从总体数据中提取特定风场的数据
    """
    if target_lat is None or target_lon is None:
        return None

    # 在风场数据中查找最近的点
    if windfarm_data is not None and 'lat' in windfarm_data.columns and 'lon' in windfarm_data.columns:
        distances = np.sqrt(
            (windfarm_data['lat'] - target_lat) ** 2 +
            (windfarm_data['lon'] - target_lon) ** 2
        )
        if len(distances) > 0:
            nearest_idx = distances.idxmin()
            nearest_point = windfarm_data.iloc[nearest_idx]

            # 这里可以根据需要从总体数据中提取对应位置的数据
            # 目前返回总体数据的副本（假设所有风场共享相同的时间序列模式）
            return overall_data.copy()

    return None


def display_wind_farm_selector(wind_farms):
    """
    显示风场选择器并返回选中的风场数据
    """
    st.markdown("##### 🎯 选择要分析的风场")

    # 创建选项卡式选择器
    if len(wind_farms) <= 4:
        # 对于4个及以下风场，使用按钮式选择
        cols = st.columns(len(wind_farms))
        selected_farm = None

        for i, farm in enumerate(wind_farms):
            with cols[i]:
                # 显示风场基本信息卡片
                btn_text = (
                    f"**{farm['name']}**\n\n"
                    f"📍位置: ({farm.get('lat', 'N/A'):.4f}, {farm.get('lon', 'N/A'):.4f})\n"
                    f"⚡平均功率: {farm['performance'].get('avg_power', 0):.1f} MW\n"
                    f"🌪️平均风速: {farm['performance'].get('avg_wind_speed', 0):.1f} m/s"
                )

                if st.button(btn_text, key=f"farm_btn_{i}", use_container_width=True):
                    selected_farm = farm

        # 如果没有选择，默认选择第一个风场
        if selected_farm is None:
            selected_farm = wind_farms[0]
            st.info(f"🔍 当前显示: {selected_farm['name']} 的储能调度数据")

        return selected_farm
    else:
        # 对于多于4个风场，使用下拉选择器
        farm_options = {f"{farm['name']} (📍{farm.get('lat', 'N/A'):.4f}, {farm.get('lon', 'N/A'):.4f})": farm
                        for farm in wind_farms}
        selected_option = st.selectbox("选择风场:", list(farm_options.keys()))
        return farm_options[selected_option]


def display_farm_overview(farm_data):
    """
    显示选中风场的概况信息
    """
    st.markdown("###### 🏭 风场基本信息")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("风场名称", farm_data['name'])

    with col2:
        lat = farm_data.get('lat', 'N/A')
        lon = farm_data.get('lon', 'N/A')
        if lat != 'N/A' and lon != 'N/A':
            st.metric("地理位置", f"({lat:.4f}, {lon:.4f})")
        else:
            st.metric("地理位置", "未知")

    with col3:
        avg_wind = farm_data['performance'].get('avg_wind_speed', 0)
        st.metric("平均风速", f"{avg_wind:.1f} m/s")

    with col4:
        capacity_factor = farm_data['performance'].get('capacity_factor', 0)
        st.metric("容量系数", f"{capacity_factor:.1%}")

    # 性能指标
    st.markdown("###### 📊 性能指标")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:
        avg_power = farm_data['performance'].get('avg_power', 0)
        st.metric("平均功率", f"{avg_power:.1f} MW")

    with perf_col2:
        max_power = farm_data['performance'].get('max_power', 0)
        st.metric("最大功率", f"{max_power:.1f} MW")

    with perf_col3:
        utilization = farm_data['performance'].get('utilization_rate', 0)
        st.metric("利用率", f"{utilization:.1%}")

    with perf_col4:
        curtailment_rate = farm_data['performance'].get('curtailment_rate', 0)
        st.metric("弃风率", f"{curtailment_rate:.1%}")


def get_farm_storage_data(optimization_result, farm_id, farm_index):
    """
    获取特定风场的储能数据
    """
    # 尝试从优化结果中获取该风场的数据
    farm_keys = [
        f'farm_{farm_index}_data',
        f'wind_farm_{farm_index}',
        f'farm_specific_{farm_index}',
        'time_series_data',
        'storage_schedule'
    ]

    for key in farm_keys:
        if key in optimization_result and optimization_result[key] is not None:
            data = optimization_result[key]
            if isinstance(data, (pd.DataFrame, dict)):
                return data

    # 如果没有单独的风场数据，从总体数据中提取或模拟
    return extract_farm_data_from_overall(optimization_result, farm_index)


def extract_farm_data_from_overall(optimization_result, farm_index):
    """
    从总体数据中提取特定风场的数据
    """
    overall_data = get_time_series_data(optimization_result)

    if overall_data is None:
        # 创建样本数据用于演示
        return create_sample_farm_data(farm_index)

    # 如果是多风场数据，尝试根据风场索引进行数据分割
    farm_specific_columns = {
        'wind_power': f'wind_power_farm_{farm_index}',
        'grid_power': f'grid_power_farm_{farm_index}',
        'battery_power': f'battery_power_farm_{farm_index}',
        'storage_soc': f'storage_soc_farm_{farm_index}'
    }

    farm_data = overall_data.copy()

    # 重命名列名为标准名称
    for standard_name, farm_specific_name in farm_specific_columns.items():
        if farm_specific_name in farm_data.columns:
            farm_data[standard_name] = farm_data[farm_specific_name]

    # 如果没有风场特定列，使用总体数据（可能所有风场共享储能系统）
    if 'wind_power' not in farm_data.columns and 'wind_power_total' in farm_data.columns:
        # 假设总功率平均分配到各风场
        farm_data['wind_power'] = farm_data['wind_power_total'] / (farm_index + 1)
        farm_data['grid_power'] = farm_data['grid_power_total'] / (farm_index + 1)
        farm_data['battery_power'] = farm_data['battery_power_total'] / (farm_index + 1)
        farm_data['storage_soc'] = farm_data['storage_soc_total']

    return farm_data


def create_sample_farm_data(farm_index):
    """
    创建样本风场数据用于演示
    """
    # 创建时间序列
    time_index = pd.date_range('2024-01-01 00:00:00', periods=96, freq='15T')

    # 基于风场索引创建不同的功率模式
    base_wind = 15 + farm_index * 2  # 不同风场的基础功率

    # 模拟风电功率（考虑不同地理位置的风速差异）
    wind_power = base_wind + 8 * np.sin(2 * np.pi * np.arange(96) / 96) + \
                 3 * np.sin(6 * np.pi * np.arange(96) / 96) + \
                 np.random.normal(0, 1, 96)

    # 添加风场特定的波动特性
    if farm_index == 0:
        wind_power = wind_power * 0.9  # 风场1功率略低
    elif farm_index == 1:
        wind_power = wind_power * 1.1  # 风场2功率略高
    elif farm_index == 2:
        # 风场3有更明显的波动
        wind_power = wind_power + 5 * np.sin(12 * np.pi * np.arange(96) / 96)

    # 限制功率在合理范围内
    wind_power = np.clip(wind_power, 0, 25)

    # 模拟储能调度效果
    grid_limit = 20  # MW电网限制
    battery_power = np.zeros(96)
    grid_power = np.zeros(96)
    storage_soc = np.zeros(96)

    current_soc = 50  # 初始SOC 50%

    for i in range(96):
        # 计算需要储能调节的功率
        excess_power = wind_power[i] - grid_limit

        if excess_power > 0:
            # 风电功率超过限制，储能充电
            charge_power = min(excess_power, 10)  # 最大充电功率10MW
            battery_power[i] = -charge_power
            grid_power[i] = grid_limit
            current_soc = min(100, current_soc + charge_power * 0.25 / 30 * 100)  # 假设储能容量30MWh
        elif wind_power[i] < grid_limit and current_soc > 20:
            # 风电功率不足，储能放电
            discharge_needed = grid_limit - wind_power[i]
            max_discharge = min(discharge_needed, 10, (current_soc - 20) / 100 * 30 / 0.25)
            battery_power[i] = max_discharge
            grid_power[i] = wind_power[i] + max_discharge
            current_soc = max(20, current_soc - max_discharge * 0.25 / 30 * 100)
        else:
            # 无需储能调节
            battery_power[i] = 0
            grid_power[i] = wind_power[i]

        storage_soc[i] = current_soc

    farm_data = pd.DataFrame({
        'timestamp': time_index,
        'wind_power': wind_power,
        'grid_power': grid_power,
        'battery_power': battery_power,
        'storage_soc': storage_soc
    })

    farm_data = farm_data.set_index('timestamp')
    return farm_data


def calculate_farm_performance(storage_data, windfarm_data, location):
    """
    计算风场性能指标
    """
    if storage_data is None:
        return {}

    time_data = get_time_series_data({'dummy': storage_data})

    if time_data is None:
        return {}

    performance = {}

    # 计算功率相关指标
    if 'wind_power' in time_data.columns:
        performance['avg_power'] = time_data['wind_power'].mean()
        performance['max_power'] = time_data['wind_power'].max()
        performance['min_power'] = time_data['wind_power'].min()

    # 计算弃风率
    if 'wind_power' in time_data.columns and 'grid_power' in time_data.columns:
        wind_energy = time_data['wind_power'].sum() * 0.25  # 15分钟间隔
        grid_energy = time_data['grid_power'].sum() * 0.25
        curtailment_energy = max(wind_energy - grid_energy, 0)
        performance['curtailment_rate'] = (curtailment_energy / wind_energy) if wind_energy > 0 else 0

    # 从原始风场数据获取风速信息
    if windfarm_data is not None and location is not None:
        lat, lon = location[0], location[1]
        if (lat is not None and lon is not None and
                'lat' in windfarm_data.columns and 'lon' in windfarm_data.columns):

            # 查找最近点的数据
            distances = np.sqrt(
                (windfarm_data['lat'] - lat) ** 2 +
                (windfarm_data['lon'] - lon) ** 2
            )
            if len(distances) > 0:
                nearest_idx = distances.idxmin()

                if 'predicted_wind_speed' in windfarm_data.columns:
                    performance['avg_wind_speed'] = windfarm_data.loc[nearest_idx, 'predicted_wind_speed']

                if 'wind_utilization_rate' in windfarm_data.columns:
                    performance['utilization_rate'] = windfarm_data.loc[nearest_idx, 'wind_utilization_rate']

    # 如果无法从原始数据获取风速，基于功率估算
    if 'avg_wind_speed' not in performance and 'avg_power' in performance:
        # 简单估算：假设功率与风速立方成正比
        performance['avg_wind_speed'] = (performance['avg_power'] / 0.5) ** (1 / 3)

    # 计算容量系数（假设单台风机容量为2.5MW）
    turbine_capacity = 2.5  # MW
    if 'avg_power' in performance:
        performance['capacity_factor'] = performance['avg_power'] / turbine_capacity

    return performance


def display_power_time_series(storage_data):
    """
    展示风机原始出力 vs 并网出力 vs 储能充放电（时间序列图）
    """
    time_data = get_time_series_data({'dummy': storage_data})

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

    st.plotly_chart(fig, use_container_width=True, key="power_chart")

    # 显示关键统计信息
    display_power_statistics(time_data, grid_limit)


def display_power_statistics(time_data, grid_limit):
    """显示功率统计信息"""
    if 'wind_power' not in time_data or 'grid_power' not in time_data:
        return

    st.markdown("###### 📈 关键功率统计")

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

    st.markdown("###### 🎯 调度效果评估")
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


def display_ramp_rate_comparison(storage_data):
    """
    展示爬坡速率对比图：无储能 vs 有储能
    """
    # 获取时间序列数据
    time_data = get_time_series_data({'dummy': storage_data})

    if time_data is None or 'wind_power' not in time_data or 'grid_power' not in time_data:
        st.error("❌ 无法获取功率时间序列数据")
        return

    # 计算爬坡速率 (MW/15min)
    time_interval_hours = 0.25  # 15分钟 = 0.25小时

    # 计算无储能时的爬坡速率（风电原始出力）
    wind_ramp = time_data['wind_power'].diff() / time_interval_hours  # MW/h

    # 计算有储能时的爬坡速率（并网出力）
    grid_ramp = time_data['grid_power'].diff() / time_interval_hours  # MW/h

    # 创建爬坡速率对比图
    fig = go.Figure()

    # 添加无储能爬坡曲线
    fig.add_trace(
        go.Scatter(
            x=time_data.index,
            y=wind_ramp,
            name='无储能 Ramp_wind(t)',
            line=dict(color='red', width=2.5),
            opacity=0.8
        )
    )

    # 添加有储能爬坡曲线
    fig.add_trace(
        go.Scatter(
            x=time_data.index,
            y=grid_ramp,
            name='有储能 Ramp_grid(t)',
            line=dict(color='green', width=2.5),
            opacity=0.9
        )
    )

    # 添加零线
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="black",
        line_width=1
    )

    # 添加典型爬坡限制线（可选）
    typical_ramp_limit = 30  # MW/h，典型电网爬坡限制
    fig.add_hline(
        y=typical_ramp_limit,
        line_dash="dash",
        line_color="orange",
        line_width=1.5,
        annotation_text=f"典型爬坡限制 +{typical_ramp_limit} MW/h",
        annotation_position="top right"
    )

    fig.add_hline(
        y=-typical_ramp_limit,
        line_dash="dash",
        line_color="orange",
        line_width=1.5,
        annotation_text=f"典型爬坡限制 -{typical_ramp_limit} MW/h",
        annotation_position="bottom right"
    )

    # 更新布局
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="爬坡速率对比分析 - 储能平滑效果验证",
        xaxis_title="时间",
        yaxis_title="爬坡速率 (MW/h)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True, key="ramp_rate_chart")

    # 显示爬坡统计信息
    display_ramp_statistics(wind_ramp, grid_ramp, typical_ramp_limit)


def display_ramp_statistics(wind_ramp, grid_ramp, ramp_limit):
    """显示爬坡速率统计信息"""
    st.markdown("###### 📊 爬坡平滑效果统计")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # 最大上升爬坡 - 确保获取标量值
        max_wind_ramp_up = wind_ramp.max()
        max_grid_ramp_up = grid_ramp.max()
        if hasattr(max_wind_ramp_up, 'iloc'):
            max_wind_ramp_up = max_wind_ramp_up.iloc[0] if len(max_wind_ramp_up) > 0 else 0
        if hasattr(max_grid_ramp_up, 'iloc'):
            max_grid_ramp_up = max_grid_ramp_up.iloc[0] if len(max_grid_ramp_up) > 0 else 0

        ramp_up_reduction = max_wind_ramp_up - max_grid_ramp_up
        improvement_up = (ramp_up_reduction / max_wind_ramp_up * 100) if max_wind_ramp_up > 0 else 0

        st.metric(
            "最大上升爬坡",
            f"{max_grid_ramp_up:.1f} MW/h",
            delta=f"降低{ramp_up_reduction:.1f} MW/h ({improvement_up:.1f}%)",
            delta_color="normal" if improvement_up > 0 else "off"
        )

    with col2:
        # 最大下降爬坡 - 确保获取标量值
        max_wind_ramp_down = abs(wind_ramp.min())
        max_grid_ramp_down = abs(grid_ramp.min())
        if hasattr(max_wind_ramp_down, 'iloc'):
            max_wind_ramp_down = max_wind_ramp_down.iloc[0] if len(max_wind_ramp_down) > 0 else 0
        if hasattr(max_grid_ramp_down, 'iloc'):
            max_grid_ramp_down = max_grid_ramp_down.iloc[0] if len(max_grid_ramp_down) > 0 else 0

        ramp_down_reduction = max_wind_ramp_down - max_grid_ramp_down
        improvement_down = (ramp_down_reduction / max_wind_ramp_down * 100) if max_wind_ramp_down > 0 else 0

        st.metric(
            "最大下降爬坡",
            f"{max_grid_ramp_down:.1f} MW/h",
            delta=f"降低{ramp_down_reduction:.1f} MW/h ({improvement_down:.1f}%)",
            delta_color="normal" if improvement_down > 0 else "off"
        )

    with col3:
        # 爬坡标准差（波动性） - 确保获取标量值
        std_wind = wind_ramp.std()
        std_grid = grid_ramp.std()
        if hasattr(std_wind, 'iloc'):
            std_wind = std_wind.iloc[0] if len(std_wind) > 0 else 0
        if hasattr(std_grid, 'iloc'):
            std_grid = std_grid.iloc[0] if len(std_grid) > 0 else 0

        std_reduction = std_wind - std_grid
        std_improvement = (std_reduction / std_wind * 100) if std_wind > 0 else 0

        st.metric(
            "爬坡波动性",
            f"{std_grid:.1f} MW/h",
            delta=f"降低{std_reduction:.1f} MW/h ({std_improvement:.1f}%)",
            delta_color="normal" if std_improvement > 0 else "off"
        )

    with col4:
        # 超限次数对比 - 确保使用正确的数据类型
        wind_ramp_abs = wind_ramp.abs()
        grid_ramp_abs = grid_ramp.abs()
        wind_exceedances = np.sum(wind_ramp_abs > ramp_limit)
        grid_exceedances = np.sum(grid_ramp_abs > ramp_limit)
        exceedance_reduction = wind_exceedances - grid_exceedances

        st.metric(
            "爬坡超限次数",
            f"{grid_exceedances}次",
            delta=f"减少{exceedance_reduction}次",
            delta_color="normal" if exceedance_reduction > 0 else "off"
        )

    # 详细效果分析
    st.markdown("###### 🎯 储能平滑效果评估")

    total_improvement = (improvement_up + improvement_down) / 2

    if total_improvement > 30:
        st.success(f"✅ **优秀平滑效果**：储能系统将最大爬坡速率平均降低 {total_improvement:.1f}%，显著提升了电网稳定性")
    elif total_improvement > 15:
        st.info(f"📊 **良好平滑效果**：储能系统将最大爬坡速率平均降低 {total_improvement:.1f}%，有效改善了功率波动")
    else:
        st.warning(f"⚠️ **有限平滑效果**：储能系统将最大爬坡速率平均降低 {total_improvement:.1f}%，建议优化储能控制策略")

    # 关键爬坡事件分析
    analyze_key_ramp_events(wind_ramp, grid_ramp, ramp_limit)


def analyze_key_ramp_events(wind_ramp, grid_ramp, ramp_limit):
    """分析关键爬坡事件"""
    # 找出最大的5个爬坡事件
    largest_wind_ramps = wind_ramp.abs().nlargest(5)

    if len(largest_wind_ramps) > 0:
        st.markdown("###### 🔍 关键爬坡事件分析")

        for i, (time_idx, wind_ramp_value) in enumerate(largest_wind_ramps.items()):
            if time_idx in grid_ramp.index:
                # 确保获取的是标量值而不是Series
                grid_ramp_value = grid_ramp.loc[time_idx]
                # 如果是Series，取第一个值
                if hasattr(grid_ramp_value, 'iloc'):
                    grid_ramp_value = grid_ramp_value.iloc[0] if len(grid_ramp_value) > 0 else 0

                reduction = abs(wind_ramp_value) - abs(grid_ramp_value)
                reduction_pct = (reduction / abs(wind_ramp_value) * 100) if wind_ramp_value != 0 else 0

                # 判断爬坡方向
                ramp_direction = "上升" if wind_ramp_value > 0 else "下降"

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**事件 {i + 1}** - {time_idx.strftime('%H:%M')} ({ramp_direction}爬坡)")
                with col2:
                    st.write(f"无储能: {abs(wind_ramp_value):.1f} MW/h")
                with col3:
                    st.write(f"有储能: {abs(grid_ramp_value):.1f} MW/h → 降低{reduction_pct:.1f}%")

                # 添加进度条显示改善效果
                if reduction_pct > 0:
                    st.progress(min(reduction_pct / 100, 1.0))
                else:
                    st.progress(0.0)


def display_curtailment_analysis(storage_data):
    """
    展示弃风功率堆叠图：显示超过20MW部分被削减 + 储能吸收情况
    """
    # 获取时间序列数据
    time_data = get_time_series_data({'dummy': storage_data})

    if time_data is None or 'wind_power' not in time_data or 'grid_power' not in time_data:
        st.error("❌ 无法获取功率时间序列数据")
        return

    # 计算弃风情况 - 基于20MW电网限制
    grid_limit = 20  # MW - 电网限制
    battery_capacity = 30  # MWh - 储能容量

    # 计算理论弃风（超过20MW电网限制的部分）
    theoretical_curtailment = np.maximum(time_data['wind_power'] - grid_limit, 0)

    # 计算储能吸收的功率（充电功率的绝对值）
    storage_absorption = np.abs(np.minimum(time_data['battery_power'], 0))

    # 计算实际弃风（理论弃风减去储能吸收后仍超限的部分）
    actual_curtailment = np.maximum(theoretical_curtailment - storage_absorption, 0)

    # 计算储能有效吸收（不超过理论弃风的部分）
    effective_storage_absorption = np.minimum(storage_absorption, theoretical_curtailment)

    # 创建弃风分析堆叠图
    fig = go.Figure()

    # 添加堆叠区域 - 展示超过20MW部分的处理情况
    fig.add_trace(go.Scatter(
        x=time_data.index,
        y=effective_storage_absorption,
        name='储能吸收 (>20MW部分)',
        stackgroup='one',
        fillcolor='rgba(255, 165, 0, 0.7)',  # 橙色
        line=dict(width=0),
        hovertemplate='%{x|%H:%M}<br>储能吸收: %{y:.1f} MW<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=time_data.index,
        y=actual_curtailment,
        name='实际弃风 (>20MW部分)',
        stackgroup='one',
        fillcolor='rgba(255, 0, 0, 0.7)',  # 红色
        line=dict(width=0),
        hovertemplate='%{x|%H:%M}<br>实际弃风: %{y:.1f} MW<extra></extra>'
    ))

    # 添加风电功率参考线
    fig.add_trace(go.Scatter(
        x=time_data.index,
        y=time_data['wind_power'],
        name='风电原始功率',
        line=dict(color='blue', width=1, dash='dot'),
        opacity=0.6,
        hovertemplate='%{x|%H:%M}<br>风电功率: %{y:.1f} MW<extra></extra>'
    ))

    # 添加并网功率线（绿色实线）
    fig.add_trace(go.Scatter(
        x=time_data.index,
        y=time_data['grid_power'],
        name='并网功率',
        line=dict(color='green', width=2),
        opacity=0.8,
        hovertemplate='%{x|%H:%M}<br>并网功率: %{y:.1f} MW<extra></extra>'
    ))

    # 添加电网限制线（20MW）
    fig.add_hline(
        y=grid_limit,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f"电网限制 {grid_limit} MW",
        annotation_position="top left"
    )

    # 更新布局
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="弃风功率分析 - 超过20MW部分处理情况",
        xaxis_title="时间",
        yaxis_title="功率 (MW)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text="🔍 堆叠区域显示超过20MW部分的处理情况",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True, key="curtailment_chart")

    # 显示弃风统计信息
    display_curtailment_statistics(
        theoretical_curtailment,
        effective_storage_absorption,
        actual_curtailment,
        time_data
    )


def display_curtailment_statistics(theoretical_curtailment, storage_absorption, actual_curtailment, time_data):
    """显示弃风统计信息"""
    st.markdown("###### 📊 超过20MW部分处理统计")

    # 计算总能量（15分钟间隔 = 0.25小时）
    total_theoretical_energy = theoretical_curtailment.sum() * 0.25  # MWh
    total_storage_energy = storage_absorption.sum() * 0.25  # MWh
    total_actual_energy = actual_curtailment.sum() * 0.25  # MWh

    # 计算比例
    absorption_ratio = (total_storage_energy / total_theoretical_energy * 100) if total_theoretical_energy > 0 else 0
    curtailment_ratio = (total_actual_energy / total_theoretical_energy * 100) if total_theoretical_energy > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "超限总能量",
            f"{total_theoretical_energy:.1f} MWh",
            help="风电功率超过20MW限制的总能量"
        )

    with col2:
        st.metric(
            "储能吸收能量",
            f"{total_storage_energy:.1f} MWh",
            delta=f"{absorption_ratio:.1f}%",
            delta_color="normal" if absorption_ratio > 0 else "off",
            help="储能系统吸收的超限能量"
        )

    with col3:
        st.metric(
            "实际弃风能量",
            f"{total_actual_energy:.1f} MWh",
            delta=f"{curtailment_ratio:.1f}%",
            delta_color="inverse" if curtailment_ratio > 0 else "off",
            help="储能满电后仍需弃风的能量"
        )

    with col4:
        # 计算储能利用率
        battery_capacity = 30  # MWh
        storage_utilization = (total_storage_energy / battery_capacity * 100) if battery_capacity > 0 else 0
        st.metric(
            "储能利用率",
            f"{storage_utilization:.1f}%",
            help="储能容量被用于吸收超限功率的利用程度"
        )

    # 详细分析
    st.markdown("###### 🎯 储能规模充足性评估")

    if total_actual_energy == 0:
        st.success("✅ **储能规模充足**：所有超过20MW的超限功率均被储能吸收，无实际弃风")
    elif curtailment_ratio < 20:
        st.info(f"📊 **储能规模基本合适**：{curtailment_ratio:.1f}%的超限功率未被吸收，建议适当增加储能容量")
    elif curtailment_ratio < 50:
        st.warning(f"⚠️ **储能规模略有不足**：{curtailment_ratio:.1f}%的超限功率未被吸收，建议增加储能容量")
    else:
        st.error(f"❌ **储能规模严重不足**：{curtailment_ratio:.1f}%的超限功率未被吸收，急需扩大储能容量")

    # 显示关键弃风时段分析
    display_key_curtailment_periods(actual_curtailment, storage_absorption, time_data)


def display_key_curtailment_periods(actual_curtailment, storage_absorption, time_data):
    """显示关键弃风时段分析"""
    # 找出实际弃风最大的5个时段
    largest_curtailments = actual_curtailment.nlargest(5)

    if len(largest_curtailments) > 0:
        st.markdown("###### 🔍 关键弃风时段分析")

        for i, (time_idx, curtailment_power) in enumerate(largest_curtailments.items()):
            if time_idx in storage_absorption.index:
                storage_power = storage_absorption.loc[time_idx]
                wind_power = time_data.loc[time_idx, 'wind_power']
                grid_limit = 20

                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.write(f"**时段 {i + 1}** - {time_idx.strftime('%H:%M')}")
                with col2:
                    st.write(f"风电: {wind_power:.1f} MW")
                with col3:
                    st.write(f"储能吸收: {storage_power:.1f} MW")
                with col4:
                    st.write(f"实际弃风: {curtailment_power:.1f} MW")

                # 显示功率分配进度条
                total_excess = wind_power - grid_limit
                if total_excess > 0:
                    storage_ratio = storage_power / total_excess

                    st.write(f"储能吸收比例: {storage_ratio:.1%}")
                    st.progress(float(storage_ratio))

                    # 添加分隔线
                    st.markdown("---")


def display_soc_time_series(storage_data):
    """
    展示储能SOC时间序列曲线，包括安全区间检查
    """
    # 获取时间序列数据
    time_data = get_time_series_data({'dummy': storage_data})

    if time_data is None or 'storage_soc' not in time_data:
        st.error("❌ 无法获取SOC时间序列数据")
        return

    # 创建SOC图表
    fig = go.Figure()

    # 添加SOC曲线
    fig.add_trace(
        go.Scatter(
            x=time_data.index,
            y=time_data['storage_soc'],
            name='储能SOC',
            line=dict(color='purple', width=3),
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.1)'
        )
    )

    # 添加安全SOC区间
    soc_min, soc_max = 20, 90  # 安全SOC区间

    # 安全区间填充
    fig.add_hrect(
        y0=soc_min, y1=soc_max,
        fillcolor="rgba(0, 255, 0, 0.1)",
        layer="below", line_width=0,
        annotation_text="安全SOC区间",
        annotation_position="top left"
    )

    # 添加上下界限
    fig.add_hline(
        y=soc_min,
        line_dash="dash",
        line_color="orange",
        line_width=2,
        annotation_text=f"下限 {soc_min}%",
        annotation_position="bottom right"
    )

    fig.add_hline(
        y=soc_max,
        line_dash="dash",
        line_color="orange",
        line_width=2,
        annotation_text=f"上限 {soc_max}%",
        annotation_position="top right"
    )

    # 检查SOC越界情况
    soc_violations = time_data[
        (time_data['storage_soc'] < soc_min) |
        (time_data['storage_soc'] > soc_max)
        ]

    # 标记越界点
    if len(soc_violations) > 0:
        fig.add_trace(
            go.Scatter(
                x=soc_violations.index,
                y=soc_violations['storage_soc'],
                name='SOC越界',
                mode='markers',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='x-thin',
                    line=dict(width=2, color='darkred')
                )
            )
        )

    # 更新布局
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="储能SOC时间序列与安全区间分析",
        xaxis_title="时间",
        yaxis_title="SOC (%)",
        yaxis_range=[0, 100],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True, key="soc_chart")

    # 显示SOC分析统计
    display_soc_statistics(time_data, soc_violations, soc_min, soc_max)


def display_soc_statistics(time_data, soc_violations, soc_min, soc_max):
    """显示SOC分析统计信息"""
    st.markdown("###### 📊 SOC运行状态分析")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # 平均SOC
        avg_soc = time_data['storage_soc'].mean()
        status_color = "normal" if soc_min <= avg_soc <= soc_max else "inverse"
        st.metric(
            "平均SOC",
            f"{avg_soc:.1f}%",
            delta="正常区间" if soc_min <= avg_soc <= soc_max else "异常",
            delta_color=status_color
        )

    with col2:
        # SOC波动范围
        soc_range = time_data['storage_soc'].max() - time_data['storage_soc'].min()
        st.metric("SOC波动范围", f"{soc_range:.1f}%")

    with col3:
        # 越界次数
        violation_count = len(soc_violations)
        violation_percent = (violation_count / len(time_data)) * 100
        st.metric(
            "SOC越界次数",
            f"{violation_count}次",
            delta=f"{violation_percent:.1f}%",
            delta_color="inverse" if violation_count > 0 else "normal"
        )

    with col4:
        # SOC变化分析
        soc_changes = time_data['storage_soc'].diff().abs()
        max_change = soc_changes.max()
        st.metric("最大SOC变化率", f"{max_change:.1f}%/15min")

    # 详细分析
    if len(soc_violations) > 0:
        st.warning(f"⚠️ SOC安全警告：检测到 {len(soc_violations)} 个时段SOC超出安全区间({soc_min}%-{soc_max}%)")

        # 显示最严重的越界情况
        worst_violation = soc_violations.loc[soc_violations['storage_soc'].idxmin()] if len(
            soc_violations) > 0 else None
        if worst_violation is not None:
            worst_soc = worst_violation['storage_soc']
            worst_time = worst_violation.name
            violation_type = "过低" if worst_soc < soc_min else "过高"
            st.error(f"**最严重越界**：{worst_time.strftime('%H:%M')}时 SOC={worst_soc:.1f}% ({violation_type})")
    else:
        st.success("✅ SOC运行正常：所有时段均在安全区间内")


def has_storage_data(optimization_result):
    """检查优化结果中是否包含储能调度数据"""
    storage_keys = [
        'storage_schedule', 'battery_power', 'grid_power', 'wind_power',
        'storage_soc', 'energy_storage_data', 'time_series_data'
    ]

    for key in storage_keys:
        if key in optimization_result and optimization_result[key] is not None:
            return True

    # 检查是否有风场特定数据
    if 'farm_locations' in optimization_result:
        return True

    return False


def get_time_series_data(optimization_result):
    """
    从优化结果中提取时间序列数据
    返回 DataFrame 或 None
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
                # 确保有正确的时间索引
                if not isinstance(data.index, pd.DatetimeIndex):
                    # 尝试从timestamp列创建索引
                    if 'timestamp' in data.columns:
                        data = data.set_index('timestamp')
                return data

            # 如果是字典格式，转换为DataFrame
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
                # 设置时间索引
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                return df

    # 检查optimization_result本身是否是DataFrame
    if isinstance(optimization_result, pd.DataFrame):
        # 数据验证和修复
        df = optimization_result.copy()

        # 检查SOC单位，如果是小数转换为百分比
        if 'storage_soc' in df.columns and df['storage_soc'].max() <= 1.0:
            df['storage_soc'] = df['storage_soc'] * 100

        # 确保有正确的时间索引
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                # 创建默认时间索引
                start_time = pd.Timestamp('2024-01-01 00:00:00')
                time_deltas = pd.timedelta_range(start='0 minutes', periods=len(df), freq='15T')
                df.index = start_time + time_deltas

        return df

    # 如果没有找到数据，返回None
    return None