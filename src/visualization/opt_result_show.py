import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point

from src.optimization.algorithm_convergence_curve import evaluate_solution_quality
from src.utils.create_map import create_maale_gilboa_base_map
from src.utils.plotting_functions import create_convergence_chart, create_wind_farm_tables, create_wind_resource_tables, \
    create_optimization_comparison_table, create_wind_speed_histogram


# 显示优化结果 - 数据分析部分
def display_optimization_result(result, df):
    st.subheader(f"最优风电场布局与算法收敛分析（{result.get('algorithm', '未知算法')}）")

    # 🔧 明确获取收敛曲线数据
    algorithm_name = result.get('algorithm', '未知算法')

    # 如果是从多算法对比来的结果，需要特殊处理
    if 'comparison_source' in result:
        algorithm_name = result.get('comparison_source', algorithm_name)

    # 获取正确的收敛曲线
    fitness_history = result.get("fitness_history")

    # 如果收敛曲线过长（超过5000次），截取或警告
    if fitness_history and len(fitness_history) > 5000:
        st.warning(f"⚠️ 收敛曲线包含 {len(fitness_history)} 次迭代，显示前5000次")
        fitness_history = fitness_history[:5000]

    # 显示算法信息
    st.info(f"**算法**: {algorithm_name} | **迭代次数**: {len(fitness_history) if fitness_history else 'N/A'}")

    # 🔧 使用真实计算的最优位置数据
    if 'best_positions_data' in result and not result['best_positions_data'].empty:
        # 使用优化算法返回的真实最优位置数据
        all_wind_farm = result['best_positions_data'].copy()
        all_wind_farm["turbine_id"] = [f"T{i + 1}" for i in range(len(all_wind_farm))]

        # 使用优化算法计算的真实发电量结果
        power_results = result.get('power_results')

    else:
        # 回退到原来的方式（兼容性）
        st.warning("⚠️ 使用兼容模式，可能不是最优结果")

        # 🔧 修复：兼容不同的解决方案键名
        sol = None
        possible_solution_keys = ['solution', 'best_positions', 'positions', 'selected_indices', 'best_solution']

        for key in possible_solution_keys:
            if key in result and result[key]:
                sol = result[key]
                break

        # 如果没有找到明确的解决方案键，尝试找到第一个列表/数组类型的值
        if sol is None:
            for key, value in result.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    sol = value
                    break

        if sol is None or len(sol) == 0:
            st.error("❌ 没有找到有效的解决方案")
            return

        # 数据预处理 - 确保索引有效
        try:
            # 过滤掉超出数据范围的索引
            valid_indices = [idx for idx in sol if idx in df.index]
            if not valid_indices:
                st.error("❌ 解决方案中的索引不在数据范围内")
                return

            all_wind_farm = df.loc[valid_indices].copy().reset_index(drop=True)
            all_wind_farm["turbine_id"] = [f"T{i + 1}" for i in range(len(all_wind_farm))]

        except Exception as e:
            st.error(f"❌ 数据处理错误: {str(e)}")
            return

        # 发电量计算
        if not all_wind_farm.empty:
            try:
                power_results = calculate_real_power_generation(all_wind_farm)
            except Exception as e:
                st.warning(f"发电量计算失败，使用简化方法: {e}")
                power_results = calculate_power_generation_simple(all_wind_farm)
        else:
            power_results = None
            st.warning("⚠️ 没有找到有效的风电场位置")

    # 🔧 关键修改：直接使用所有风电场，不进行过滤
    wind_farm_fengjie = all_wind_farm  # 直接使用所有优化结果

    # 显示风电场统计
    col1, col2, col3, col4 = st.columns(4)  # 改为4列
    with col1:
        st.metric("风电场风机总数", len(wind_farm_fengjie))
    with col2:
        if 'predicted_wind_speed' in wind_farm_fengjie.columns:
            avg_wind_speed = wind_farm_fengjie["predicted_wind_speed"].mean()
            st.metric("风电场平均风速", f"{avg_wind_speed:.1f} m/s")
        else:
            st.metric("平均风速", "N/A")
    with col3:
        fitness_value = result.get('best_fitness') or result.get('fitness') or result.get('best_score') or 0
        st.metric("最优适应度值", f"{fitness_value:.2f}")
    with col4:
        quality_rating = evaluate_solution_quality(fitness_value)
        st.metric("质量评级", quality_rating)

    # 空间过滤 - 只保留奉节县范围内的风电场（用于地图显示，但不影响数据分析）
    base_map = create_maale_gilboa_base_map()
    if base_map:
        wind_farm_in_fengjie = wind_farm_fengjie[
            wind_farm_fengjie.apply(lambda row: Point(row["lon"], row["lat"]).within(base_map['geometry']), axis=1)
        ]

        # 显示位置统计信息
        if len(wind_farm_fengjie) != len(wind_farm_in_fengjie):
            outside_count = len(wind_farm_fengjie) - len(wind_farm_in_fengjie)
            st.info(f"📍 {outside_count} 个风机在奉节县边界外（仍包含在分析中）")

        # 对于地图显示使用奉节县内的风电场，但数据分析使用全部风电场
        display_wind_farm = wind_farm_fengjie  # 使用全部风电场进行数据分析
    else:
        display_wind_farm = wind_farm_fengjie

    # 如果没有任何风电场，显示错误信息
    if display_wind_farm.empty:
        st.error("❌ 没有找到任何风电场位置")
        return

    # 算法收敛过程可视化
    st.markdown("#### 算法收敛过程")
    fitness_history = result.get("fitness_history") or result.get("convergence_history") or result.get(
        "convergence_curve") or []

    # 使用绘图函数创建收敛图表
    create_convergence_chart(fitness_history)

    st.markdown("#### 选址优化数据分析")

    # 🔧 修改：将所有详细分析内容放在下拉框中
    with st.expander("📈 详细优化分析与数据表格（点击展开）", expanded=False):
        # 🔧 新增：风速分布直方图（放在优化对比表格之前）
        st.markdown("#### 🌬️ 风速分布分析")

        # 计算基准数据（随机样本作为对比）
        baseline_data = calculate_baseline_data(df, len(wind_farm_fengjie))

        # 创建风速分布图表
        create_wind_speed_histogram(wind_farm_fengjie, baseline_data)

        # 优化前后性能指标对比
        st.markdown("#### 📈 优化算法性能指标对比")

        # 计算优化后的各项指标
        optimized_metrics = calculate_optimized_metrics(wind_farm_fengjie, power_results)

        # 计算基准指标（使用原始数据集中的随机样本作为对比）
        baseline_metrics = calculate_baseline_metrics(df, len(wind_farm_fengjie))

        # 创建对比表格
        create_optimization_comparison_table(baseline_metrics, optimized_metrics)

        # 风场详细数据统计
        st.markdown("#### 📊 风场详细数据统计")

        # 获取风场数量
        n_farms = st.session_state.get('n_farms', 2)
        n_turbines_per_farm = st.session_state.get('n_turbines_per_farm', 4)

        # 使用绘图函数创建风场数据表格
        create_wind_farm_tables(wind_farm_fengjie, n_farms, n_turbines_per_farm)

        # 风能资源性能表格
        st.markdown("#### 🌬️ 风能资源性能分析")

        # 使用绘图函数创建风能资源性能表格
        create_wind_resource_tables(wind_farm_fengjie, n_farms, n_turbines_per_farm)

        # 🔧 新增：储能调度分析（放在最底部）
        st.markdown("---")
        st.markdown("#### ⚡ 储能调度分析")

        # 检查是否有储能调度数据
        if 'storage_results' in result or 'best_strategy' in result:
            try:
                # 导入储能调度显示函数
                from src.visualization.storage_schedule_display import display_storage_schedule_analysis
                # 显示储能调度分析
                display_storage_schedule_analysis(result, df)
            except ImportError:
                st.warning("⚠️ 储能调度显示模块导入失败")
            except Exception as e:
                st.error(f"❌ 储能调度分析显示错误: {str(e)}")
        else:
            st.info("ℹ️ 当前优化结果不包含储能调度数据。要启用储能调度分析，请在优化参数中配置储能策略。")

            # 显示如何启用储能调度的提示
            with st.expander("💡 如何启用储能调度分析？"):
                st.markdown("""
                    要启用储能调度分析功能，请进行以下配置：

                    1. **选择多策略优化**：在优化参数中选择"多策略优化"
                    2. **配置储能参数**：
                       - 储能容量 (kWh)
                       - 最大充放电功率 (kW)
                       - 电网容量 (kW)
                    3. **选择储能策略**：
                       - 平滑输出
                       - 削峰填谷  
                       - 混合模式

                    启用后，系统将自动分析不同储能策略的效果并显示详细调度数据。
                    """)


def calculate_baseline_data(df, sample_size):
    """计算基准数据（随机样本）"""
    if df.empty or sample_size <= 0:
        return None

    # 从原始数据集中随机选择相同数量的样本作为基准
    if len(df) > sample_size:
        baseline_sample = df.sample(n=sample_size, random_state=42)
    else:
        baseline_sample = df.copy()

    return baseline_sample


def calculate_optimized_metrics(wind_farm_df, power_results):
    """计算优化后的各项性能指标"""
    if wind_farm_df.empty:
        return {}

    metrics = {}

    # 基础统计指标
    metrics['风机数量'] = len(wind_farm_df)

    # 风速相关指标
    if 'predicted_wind_speed' in wind_farm_df.columns:
        metrics['平均风速'] = wind_farm_df['predicted_wind_speed'].mean()
        metrics['最大风速'] = wind_farm_df['predicted_wind_speed'].max()
        metrics['最小风速'] = wind_farm_df['predicted_wind_speed'].min()
        metrics['风速标准差'] = wind_farm_df['predicted_wind_speed'].std()

    # 地形相关指标
    if 'slope' in wind_farm_df.columns:
        metrics['平均坡度'] = wind_farm_df['slope'].mean()
        metrics['最大坡度'] = wind_farm_df['slope'].max()
        metrics['坡度标准差'] = wind_farm_df['slope'].std()

    # 海拔相关指标
    if 'elevation' in wind_farm_df.columns:
        metrics['平均海拔'] = wind_farm_df['elevation'].mean()
        metrics['海拔范围'] = f"{wind_farm_df['elevation'].min():.0f}-{wind_farm_df['elevation'].max():.0f}"

    # 距离相关指标
    if 'road_distance' in wind_farm_df.columns:
        metrics['到道路平均距离'] = wind_farm_df['road_distance'].mean()
    if 'residential_distance' in wind_farm_df.columns:
        metrics['到居民区平均距离'] = wind_farm_df['residential_distance'].mean()
    if 'water_distance' in wind_farm_df.columns:
        metrics['到水体平均距离'] = wind_farm_df['water_distance'].mean()

    # 发电量指标（从power_results获取）
    if power_results:
        metrics['年发电量'] = power_results.get('total_annual_generation_gwh', 0)
        metrics['总装机容量'] = power_results.get('total_capacity_mw', 0)
        metrics['平均容量因数'] = power_results.get('average_capacity_factor', 0) * 100  # 转换为百分比
        metrics['等效满发小时'] = power_results.get('equivalent_full_load_hours', 0)

    # 风能资源指标
    air_density = 1.225
    if '平均风速' in metrics:
        metrics['风能密度'] = 0.5 * air_density * (metrics['平均风速'] ** 3)

    return metrics


def calculate_baseline_metrics(df, sample_size):
    """计算基准指标（使用原始数据集中的随机样本）"""
    if df.empty or sample_size <= 0:
        return {}

    # 从原始数据集中随机选择相同数量的样本作为基准
    if len(df) > sample_size:
        baseline_sample = df.sample(n=sample_size, random_state=42)
    else:
        baseline_sample = df.copy()

    # 计算基准样本的发电量
    baseline_power_results = calculate_real_power_generation(baseline_sample)

    # 计算基准指标
    baseline_metrics = calculate_optimized_metrics(baseline_sample, baseline_power_results)

    return baseline_metrics


def calculate_real_power_generation(wind_farm_df):
    """基于真实风速数据计算风电场发电量"""
    if wind_farm_df.empty:
        return None

    TURBINE_CONFIG = {
        'model': '金风科技 GW-140/2500',
        'rated_power': 2500,  # kW
        'rotor_diameter': 140,  # 米
        'hub_height': 90,  # 米
        'cut_in_speed': 3.0,  # m/s
        'rated_speed': 11.0,  # m/s
        'cut_out_speed': 25.0,  # m/s
        'efficiency': 0.45,  # 综合效率
    }

    def power_curve(wind_speed):
        """基于真实功率曲线计算输出功率"""
        if wind_speed < TURBINE_CONFIG['cut_in_speed']:
            return 0
        elif wind_speed < TURBINE_CONFIG['rated_speed']:
            # 立方关系计算功率
            return TURBINE_CONFIG['rated_power'] * (
                    (wind_speed ** 3 - TURBINE_CONFIG['cut_in_speed'] ** 3) /
                    (TURBINE_CONFIG['rated_speed'] ** 3 - TURBINE_CONFIG['cut_in_speed'] ** 3)
            )
        elif wind_speed <= TURBINE_CONFIG['cut_out_speed']:
            return TURBINE_CONFIG['rated_power']
        else:
            return 0

    annual_generation_per_turbine = []
    capacity_factors = []

    for _, turbine in wind_farm_df.iterrows():
        wind_speed = turbine.get('predicted_wind_speed', 0)

        # 计算理论功率输出
        theoretical_power = power_curve(wind_speed)

        # 考虑综合效率
        actual_power = theoretical_power * TURBINE_CONFIG['efficiency']

        # 年发电量 (kWh) - 8760小时/年
        annual_energy = actual_power * 8760

        annual_generation_per_turbine.append(annual_energy)

        # 容量因数
        capacity_factor = annual_energy / (TURBINE_CONFIG['rated_power'] * 8760)
        capacity_factors.append(capacity_factor)

    total_annual_generation = sum(annual_generation_per_turbine)
    avg_capacity_factor = np.mean(capacity_factors) if capacity_factors else 0
    total_capacity = len(wind_farm_df) * TURBINE_CONFIG['rated_power']
    equivalent_full_load_hours = total_annual_generation / total_capacity if total_capacity > 0 else 0

    return {
        'total_annual_generation_kwh': total_annual_generation,
        'total_annual_generation_mwh': total_annual_generation / 1000,
        'total_annual_generation_gwh': total_annual_generation / 1e6,
        'total_capacity_kw': total_capacity,
        'total_capacity_mw': total_capacity / 1000,
        'average_capacity_factor': avg_capacity_factor,
        'equivalent_full_load_hours': equivalent_full_load_hours,
        'annual_generation_per_turbine': annual_generation_per_turbine,
        'capacity_factors': capacity_factors,
        'turbine_config': TURBINE_CONFIG
    }


# 简化版发电量计算（备用）
def calculate_power_generation_simple(wind_farm_df):
    """简化的风电场发电量计算（备用）"""
    return calculate_real_power_generation(wind_farm_df)


# 保留原始函数（兼容性）
def calculate_power_generation(wind_farm_df):
    try:
        return calculate_real_power_generation(wind_farm_df)
    except Exception as e:
        st.warning(f"使用简化发电量计算: {e}")
        return calculate_power_generation_simple(wind_farm_df)


# 数据质量检查函数
def check_data_quality_for_power_calculation(wind_farm_df):
    if wind_farm_df.empty:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        wind_speeds = wind_farm_df["predicted_wind_speed"]
        avg_wind_speed = wind_speeds.mean()
        st.metric("风电场平均风速", f"{avg_wind_speed:.1f} m/s")
        if avg_wind_speed < 5.0:
            st.error("风速偏低")
        elif avg_wind_speed > 12.0:
            st.warning("风速偏高")

    with col2:
        wind_std = wind_speeds.std()
        st.metric("风电场风速标准差", f"{wind_std:.1f} m/s")
        if wind_std < 0.5:
            st.warning("风速变化较小")

    with col3:
        valid_ratio = (wind_speeds >= 3.0).mean() * 100
        st.metric("风电场有效风速比例", f"{valid_ratio:.1f}%")
        if valid_ratio < 80:
            st.warning("部分点位风速过低")
def display_wind_utilization_analysis(result, df):
    """
    显示风能利用率分析结果
    """
    # 获取优化结果中的风电场位置
    if 'farm_locations' in result:
        farm_locations = result['farm_locations']

        # 计算每个风电场的平均利用率
        utilization_data = []
        for i, farm_loc in enumerate(farm_locations):
            farm_df = df[df['lat'] == farm_loc[0]]  # 假设通过经纬度匹配
            if not farm_df.empty:
                avg_wind_speed = farm_df['predicted_wind_speed'].mean()
                avg_utilization = farm_df['wind_utilization_rate'].mean()
                composite_score = farm_df['composite_score'].mean()

                utilization_data.append({
                    '风场编号': i + 1,
                    '平均风速(m/s)': f"{avg_wind_speed:.1f}",
                    '风能利用率': f"{avg_utilization:.1%}",
                    '综合评分': f"{composite_score:.3f}"
                })

        # 显示利用率表格
        if utilization_data:
            utilization_df = pd.DataFrame(utilization_data)
            st.table(utilization_df)

            # 显示优化目标权重信息
            st.info(f"优化目标权重：风速({result.get('wind_speed_weight', 0.6)}) : "
                    f"利用率({result.get('utilization_weight', 0.4)})")

def calculate_wind_utilization(wind_speed_series):
    """
    计算风能利用率指标
    基于风速的稳定性、可利用小时数等因素
    """
    # 风速在风机工作范围内的比例（3-25 m/s）
    operational_hours = ((wind_speed_series >= 3) & (wind_speed_series <= 25)).mean()

    # 风速稳定性（标准差越小越稳定）
    wind_std = wind_speed_series.std()
    stability = 1 / (1 + wind_std)  # 标准化稳定性指标

    # 高风速利用率（>7 m/s 的比例）
    high_wind_hours = (wind_speed_series >= 7).mean()

    # 综合利用率指标
    utilization_rate = 0.5 * operational_hours + 0.3 * stability + 0.2 * high_wind_hours

    return utilization_rate