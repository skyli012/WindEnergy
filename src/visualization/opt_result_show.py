import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point
from src.utils.create_map import create_fengjie_base_map


# 显示优化结果 - 数据分析部分
def display_optimization_result(result, df):
    st.subheader(f"最优风机布局与算法收敛分析（{result.get('algorithm', '未知算法')}）")

    # 🔧 使用真实计算的最优位置数据
    if 'best_positions_data' in result and not result['best_positions_data'].empty:
        # 使用优化算法返回的真实最优位置数据
        all_turbines = result['best_positions_data'].copy()
        all_turbines["turbine_id"] = [f"T{i + 1}" for i in range(len(all_turbines))]

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

            all_turbines = df.loc[valid_indices].copy().reset_index(drop=True)
            all_turbines["turbine_id"] = [f"T{i + 1}" for i in range(len(all_turbines))]

        except Exception as e:
            st.error(f"❌ 数据处理错误: {str(e)}")
            return

        # 发电量计算
        if not all_turbines.empty:
            try:
                power_results = calculate_real_power_generation(all_turbines)
            except Exception as e:
                st.warning(f"发电量计算失败，使用简化方法: {e}")
                power_results = calculate_power_generation_simple(all_turbines)
        else:
            power_results = None
            st.warning("⚠️ 没有找到有效的风机位置")

    # 🔧 关键修改：直接使用所有风机，不进行过滤
    turbines_fengjie = all_turbines  # 直接使用所有优化结果

    # 显示风机统计
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总风机数量", len(turbines_fengjie))
    with col2:
        if 'predicted_wind_speed' in turbines_fengjie.columns:
            avg_wind_speed = turbines_fengjie["predicted_wind_speed"].mean()
            st.metric("平均风速", f"{avg_wind_speed:.1f} m/s")
        else:
            st.metric("平均风速", "N/A")
    with col3:
        fitness_value = result.get('best_fitness') or result.get('fitness') or result.get('best_score') or '未知'
        st.metric("最优适应度值", f"{fitness_value:.2f}")

    # 空间过滤 - 只保留奉节县范围内的风机（用于地图显示，但不影响数据分析）
    base_map = create_fengjie_base_map()
    if base_map:
        turbines_in_fengjie = turbines_fengjie[
            turbines_fengjie.apply(lambda row: Point(row["lon"], row["lat"]).within(base_map['geometry']), axis=1)
        ]

        # 显示位置统计信息
        if len(turbines_fengjie) != len(turbines_in_fengjie):
            outside_count = len(turbines_fengjie) - len(turbines_in_fengjie)
            st.info(f"📍 {outside_count} 个风机在奉节县边界外（仍包含在分析中）")

        # 对于地图显示使用奉节县内的风机，但数据分析使用全部风机
        display_turbines = turbines_fengjie  # 使用全部风机进行数据分析
    else:
        display_turbines = turbines_fengjie

    # 如果没有任何风机，显示错误信息
    if display_turbines.empty:
        st.error("❌ 没有找到任何风机位置")
        return

    # 算法收敛过程可视化
    st.markdown("#### 算法收敛过程")
    fitness_history = result.get("fitness_history") or result.get("convergence_history") or result.get(
        "convergence_curve") or []

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
        st.plotly_chart(fig_conv, use_container_width=True, key="convergence_chart")
    else:
        st.info("📊 未找到收敛历史数据")

    st.markdown("#### 优化结果与发电量分析")

    # 重新计算发电量（基于所有风机）
    if not display_turbines.empty:
        try:
            power_results = calculate_real_power_generation(display_turbines)
        except Exception as e:
            st.warning(f"发电量计算失败，使用简化方法: {e}")
            power_results = calculate_power_generation_simple(display_turbines)

    # 显示发电量分析（如果可用）
    if power_results and not display_turbines.empty:
        # 使用真实计算的经济指标
        economic = power_results.get('economic_analysis', {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总装机容量", f"{power_results['total_capacity_mw']:.1f} MW")
        with col2:
            st.metric("年发电量", f"{power_results['total_annual_generation_gwh']:.1f} GWh")
        with col3:
            st.metric("平均容量因数", f"{power_results['average_capacity_factor']:.1%}")
        with col4:
            st.metric("等效满发小时", f"{power_results['equivalent_full_load_hours']:.0f} h")

        st.markdown("#### 经济效益分析")

        # 使用真实计算的经济指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总投资", f"{economic.get('total_investment', 0) / 1e8:.2f} 亿元")
        with col2:
            st.metric("年发电收入", f"{economic.get('annual_revenue', 0) / 1e8:.2f} 亿元")
        with col3:
            st.metric("年运维成本", f"{economic.get('annual_om_cost', 0) / 1e8:.2f} 亿元")
        with col4:
            profit = economic.get('annual_profit', 0)
            profit_color = "normal" if profit >= 0 else "inverse"
            st.metric("年净利润", f"{profit / 1e8:.2f} 亿元", delta_color=profit_color)

        payback_period = economic.get('payback_period', float('inf'))
        if payback_period < float('inf'):
            st.metric("投资回收期", f"{payback_period:.1f} 年")
        else:
            st.metric("投资回收期", "无法回收", delta="亏损运营", delta_color="inverse")

        # 发电量分布分析
        st.markdown("#### 发电量分布分析")
        col1, col2 = st.columns(2)
        with col1:
            if power_results['capacity_factors']:
                fig_cf = go.Figure()
                fig_cf.add_trace(go.Histogram(
                    x=power_results['capacity_factors'],
                    nbinsx=20,
                    name="容量因数分布",
                    marker_color='skyblue'
                ))
                fig_cf.update_layout(
                    title="风机容量因数分布",
                    xaxis_title="容量因数",
                    yaxis_title="风机数量",
                    template="plotly_white"
                )
                st.plotly_chart(fig_cf, use_container_width=True, key="capacity_factor_histogram")

        with col2:
            if (power_results['annual_generation_per_turbine'] and
                    'predicted_wind_speed' in display_turbines.columns):
                fig_wind = go.Figure()
                fig_wind.add_trace(go.Scatter(
                    x=display_turbines["predicted_wind_speed"],
                    y=[gen / 1e6 for gen in power_results['annual_generation_per_turbine']],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=power_results['capacity_factors'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="容量因数")
                    ),
                    text=[f"T{i + 1}" for i in range(len(display_turbines))],
                    name="风机"
                ))
                fig_wind.update_layout(
                    title="风速与年发电量关系",
                    xaxis_title="风速 (m/s)",
                    yaxis_title="年发电量 (GWh)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_wind, use_container_width=True, key="wind_generation_scatter")

    else:
        # 基础信息显示（当发电量计算不可用时）
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("风机数量", len(display_turbines))
        with col2:
            if len(display_turbines) > 0 and 'predicted_wind_speed' in display_turbines.columns:
                avg_wind_speed = display_turbines["predicted_wind_speed"].mean()
                st.metric("平均风速", f"{avg_wind_speed:.1f} m/s")
            else:
                st.metric("平均风速", "N/A")
        with col3:
            if len(display_turbines) > 0 and 'wind_power_density' in display_turbines.columns:
                total_power_density = display_turbines['wind_power_density'].sum()
                st.metric("总功率密度", f"{total_power_density:.0f} W/m²")
            else:
                st.metric("数据列", "缺少功率密度")
        with col4:
            if len(display_turbines) > 0 and 'elevation' in display_turbines.columns:
                avg_elevation = display_turbines['elevation'].mean()
                st.metric("平均海拔", f"{avg_elevation:.0f} m")
            else:
                st.metric("平均海拔", "N/A")

    # 风机详细信息表格 - 显示所有风机
    st.markdown("#### 风机详细信息")
    if not display_turbines.empty:
        # 选择要显示的列 - 基于你的真实数据集
        display_columns = ["turbine_id", "lat", "lon"]

        # 添加可用的数据列
        optional_columns = {
            "predicted_wind_speed": "predicted_wind_speed",
            "elevation": "elevation",
            "slope": "slope",
            "cost": "cost",
            "road_distance": "road_distance",
            "residential_distance": "residential_distance",
            "wind_power_density": "wind_power_density"
        }

        for col_key, col_name in optional_columns.items():
            if col_name in display_turbines.columns:
                display_columns.append(col_name)

        display_df = display_turbines[display_columns].copy()

        # 格式化数值
        if "lat" in display_df.columns:
            display_df["lat"] = display_df["lat"].round(4)
        if "lon" in display_df.columns:
            display_df["lon"] = display_df["lon"].round(4)
        if "predicted_wind_speed" in display_df.columns:
            display_df["predicted_wind_speed"] = display_df["predicted_wind_speed"].round(2)
        if "elevation" in display_df.columns:
            display_df["elevation"] = display_df["elevation"].round(0)
        if "slope" in display_df.columns:
            display_df["slope"] = display_df["slope"].round(1)
        if "cost" in display_df.columns:
            display_df["cost"] = display_df["cost"].round(0)
        if "road_distance" in display_df.columns:
            display_df["road_distance"] = display_df["road_distance"].round(0)
        if "residential_distance" in display_df.columns:
            display_df["residential_distance"] = display_df["residential_distance"].round(0)
        if "wind_power_density" in display_df.columns:
            display_df["wind_power_density"] = display_df["wind_power_density"].round(0)

        # 添加发电量信息（如果可用）
        if (power_results and
                len(power_results['annual_generation_per_turbine']) == len(display_turbines)):
            display_df["年发电量(GWh)"] = [f"{x / 1e6:.2f}" for x in power_results['annual_generation_per_turbine']]
            display_df["容量因数"] = [f"{x:.1%}" for x in power_results['capacity_factors']]

        st.dataframe(display_df, use_container_width=True, key="turbine_details_table")

        # 显示风机配置说明（如果发电量计算成功）
        if power_results:
            st.markdown("#### 风机配置说明")
            config = power_results['turbine_config']
            st.write(f"""
            - 风机型号: {config['model']}
            - 单机容量: {config['rated_power'] / 1000} MW
            - 风轮直径: {config['rotor_diameter']} 米
            - 轮毂高度: {config['hub_height']} 米
            - 工作风速: {config['cut_in_speed']}-{config['rated_speed']}-{config['cut_out_speed']} m/s
            - 综合效率: {config.get('efficiency', 0.45):.0%}
            - 计算方法: 基于真实风速数据和功率曲线
            """)
    else:
        st.info("没有找到任何风机位置")

    # 显示计算时间
    if 'computation_time' in result:
        st.info(f"🕒 计算耗时: {result['computation_time']:.2f} 秒")

def calculate_real_power_generation(turbines_df):
    """基于真实风速数据计算发电量"""
    if turbines_df.empty:
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

    for _, turbine in turbines_df.iterrows():
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
    total_capacity = len(turbines_df) * TURBINE_CONFIG['rated_power']
    equivalent_full_load_hours = total_annual_generation / total_capacity if total_capacity > 0 else 0

    # 计算真实的经济指标
    electricity_price = 0.4  # 元/kWh
    investment_per_kw = 6000  # 元/kW
    om_cost_per_kw = 150  # 元/kW/年

    total_investment = total_capacity * investment_per_kw
    annual_revenue = total_annual_generation * electricity_price
    annual_om_cost = total_capacity * om_cost_per_kw
    annual_profit = annual_revenue - annual_om_cost
    payback_period = total_investment / annual_profit if annual_profit > 0 else float('inf')

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
        'turbine_config': TURBINE_CONFIG,
        'economic_analysis': {
            'total_investment': total_investment,
            'annual_revenue': annual_revenue,
            'annual_om_cost': annual_om_cost,
            'annual_profit': annual_profit,
            'payback_period': payback_period,
            'electricity_price': electricity_price,
            'investment_per_kw': investment_per_kw
        }
    }


# 简化版发电量计算（备用）
def calculate_power_generation_simple(turbines_df):
    """简化的发电量计算（备用）"""
    return calculate_real_power_generation(turbines_df)


# 保留原始函数（兼容性）
def calculate_power_generation(turbines_df):
    try:
        return calculate_real_power_generation(turbines_df)
    except Exception as e:
        st.warning(f"使用简化发电量计算: {e}")
        return calculate_power_generation_simple(turbines_df)


# 数据质量检查函数
def check_data_quality_for_power_calculation(turbines_df):
    if turbines_df.empty:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        wind_speeds = turbines_df["predicted_wind_speed"]
        avg_wind_speed = wind_speeds.mean()
        st.metric("平均风速", f"{avg_wind_speed:.1f} m/s")
        if avg_wind_speed < 5.0:
            st.error("风速偏低")
        elif avg_wind_speed > 12.0:
            st.warning("风速偏高")

    with col2:
        wind_std = wind_speeds.std()
        st.metric("风速标准差", f"{wind_std:.1f} m/s")
        if wind_std < 0.5:
            st.warning("风速变化较小")

    with col3:
        valid_ratio = (wind_speeds >= 3.0).mean() * 100
        st.metric("有效风速比例", f"{valid_ratio:.1f}%")
        if valid_ratio < 80:
            st.warning("部分点位风速过低")