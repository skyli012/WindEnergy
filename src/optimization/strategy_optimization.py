import geopandas as gpd
import streamlit as st
import pandas as pd
import numpy as np
from shapely.geometry import Point

from src.optimization.algorithm_convergence_curve import call_optimize_function
from src.prediction.ai_prediction import calculate_metrics
from src.utils.check_data import check_data_quality
from src.utils.create_map import display_fengjie_standalone_map, display_environment, display_optimization_map, \
    create_fengjie_base_map
from src.visualization.energy_storage_scheduler import calculate_wind_power_from_speed, EnergyStorageScheduler, \
    create_single_turbine_assessment, create_wind_farm_assessment
from src.visualization.opt_result_show import display_optimization_result


# ======================================================
# 🌬️ 主页面：风电场选址优化系统
# ======================================================
def strategy_optimization_page():
    # 页面标题 - 更紧凑
    st.markdown("### 🌬️ 风电场选址优化与储能调度系统")
    st.caption("基于真实优化算法计算 · 奉节县风机布局优化 · 储能消纳策略分析")

    # 初始化 session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "map"

    # ========== 地图在左边，控制面板在右边 ==========
    map_col, control_col = st.columns([2, 1])

    with map_col:
        # 显示地图内容
        if st.session_state.current_page == "map":
            display_fengjie_standalone_map()
            if "windfarm_data" not in st.session_state:
                st.info("📁 请先上传风速预测数据以查看风能分布")

        elif st.session_state.current_page == "wind":
            if "windfarm_data" in st.session_state:
                display_environment(st.session_state["windfarm_data"])
                if "optimization_result" not in st.session_state:
                    st.info("⚙️ 数据已就绪，可点击'开始优化'进行布局优化")
            else:
                st.warning("⚠️ 请先上传数据文件")
                st.session_state.current_page = "map"
                st.rerun()

        elif st.session_state.current_page == "result":
            if "windfarm_data" in st.session_state and "optimization_result" in st.session_state:
                # 在左侧地图上显示优化结果（风机位置）
                display_optimization_map(
                    st.session_state["optimization_result"],
                    st.session_state["windfarm_data"]
                )
            else:
                st.warning("⚠️ 请先完成优化计算")
                st.session_state.current_page = "wind"
                st.rerun()

    with control_col:
        st.markdown("#### ⚙️ 控制面板")

        # 算法选择单独一行
        algo = st.selectbox("优化算法",
                            ["遗传算法", "模拟退火算法", "粒子群优化算法", "PuLP优化求解器"],
                            help="选择优化算法")

        # 基础参数设置 - 独立显示
        st.markdown("**🎯 基础参数设置**")
        col1, col2 = st.columns(2)
        with col1:
            n_turbines = st.slider("风机数量", 1, 15, 5, help="选择要安装的风机数量")
        with col2:
            cost_weight = st.slider("成本权重", 0.1, 2.0, 1.0, 0.1, help="成本在优化中的重要性，值越大成本影响越大")

        # 储能系统参数
        st.markdown("**🔋 储能系统参数**")
        col3, col4 = st.columns(2)
        with col3:
            storage_capacity = st.slider("储能容量 (MWh)", 1, 50, 10, help="储能系统总容量")
        with col4:
            max_power = st.slider("最大功率 (MW)", 1, 20, 5, help="储能系统最大充放电功率")

        # 调度策略选择
        strategy = st.selectbox("储能调度策略",
                                ["出力平滑", "弃风消减", "混合策略"],
                                help="选择储能系统运行策略")

        # 固定约束条件值
        algorithm_params = {
            'n_turbines': n_turbines,
            'cost_weight': cost_weight,
            'max_slope': 35,
            'max_road_distance': 100,
            'min_residential_distance': 60,
            'min_heritage_distance': 70,
            'min_geology_distance': 80,
            'min_water_distance': 100,
            'storage_capacity': storage_capacity * 1000,  # 转换为kWh
            'max_power': max_power * 1000,  # 转换为kW
            'strategy': strategy
        }

        # 算法高级参数（可选）
        st.markdown("**🔧 算法高级参数（可选）**")
        with st.expander("算法高级参数设置", expanded=False):
            if algo == "遗传算法":
                col11, col12 = st.columns(2)
                with col11:
                    algorithm_params['pop_size'] = st.slider("种群大小", 20, 200, 50,
                                                             help="种群越大，搜索能力越强，但计算越慢")
                with col12:
                    algorithm_params['generations'] = st.slider("迭代代数", 50, 500, 100,
                                                                help="迭代次数越多，结果可能越好，但计算时间越长")

                col13, col14 = st.columns(2)
                with col13:
                    algorithm_params['mutation_rate'] = st.slider("变异率", 0.01, 0.3, 0.1, 0.01,
                                                                  help="变异率太高会破坏好解，太低会早熟收敛")
                with col14:
                    algorithm_params['crossover_rate'] = st.slider("交叉率", 0.5, 1.0, 0.8, 0.05,
                                                                   help="控制个体间交换信息的概率")

            elif algo == "模拟退火算法":
                col15, col16, col17 = st.columns(3)
                with col15:
                    algorithm_params['initial_temp'] = st.slider("初始温度", 100, 5000, 1000, 100,
                                                                 help="温度越高，接受差解的概率越大")
                with col16:
                    algorithm_params['cooling_rate'] = st.slider("降温速率", 0.85, 0.99, 0.95, 0.01,
                                                                 help="降温越慢，找到全局最优的概率越大")
                with col17:
                    algorithm_params['iterations_per_temp'] = st.slider("每温度迭代次数", 10, 200, 50,
                                                                        help="在每个温度下的搜索次数")

            elif algo == "粒子群优化算法":
                col18, col19 = st.columns(2)
                with col18:
                    algorithm_params['pop_size'] = st.slider("粒子数量", 20, 100, 30,
                                                             help="粒子数量影响搜索能力")
                with col19:
                    algorithm_params['generations'] = st.slider("迭代次数", 50, 500, 100,
                                                                help="迭代次数越多，结果可能越好")

                col20, col21, col22 = st.columns(3)
                with col20:
                    algorithm_params['w'] = st.slider("惯性权重", 0.1, 1.0, 0.7, 0.1,
                                                      help="控制粒子速度的保持程度")
                with col21:
                    algorithm_params['c1'] = st.slider("个体学习因子", 0.1, 2.0, 1.5, 0.1,
                                                       help="控制个体经验的影响")
                with col22:
                    algorithm_params['c2'] = st.slider("社会学习因子", 0.1, 2.0, 1.5, 0.1,
                                                       help="控制群体经验的影响")

            elif algo == "PuLP优化求解器":
                col23, col24 = st.columns(2)
                with col23:
                    algorithm_params['solver_type'] = st.selectbox(
                        "求解器类型",
                        ["CBC", "GLPK", "CPLEX"],
                        help="选择线性规划求解器"
                    )
                with col24:
                    algorithm_params['time_limit'] = st.slider("时间限制(秒)", 10, 300, 60,
                                                               help="求解器最大运行时间")

        # 文件上传和处理
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📂 上传风速预测数据", type=["csv"])

        # 在处理文件上传的部分添加边界过滤
        if uploaded_file is not None:
            if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                df = pd.read_csv(uploaded_file)

                # 添加必要的列
                if "predicted_wind_speed" in df.columns:
                    df["wind_power_density"] = 0.5 * 1.225 * (df["predicted_wind_speed"] ** 3)

                # 首先过滤奉节县边界内的点
                base_map = create_fengjie_base_map()
                if base_map:
                    # 创建几何点并检查是否在边界内
                    geometries = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
                    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

                    # 过滤边界内的点
                    within_boundary = gdf.within(base_map['geometry'])
                    df = df[within_boundary].copy().reset_index(drop=True)

                    st.info(f"🗺️ 过滤后：{len(df)} 个点在奉节县边界内")

                # 然后设置有效点位 - 使用新的连续字段
                df["valid"] = (
                        (df["predicted_wind_speed"] >= 5.0) &  # 降低风速要求
                        (df["slope"] <= 35) &  # 坡度约束
                        (df["elevation"] >= 150) & (df["elevation"] <= 1600)  # 海拔约束
                )

                st.session_state["windfarm_data"] = df
                st.session_state.last_uploaded_file = uploaded_file.name
                st.success("✅ 数据加载成功")

                # 显示数据质量检查
                check_data_quality(df)

                # 立即重定向到风能分布页面
                st.session_state.current_page = "wind"
                st.rerun()
        else:
            # 如果文件被删除，清除相关状态
            if 'last_uploaded_file' in st.session_state:
                del st.session_state.last_uploaded_file
            if 'windfarm_data' in st.session_state:
                del st.session_state.windfarm_data
            if 'optimization_result' in st.session_state:
                del st.session_state.optimization_result

        # 优化按钮
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
        if "windfarm_data" in st.session_state:
            # 数据质量警告
            df = st.session_state["windfarm_data"]
            if "predicted_wind_speed" in df.columns and df["predicted_wind_speed"].std() < 0.5:
                st.warning("⚠️ 风速数据变化较小，可能影响优化效果")

            # 显示有效点位信息
            valid_count = df['valid'].sum() if 'valid' in df.columns else 0
            if valid_count < algorithm_params['n_turbines']:
                st.error(f"❌ 有效点位数量({valid_count})少于目标风机数量({algorithm_params['n_turbines']})")
                st.info("💡 建议：减少风机数量或检查数据约束条件")
            else:
                st.success(f"✅ 有效点位数量({valid_count})满足目标风机数量({algorithm_params['n_turbines']})")

            if st.button("🚀 开始优化计算", use_container_width=True, type="primary"):
                with st.spinner("正在计算最优布局..."):
                    try:
                        # 使用真实优化函数调用
                        result = call_optimize_function(df, algo, algorithm_params)
                        st.session_state["optimization_result"] = result
                        st.success("🎯 优化完成")
                        st.session_state.current_page = "result"
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 优化计算失败: {str(e)}")
                        st.info("💡 建议：尝试使用基础参数或检查数据格式")
        else:
            st.button("🚀 开始优化计算", use_container_width=True, disabled=True)

    # ========== 优化结果详情展示在页面下端 ==========
    if st.session_state.current_page == "result" and "optimization_result" in st.session_state:
        st.markdown("---")
        st.markdown("#### 📊 优化结果分析")

        result = st.session_state["optimization_result"]
        df = st.session_state["windfarm_data"]

        # 直接调用 display_optimization_result，其中已经包含了收敛图
        display_optimization_result(result, df)

        # ========== 储能调度分析 ==========
        st.markdown("---")
        st.markdown("#### 🔋 储能调度策略分析")

        # 从优化结果中提取风机位置数据
        show_storage_analysis = False
        metrics = None

        # 方法1: 尝试从优化结果中获取风机位置
        best_locations = result.get('best_locations', [])

        # 方法2: 如果best_locations不存在，尝试其他可能的键
        if not best_locations:
            possible_keys = ['solution', 'best_solution', 'selected_indices', 'positions']
            for key in possible_keys:
                if key in result and result[key]:
                    best_locations = result[key]
                    break

        # 方法3: 如果还是没有找到，从display_optimization_result中推断
        if not best_locations and 'best_positions_data' in result:
            # 使用优化算法返回的真实最优位置数据
            all_turbines = result['best_positions_data']
            if not all_turbines.empty:
                best_locations = all_turbines.index.tolist()

        st.info(f"🔍 找到 {len(best_locations)} 个最优风机位置")

        if best_locations and len(best_locations) > 0:
            # 使用最优风机的真实数据
            try:
                # 获取最优风机对应的数据
                optimal_turbines = df.loc[best_locations[:n_turbines]]

                if not optimal_turbines.empty and "predicted_wind_speed" in optimal_turbines.columns:
                    # 模拟24小时风速数据（基于真实风机位置的风速）
                    time_hours = 24
                    hours = list(range(time_hours))

                    # 使用真实风机位置的风速数据创建波动序列
                    base_wind_speeds = optimal_turbines['predicted_wind_speed'].values

                    # 为每个风机创建24小时的风速序列
                    hourly_wind_speeds_all = []
                    for base_speed in base_wind_speeds:
                        # 基于基础风速创建有波动性的序列
                        np.random.seed(42)  # 固定随机种子以便重现
                        hourly_variation = base_speed + np.random.normal(0, 1.5, time_hours)
                        hourly_variation = np.clip(hourly_variation, 3, 25)
                        hourly_wind_speeds_all.append(hourly_variation)

                    # 计算每个风机的发电功率
                    turbine_capacity = 2000  # kW
                    wind_power_all = []
                    for hourly_speeds in hourly_wind_speeds_all:
                        turbine_power = calculate_wind_power_from_speed(hourly_speeds, turbine_capacity)
                        wind_power_all.append(turbine_power)

                    # 汇总所有风机的总功率
                    wind_power_total = np.sum(wind_power_all, axis=0)

                    # 初始化储能调度器
                    scheduler = EnergyStorageScheduler(
                        capacity_kwh=algorithm_params['storage_capacity'],
                        max_power_kw=algorithm_params['max_power']
                    )

                    # 应用调度策略
                    if strategy == "出力平滑":
                        smoothed_power, battery_soc, charge_discharge = scheduler.smoothing_strategy(wind_power_total)
                        delivered_power = smoothed_power
                        curtailed_power = np.maximum(wind_power_total - smoothed_power, 0)

                    elif strategy == "弃风消减":
                        grid_capacity = np.percentile(wind_power_total, 70)  # 假设电网接收容量为70%分位数
                        delivered_power, curtailed_power, battery_soc, charge_discharge = \
                            scheduler.curtailment_reduction_strategy(wind_power_total, grid_capacity)
                    else:  # 混合策略
                        # 先平滑，再考虑弃风
                        smoothed_power, battery_soc, charge_discharge = scheduler.smoothing_strategy(wind_power_total)
                        grid_capacity = np.percentile(smoothed_power, 80)
                        delivered_power, curtailed_power, _, _ = \
                            scheduler.curtailment_reduction_strategy(smoothed_power, grid_capacity)

                    # 计算指标
                    metrics = calculate_metrics(wind_power_total, delivered_power, curtailed_power)
                    show_storage_analysis = True

                    # 显示储能调度分析结果
                    if show_storage_analysis:
                        # 显示所有风机汇总信息
                        st.markdown("#### 📋 风机列表")
                        turbine_info = optimal_turbines[
                            ['lat', 'lon', 'predicted_wind_speed', 'elevation', 'slope']].copy()
                        turbine_info['风机编号'] = [f'T{i + 1}' for i in range(len(turbine_info))]
                        turbine_info['平均功率(kW)'] = [np.mean(power) for power in wind_power_all]
                        turbine_info['最大功率(kW)'] = [np.max(power) for power in wind_power_all]
                        turbine_info['可消纳电量(MWh)'] = [np.sum(power) / 1000 for power in wind_power_all]

                        # 重新排列列顺序
                        display_columns = ['风机编号', 'lat', 'lon', 'predicted_wind_speed', '平均功率(kW)',
                                           '最大功率(kW)', '可消纳电量(MWh)', 'elevation', 'slope']
                        display_columns = [col for col in display_columns if col in turbine_info.columns]
                        turbine_info = turbine_info[display_columns]

                        st.dataframe(turbine_info, use_container_width=True)

                        # 风机选择器
                        st.markdown("#### 🔍 选择要查看的风机")

                        # 创建风机选择下拉菜单
                        turbine_options = [
                            f"T{i + 1} (经度: {optimal_turbines.iloc[i]['lon']:.4f}, 纬度: {optimal_turbines.iloc[i]['lat']:.4f})"
                            for i in range(len(optimal_turbines))]

                        selected_turbine = st.selectbox(
                            "选择风机查看详细储能调度评估",
                            options=turbine_options,
                            index=0,
                            help="选择要查看详细储能调度分析的风机"
                        )

                        # 获取选中的风机索引
                        selected_index = turbine_options.index(selected_turbine)

                        # 显示选中的风机详细评估
                        st.markdown(f"---")
                        st.markdown(f"### 🌬️ 风机 T{selected_index + 1} 储能调度详细评估")

                        # 获取当前选中风机的数据
                        current_turbine_power = wind_power_all[selected_index]

                        # 为单个风机创建储能调度（使用总储能系统的一部分）
                        individual_storage_capacity = algorithm_params['storage_capacity'] / len(optimal_turbines)
                        individual_max_power = algorithm_params['max_power'] / len(optimal_turbines)

                        individual_scheduler = EnergyStorageScheduler(
                            capacity_kwh=individual_storage_capacity,
                            max_power_kw=individual_max_power
                        )

                        # 对单个风机应用调度策略
                        if strategy == "出力平滑":
                            individual_smoothed, individual_soc, individual_charge = individual_scheduler.smoothing_strategy(
                                current_turbine_power)
                            individual_delivered = individual_smoothed
                            individual_curtailed = np.maximum(current_turbine_power - individual_smoothed, 0)

                        elif strategy == "弃风消减":
                            individual_grid_capacity = np.percentile(current_turbine_power, 70)
                            individual_delivered, individual_curtailed, individual_soc, individual_charge = \
                                individual_scheduler.curtailment_reduction_strategy(current_turbine_power,
                                                                                    individual_grid_capacity)
                        else:  # 混合策略
                            individual_smoothed, individual_soc, individual_charge = individual_scheduler.smoothing_strategy(
                                current_turbine_power)
                            individual_grid_capacity = np.percentile(individual_smoothed, 80)
                            individual_delivered, individual_curtailed, _, _ = \
                                individual_scheduler.curtailment_reduction_strategy(individual_smoothed,
                                                                                    individual_grid_capacity)

                        # 计算单个风机的指标
                        individual_metrics = calculate_metrics(current_turbine_power, individual_delivered,
                                                               individual_curtailed)

                        # 显示选中风机的详细信息
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("经度", f"{optimal_turbines.iloc[selected_index]['lon']:.4f}")
                        with col2:
                            st.metric("纬度", f"{optimal_turbines.iloc[selected_index]['lat']:.4f}")
                        with col3:
                            st.metric("基础风速",
                                      f"{optimal_turbines.iloc[selected_index]['predicted_wind_speed']:.1f} m/s")
                        with col4:
                            st.metric("分配储能", f"{individual_storage_capacity / 1000:.1f} MWh")

                        # 显示单个风机评估
                        create_single_turbine_assessment(
                            current_turbine_power,
                            individual_delivered,
                            individual_curtailed,
                            individual_soc,
                            hours
                        )

                else:
                    st.warning("⚠️ 最优风机数据中缺少风速信息，无法进行储能调度分析")

            except Exception as e:
                st.error(f"❌ 储能调度分析失败: {str(e)}")
                st.info("💡 建议：检查数据格式或减少风机数量")
        else:
            st.warning("⚠️ 未找到有效的风机位置数据，无法进行储能调度分析")

        # 调试信息
        with st.expander("🔍 调试信息"):
            debug_info = {
                "算法参数": {k: v for k, v in algorithm_params.items() if k not in ['storage_capacity', 'max_power']},
                "储能配置": f"{storage_capacity} MWh, {max_power} MW",
                "调度策略": strategy,
                "最终适应度": result.get('best_fitness', '未知'),
                "数据点数": len(df),
                "有效点数": df['valid'].sum() if 'valid' in df.columns else '未知',
                "找到的风机位置数": len(best_locations) if 'best_locations' in locals() else 0,
                "优化模式": "真实算法计算"
            }

            if metrics is not None:
                debug_info["性能指标"] = metrics

            st.json(debug_info)

# ======================================================
# 🚀 运行 Streamlit
# ======================================================
if __name__ == "__main__":
    strategy_optimization_page()