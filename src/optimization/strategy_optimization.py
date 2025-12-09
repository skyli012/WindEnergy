import geopandas as gpd
import streamlit as st
from shapely.geometry import Point

from src.optimization.algorithm_convergence_curve import call_optimize_function_with_all_strategies, \
    calculate_wind_utilization
from src.utils.create_map import display_maale_gilboa_standalone_map, display_environment, display_optimization_map, \
    create_maale_gilboa_base_map
from src.visualization.opt_result_show import display_optimization_result, display_wind_utilization_analysis
from src.visualization.storage_schedule_display import display_storage_schedule_analysis


# ======================================================
# 🌬️ 主页面：风电场选址优化系统
# ======================================================
def strategy_optimization_page():
    # 页面标题 - 更紧凑
    st.markdown("### 🌬️ 风电场选址-储能协同决策")
    st.caption("基于真实优化算法计算 · 山地风电场布局优化")

    # 初始化 session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "map"
    if 'algorithm_comparison_results' not in st.session_state:
        st.session_state.algorithm_comparison_results = {}

    # 初始化风场数量
    if 'n_farms' not in st.session_state:
        st.session_state.n_farms = 2
    if 'n_turbines_per_farm' not in st.session_state:
        st.session_state.n_turbines_per_farm = 4

    # 🔥 新增：初始化储能决策变量
    if 'storage_capacity_mwh' not in st.session_state:
        st.session_state.storage_capacity_mwh = 60  # 默认60 MWh
    if 'storage_power_mw' not in st.session_state:
        st.session_state.storage_power_mw = 30  # 默认30 MW

    # ========== 地图在左边，控制面板在右边 ==========
    map_col, control_col = st.columns([2, 1])

    with map_col:
        # 显示地图内容
        if st.session_state.current_page == "map":
            display_maale_gilboa_standalone_map()
            if "windfarm_data" not in st.session_state:
                # 检查是否有从数据导入页面导入的数据
                if 'dataset' in st.session_state:
                    # 自动处理数据
                    process_imported_data()
                else:
                    st.info("📁 请先在数据导入页面导入风速预测数据")

        elif st.session_state.current_page == "wind":
            if "windfarm_data" in st.session_state:
                display_environment(st.session_state["windfarm_data"])
                if "optimization_result" not in st.session_state:
                    st.info("⚙️ 数据已就绪，可点击'开始优化'进行风电场布局优化")
            else:
                st.warning("⚠️ 请先在数据导入页面导入数据")
                st.session_state.current_page = "map"
                st.rerun()

        elif st.session_state.current_page == "result":
            if "windfarm_data" in st.session_state and "optimization_result" in st.session_state:
                # 在左侧地图上显示优化结果（多个风电场位置）
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

        # 基础参数设置 - 增加风场数量选择
        st.markdown("**🎯 基础参数设置**")
        col1, col2 = st.columns(2)
        with col1:
            # 修改这里：使用 st.session_state 来保存和读取风场数量
            n_farms = st.slider("风场数量", 1, 5, st.session_state.n_farms, help="选择要建设的风电场数量")
            # 保存到 session state
            st.session_state.n_farms = n_farms

        with col2:
            # 同样修改单场风机数
            n_turbines = st.slider("单场风机数", 1, 10, st.session_state.n_turbines_per_farm,
                                   help="每个风电场安装的风机数量")
            st.session_state.n_turbines_per_farm = n_turbines

        # 计算总风机数量
        total_turbines = n_farms * n_turbines

        # 🔥 新增：储能决策变量设置
        st.markdown("**🔋 储能系统配置**")
        col_storage1, col_storage2 = st.columns(2)
        with col_storage1:
            # 储能容量选择 (MWh)
            storage_capacity_mwh = st.slider(
                "储能容量 (MWh)",
                10, 200, st.session_state.storage_capacity_mwh, 10,
                help="储能系统总能量容量，影响存储时间和削峰填谷能力"
            )
            st.session_state.storage_capacity_mwh = storage_capacity_mwh

        with col_storage2:
            # 储能功率选择 (MW)
            storage_power_mw = st.slider(
                "储能功率 (MW)",
                5, 100, st.session_state.storage_power_mw, 5,
                help="储能系统最大充放电功率，影响调节速度和能力"
            )
            st.session_state.storage_power_mw = storage_power_mw

        # 根据风场数量设置合理的固定间距
        if n_farms == 1:
            min_farm_distance = 0  # 单个风场不需要间距约束
        elif n_farms == 2:
            min_farm_distance = 3.0  # 2个风场，3km间距
        elif n_farms == 3:
            min_farm_distance = 2.5  # 3个风场，2.5km间距
        elif n_farms == 4:
            min_farm_distance = 2.0  # 4个风场，2km间距
        else:  # n_farms == 5
            min_farm_distance = 1.5  # 5个风场，1.5km间距

        # 风机参数
        TURBINE_DIAMETER = 140  # 米（金风科技 GW-140/2500 风机直径）

        # 设置合理的固定间距值
        DOWNWIND_DISTANCE_RATIO = 8.0  # 主风向间距 8倍D
        CROSSWIND_DISTANCE_RATIO = 4.0  # 侧向间距 4倍D

        # 计算实际间距
        min_downwind_distance = DOWNWIND_DISTANCE_RATIO * TURBINE_DIAMETER  # 米
        min_crosswind_distance = CROSSWIND_DISTANCE_RATIO * TURBINE_DIAMETER  # 米

        # 新增：风能利用率权重设置
        st.markdown("**📈 优化目标权重**")
        col_weight1, col_weight2 = st.columns(2)
        with col_weight1:
            wind_speed_weight = st.slider("风速权重", 0.1, 1.0, 0.6, 0.1,
                                          help="风速在综合评分中的权重")
        with col_weight2:
            utilization_weight = st.slider("利用率权重", 0.1, 1.0, 0.4, 0.1,
                                           help="风能利用率在综合评分中的权重")

        # 固定约束条件值 - 使用合理的固定风场间距和风机间距
        algorithm_params = {
            'n_farms': n_farms,
            'n_turbines_per_farm': n_turbines,
            'total_turbines': total_turbines,
            'max_slope': 35,
            'max_road_distance': 100,
            'min_residential_distance': 60,
            'min_heritage_distance': 70,
            'min_geology_distance': 80,
            'min_water_distance': 100,
            'min_farm_distance': min_farm_distance * 1000,  # 转换为米
            'min_downwind_distance': min_downwind_distance,  # 主风向间距
            'min_crosswind_distance': min_crosswind_distance,  # 侧向间距
            'turbine_diameter': TURBINE_DIAMETER,  # 风机直径
            'wind_speed_weight': wind_speed_weight,
            'utilization_weight': utilization_weight,
            # 🔥 新增：储能决策变量（传递给优化算法）
            'storage_capacity': storage_capacity_mwh * 1000,  # 转换为kWh
            'storage_power': storage_power_mw * 1000,  # 转换为kW
            'storage_capacity_mwh': storage_capacity_mwh,
            'storage_power_mw': storage_power_mw,
        }

        # 算法选择单独一行
        algo = st.selectbox("优化算法",
                            ["遗传算法", "模拟退火算法", "粒子群优化算法", "PuLP优化求解器"],
                            help="选择优化算法")

        # 算法高级参数（可选）
        st.markdown("**🔧 算法高级参数（可选）**")
        with st.expander("算法高级参数设置", expanded=False):
            if algo == "遗传算法":
                col11, col12 = st.columns(2)
                with col11:
                    # 根据问题复杂度调整种群大小
                    base_pop_size = 50
                    pop_size_multiplier = n_farms * 2
                    recommended_pop = base_pop_size + pop_size_multiplier * 10
                    algorithm_params['pop_size'] = st.slider("种群大小", 20, 300, recommended_pop,
                                                             help=f"推荐值: {recommended_pop} (适应{n_farms}个风场)")
                with col12:
                    algorithm_params['generations'] = st.slider("迭代代数", 50, 500, 100 + n_farms * 20,
                                                                help="多风场问题需要更多迭代")

                col13, col14 = st.columns(2)
                with col13:
                    algorithm_params['mutation_rate'] = st.slider("变异率", 0.01, 0.3, 0.1, 0.01)
                with col14:
                    algorithm_params['crossover_rate'] = st.slider("交叉率", 0.5, 1.0, 0.8, 0.05)

            elif algo == "模拟退火算法":
                col15, col16, col17 = st.columns(3)
                with col15:
                    algorithm_params['initial_temp'] = st.slider("初始温度", 100, 5000, 1000 + n_farms * 200, 100)
                with col16:
                    algorithm_params['cooling_rate'] = st.slider("降温速率", 0.85, 0.99, 0.95, 0.01)
                with col17:
                    algorithm_params['iterations_per_temp'] = st.slider("每温度迭代次数", 10, 200, 50 + n_farms * 10)

            elif algo == "粒子群优化算法":
                col18, col19 = st.columns(2)
                with col18:
                    base_particles = 30
                    recommended_particles = base_particles + n_farms * 5
                    algorithm_params['pop_size'] = st.slider("粒子数量", 20, 150, recommended_particles,
                                                             help=f"推荐值: {recommended_particles}")
                with col19:
                    algorithm_params['generations'] = st.slider("迭代次数", 50, 500, 100 + n_farms * 25)

                col20, col21, col22 = st.columns(3)
                with col20:
                    algorithm_params['w'] = st.slider("惯性权重", 0.1, 1.0, 0.7, 0.1)
                with col21:
                    algorithm_params['c1'] = st.slider("个体学习因子", 0.1, 2.0, 1.5, 0.1)
                with col22:
                    algorithm_params['c2'] = st.slider("社会学习因子", 0.1, 2.0, 1.5, 0.1)

            elif algo == "PuLP优化求解器":
                col23, col24 = st.columns(2)
                with col23:
                    algorithm_params['solver_type'] = st.selectbox(
                        "求解器类型",
                        ["CBC", "GLPK", "CPLEX"],
                        help="选择线性规划求解器"
                    )
                with col24:
                    base_time = 60
                    recommended_time = base_time + n_farms * 30
                    algorithm_params['time_limit'] = st.slider("时间限制(秒)", 10, 600, recommended_time,
                                                               help=f"推荐值: {recommended_time}秒")

        # 数据状态显示
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

        # 检查数据状态
        if 'dataset' in st.session_state and "windfarm_data" not in st.session_state:
            st.info("📊 检测到已导入数据，点击按钮处理数据")
            if st.button("🔄 处理导入的数据", use_container_width=True):
                process_imported_data()
                st.rerun()

        elif "windfarm_data" in st.session_state:
            # st.success("✅ 数据已就绪")
            df = st.session_state["windfarm_data"]

            # 显示数据基本信息
            valid_count = df['valid'].sum() if 'valid' in df.columns else 0

            # 🔥 新增：显示风电和储能配置信息
            total_capacity_mw = total_turbines * 2.5  # 2.5MW每台风机
            storage_ratio = storage_power_mw / total_capacity_mw if total_capacity_mw > 0 else 0
            st.info(
                f"📊 配置信息: {total_turbines}台风机 ({total_capacity_mw:.1f} MW) + {storage_capacity_mwh} MWh储能 ({storage_power_mw} MW)")

        else:
            st.info("📁 请在数据导入页面先上传数据文件")

        # 优化按钮
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
        if "windfarm_data" in st.session_state:
            # 数据质量警告
            df = st.session_state["windfarm_data"]
            if "predicted_wind_speed" in df.columns and df["predicted_wind_speed"].std() < 0.5:
                st.warning("⚠️ 风速数据变化较小，可能影响优化效果")

            # 多算法对比选项
            st.markdown("**📊 算法对比选项**")
            compare_algorithms = st.checkbox("运行多算法对比",
                                             help="同时运行遗传算法、模拟退火、粒子群优化进行对比分析（不包含PuLP求解器）")

            if st.button("🚀 开始优化计算", use_container_width=True, type="primary"):
                with st.spinner(f"正在计算{n_farms}个风电场 + {storage_capacity_mwh}MWh储能的最优布局..."):
                    try:
                        if compare_algorithms:
                            # 运行多算法对比（不包含PuLP）
                            run_algorithm_comparison(df, algorithm_params, n_farms)
                        else:
                            # 单个算法优化
                            result = call_optimize_function_with_all_strategies(df, algo, algorithm_params)
                            st.session_state["optimization_result"] = result
                            st.success(f"🎯 {n_farms}个风电场 + {storage_capacity_mwh}MWh储能优化完成")
                            st.session_state.current_page = "result"
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ 优化计算失败: {str(e)}")
                        st.info("💡 建议：尝试减少风场数量或使用更宽松的约束条件")
        else:
            st.button("🚀 开始优化计算", use_container_width=True, disabled=True)

    # 在优化结果显示部分确保正确调用
    if st.session_state.current_page == "result" and "optimization_result" in st.session_state:
        st.markdown("---")
        st.markdown("#### 📊 多风场优化结果分析")

        result = st.session_state["optimization_result"]
        df = st.session_state["windfarm_data"]

        # 显示多风场特定的分析结果
        display_optimization_result(result, df)

        # 显示风能利用率分析
        display_wind_utilization_analysis(result, df)

        # 🔥 新增：显示储能配置结果
        display_storage_configuration(result)

        # 显示算法对比结果（如果有）
        if st.session_state.algorithm_comparison_results:
            display_algorithm_comparison()


def display_storage_configuration(result):
    """显示储能配置结果"""
    if 'storage_params' in result:
        storage_params = result['storage_params']
        st.markdown("---")
        st.markdown("#### 🔋 储能配置详情")

        col1, col2 = st.columns(2)
        with col1:
            capacity_mwh = storage_params.get('storage_capacity_mwh',
                                              storage_params.get('storage_capacity', 60000) / 1000)
            st.metric("储能容量", f"{capacity_mwh:.1f} MWh")
        with col2:
            power_mw = storage_params.get('storage_power_mw',
                                          storage_params.get('storage_power', 30000) / 1000)
            st.metric("储能功率", f"{power_mw:.1f} MW")

        # 计算储能时长
        if power_mw > 0:
            storage_hours = capacity_mwh / power_mw
            st.info(f"🔋 储能时长: {storage_hours:.1f} 小时")


def run_algorithm_comparison(df, algorithm_params, n_farms):
    """运行多算法对比分析 - 不包含PuLP优化求解器"""
    # 只包含元启发式算法，不包含数学规划求解器
    algorithms = ["遗传算法", "模拟退火算法", "粒子群优化算法"]
    comparison_results = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, algo in enumerate(algorithms):
        status_text.text(f"正在运行 {algo}...")

        try:
            # 复制参数以避免冲突
            current_params = algorithm_params.copy()

            # 运行优化
            result = call_optimize_function_with_all_strategies(df, algo, current_params)

            # 存储结果
            comparison_results[algo] = {
                'result': result,
                'fitness': result.get('best_fitness', 0),
                'computation_time': result.get('computation_time', 0),
                'algorithm': algo,
                'n_farms': n_farms
            }

            st.success(f"✅ {algo} 完成")

        except Exception as e:
            st.error(f"❌ {algo} 运行失败: {str(e)}")
            comparison_results[algo] = {
                'result': None,
                'fitness': 0,
                'computation_time': 0,
                'algorithm': algo,
                'error': str(e)
            }

        progress_bar.progress((i + 1) / len(algorithms))

    progress_bar.empty()
    status_text.empty()

    # 保存对比结果
    st.session_state.algorithm_comparison_results = comparison_results

    # 选择最佳结果作为主要显示结果
    best_algo = None
    best_fitness = -1
    for algo, data in comparison_results.items():
        if data['result'] and data['fitness'] > best_fitness:
            best_fitness = data['fitness']
            best_algo = algo

    if best_algo:
        st.session_state["optimization_result"] = comparison_results[best_algo]['result']
        st.success(f"🏆 最佳算法: {best_algo} (适应度: {best_fitness:.3f})")

        # 显示算法对比说明
        st.info("""
        💡 **算法对比说明**：
        - 对比包含：遗传算法、模拟退火、粒子群优化
        - 未包含PuLP求解器（数学规划方法，适用场景不同）
        - 如需使用PuLP求解器，请单独选择运行
        """)

        st.session_state.current_page = "result"
        st.rerun()


def display_algorithm_comparison():
    """显示算法对比分析结果 - 使用下拉框展示"""
    # 在函数内部导入 pandas
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    # 使用下拉框来展开/收起对比分析内容
    with st.expander("📈 优化算法对比分析", expanded=False):
        # 添加对比范围说明
        st.info("""
        🔍 **对比范围说明**：
        - ✅ 包含：遗传算法、模拟退火算法、粒子群优化算法
        - ⚠️ 未包含：PuLP优化求解器（数学规划方法，适用场景不同）
        - 📊 对比指标：适应度得分、计算时间、发电性能、解的质量
        """)

        comparison_results = st.session_state.algorithm_comparison_results

        # 如果没有对比结果，显示提示信息
        if not comparison_results:
            st.warning("暂无算法对比结果，请先运行多算法对比分析")
            return

        # 创建综合对比表格
        comparison_data = []
        for algo, data in comparison_results.items():
            if data['result']:
                power_results = data['result'].get('power_results', {})
                layout_metrics = data['result'].get('layout_metrics', {})

                # 安全获取数据，避免KeyError，并提供合理的默认值
                total_annual_generation_gwh = power_results.get('total_annual_generation_gwh', 0)
                average_capacity_factor = power_results.get('average_capacity_factor', 0)

                # 关键修复：直接从优化结果中获取平均风速
                if 'average_wind_speed' in power_results:
                    average_wind_speed = power_results['average_wind_speed']
                else:
                    # 从优化结果的风机数据中计算平均风速
                    best_positions_data = data['result'].get('best_positions_data', pd.DataFrame())
                    if not best_positions_data.empty and 'predicted_wind_speed' in best_positions_data.columns:
                        average_wind_speed = best_positions_data['predicted_wind_speed'].mean()
                    else:
                        # 备用方案：从原始数据中获取
                        df = st.session_state.get("windfarm_data")
                        if df is not None and "predicted_wind_speed" in df.columns:
                            if 'best_positions' in data['result']:
                                positions = data['result']['best_positions']
                                wind_speeds = []
                                for pos_idx in positions:
                                    if pos_idx in df.index:
                                        wind_speeds.append(df.loc[pos_idx, 'predicted_wind_speed'])
                                average_wind_speed = sum(wind_speeds) / len(wind_speeds) if wind_speeds else 0
                            else:
                                average_wind_speed = df['predicted_wind_speed'].mean()
                        else:
                            average_wind_speed = 0

                # 获取风能利用率
                wind_utilization_rate = power_results.get('wind_utilization_rate', 0)
                if wind_utilization_rate == 0:
                    # 从优化结果中计算风能利用率
                    best_positions_data = data['result'].get('best_positions_data', pd.DataFrame())
                    if not best_positions_data.empty:
                        if 'wind_utilization_rate' in best_positions_data.columns:
                            wind_utilization_rate = best_positions_data['wind_utilization_rate'].mean()
                        else:
                            # 计算综合风能利用率
                            utilization_rates = []
                            for _, turbine in best_positions_data.iterrows():
                                wind_speed = turbine.get('predicted_wind_speed', 0)
                                utilization = calculate_wind_utilization(pd.Series([wind_speed]))
                                utilization_rates.append(utilization)
                            wind_utilization_rate = sum(utilization_rates) / len(
                                utilization_rates) if utilization_rates else 0

                # 关键修复：布局效率计算
                if 'layout_efficiency' in layout_metrics:
                    layout_efficiency = layout_metrics['layout_efficiency']
                else:
                    # 基于适应度得分和约束满足程度计算布局效率
                    fitness = data['fitness']
                    constraints_violated = data['result'].get('constraints_violated', {})

                    # 计算约束满足度
                    total_violations = sum(constraints_violated.values())
                    constraint_satisfaction = max(0, 1 - total_violations / 10)  # 假设最多10个约束

                    # 计算布局效率
                    max_possible_fitness = 1000  # 根据您的适应度函数调整
                    fitness_efficiency = min(fitness / max_possible_fitness, 1.0) if max_possible_fitness > 0 else 0

                    # 综合布局效率
                    layout_efficiency = 0.7 * fitness_efficiency + 0.3 * constraint_satisfaction

                comparison_data.append({
                    '算法': algo,
                    '适应度得分': f"{data['fitness']:.4f}",
                    '计算时间(秒)': f"{data['computation_time']:.2f}",
                    '年发电量(GWh)': f"{total_annual_generation_gwh:.2f}",
                    '容量因数(%)': f"{average_capacity_factor * 100:.1f}",
                    '平均风速(m/s)': f"{average_wind_speed:.2f}",
                    '风能利用率(%)': f"{wind_utilization_rate * 100:.1f}",
                    '布局效率': f"{layout_efficiency:.3f}",
                    '状态': '✅ 成功'
                })
            else:
                comparison_data.append({
                    '算法': algo,
                    '适应度得分': '0.0000',
                    '计算时间(秒)': '0.00',
                    '年发电量(GWh)': 'N/A',
                    '容量因数(%)': 'N/A',
                    '平均风速(m/s)': 'N/A',
                    '风能利用率(%)': 'N/A',
                    '布局效率': 'N/A',
                    '状态': f'❌ 失败: {data.get("error", "未知错误")}'
                })

        # 显示综合对比表格
        comparison_df = pd.DataFrame(comparison_data)
        st.markdown("**📋 综合性能对比表**")
        st.dataframe(comparison_df, use_container_width=True)

        # ========== 增强的可视化部分 ==========
        st.markdown("---")
        st.markdown("#### 📊 多维度性能分析")

        # 准备可视化数据
        algorithms = []
        fitness_scores = []
        computation_times = []
        annual_generation = []
        capacity_factors = []
        avg_wind_speeds = []
        wind_utilization = []
        layout_efficiency_list = []

        for algo, data in comparison_results.items():
            if data['result']:
                algorithms.append(algo)
                fitness_scores.append(data['fitness'])
                computation_times.append(data['computation_time'])

                power_results = data['result'].get('power_results', {})
                layout_metrics = data['result'].get('layout_metrics', {})

                # 安全获取数据，提供默认值
                annual_generation.append(power_results.get('total_annual_generation_gwh', 0))
                capacity_factors.append(power_results.get('average_capacity_factor', 0) * 100)

                # 使用表格中相同的逻辑获取平均风速
                if 'average_wind_speed' in power_results:
                    avg_wind_speeds.append(power_results['average_wind_speed'])
                else:
                    best_positions_data = data['result'].get('best_positions_data', pd.DataFrame())
                    if not best_positions_data.empty and 'predicted_wind_speed' in best_positions_data.columns:
                        avg_wind_speed = best_positions_data['predicted_wind_speed'].mean()
                    else:
                        df = st.session_state.get("windfarm_data")
                        if df is not None and "predicted_wind_speed" in df.columns:
                            if 'best_positions' in data['result']:
                                positions = data['result']['best_positions']
                                wind_speeds = []
                                for pos_idx in positions:
                                    if pos_idx in df.index:
                                        wind_speeds.append(df.loc[pos_idx, 'predicted_wind_speed'])
                                avg_wind_speed = sum(wind_speeds) / len(wind_speeds) if wind_speeds else 0
                            else:
                                avg_wind_speed = df['predicted_wind_speed'].mean()
                        else:
                            avg_wind_speed = 0
                    avg_wind_speeds.append(avg_wind_speed)

                # 获取风能利用率
                if 'wind_utilization_rate' in power_results:
                    wind_utilization.append(power_results['wind_utilization_rate'] * 100)
                else:
                    best_positions_data = data['result'].get('best_positions_data', pd.DataFrame())
                    if not best_positions_data.empty:
                        if 'wind_utilization_rate' in best_positions_data.columns:
                            utilization_rate = best_positions_data['wind_utilization_rate'].mean()
                        else:
                            utilization_rates = []
                            for _, turbine in best_positions_data.iterrows():
                                wind_speed = turbine.get('predicted_wind_speed', 0)
                                utilization = calculate_wind_utilization(pd.Series([wind_speed]))
                                utilization_rates.append(utilization)
                            utilization_rate = sum(utilization_rates) / len(utilization_rates) if utilization_rates else 0
                        wind_utilization.append(utilization_rate * 100)
                    else:
                        wind_utilization.append(0)

                # 使用表格中相同的逻辑获取布局效率
                if 'layout_efficiency' in layout_metrics:
                    layout_efficiency_list.append(layout_metrics['layout_efficiency'])
                else:
                    fitness = data['fitness']
                    constraints_violated = data['result'].get('constraints_violated', {})
                    total_violations = sum(constraints_violated.values())
                    constraint_satisfaction = max(0, 1 - total_violations / 10)
                    max_possible_fitness = 1000
                    fitness_efficiency = min(fitness / max_possible_fitness, 1.0) if max_possible_fitness > 0 else 0
                    layout_efficiency = 0.7 * fitness_efficiency + 0.3 * constraint_satisfaction
                    layout_efficiency_list.append(layout_efficiency)

        # 继续原有的可视化代码...
        if algorithms:
            # ========== 雷达图多维度对比 ==========
            st.markdown("**🎯 多维度性能雷达图**")

            # 归一化数据用于雷达图（0-1范围）
            def normalize_data(data):
                if not data or max(data) == min(data):
                    return [0.5] * len(data)  # 如果全部相同返回中间值
                return [(x - min(data)) / (max(data) - min(data)) for x in data]

            # 检查数据有效性
            valid_categories = []
            normalized_data_sets = []

            # 为每个算法准备数据
            for i, algo in enumerate(algorithms):
                algo_data = []
                category_names = []

                # 适应度（越高越好）
                if fitness_scores:
                    algo_data.append(normalize_data(fitness_scores)[i])
                    if '适应度' not in category_names:
                        category_names.append('适应度')

                # 速度（时间的倒数，越高越好）
                if computation_times:
                    speed_data = [1 / t if t > 0 else 0 for t in computation_times]
                    algo_data.append(normalize_data(speed_data)[i] if speed_data else 0.5)
                    if '速度' not in category_names:
                        category_names.append('速度')

                # 发电量（越高越好）
                if annual_generation and any(annual_generation):
                    algo_data.append(normalize_data(annual_generation)[i])
                    if '发电量' not in category_names:
                        category_names.append('发电量')

                # 效率（越高越好）
                if layout_efficiency_list and any(layout_efficiency_list):
                    algo_data.append(normalize_data(layout_efficiency_list)[i])
                    if '效率' not in category_names:
                        category_names.append('效率')

                # 利用率（越高越好）
                if wind_utilization and any(wind_utilization):
                    algo_data.append(normalize_data(wind_utilization)[i])
                    if '利用率' not in category_names:
                        category_names.append('利用率')

                # 闭合雷达图
                if algo_data:
                    algo_data.append(algo_data[0])
                    normalized_data_sets.append(algo_data)

            if normalized_data_sets and category_names:
                # 闭合类别名称
                radar_categories = category_names + [category_names[0]]

                # 创建雷达图
                fig_radar = go.Figure()

                for i, algo in enumerate(algorithms):
                    if i < len(normalized_data_sets):
                        fig_radar.add_trace(go.Scatterpolar(
                            r=normalized_data_sets[i],
                            theta=radar_categories,
                            fill='toself',
                            name=algo,
                            line=dict(width=2)
                        ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="多维度算法性能对比",
                    height=500
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.warning("⚠️ 数据不足生成雷达图")

            # 继续其他可视化代码...
            # ========== 性能指标仪表盘 ==========
            st.markdown("**📈 性能指标仪表盘**")

            # 创建综合对比子图
            fig_metrics = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    '适应度得分对比',
                    '计算时间(秒)',
                    '年发电量(GWh)',
                    '容量因数(%)',
                    '平均风速(m/s)',
                    '布局效率'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.08
            )

            # 定义不同算法的颜色
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(algorithms)]

            # 第一行，第一列：适应度得分
            if fitness_scores:
                fig_metrics.add_trace(
                    go.Bar(name='适应度', x=algorithms, y=fitness_scores,
                           marker_color=colors, text=[f'{score:.4f}' for score in fitness_scores],
                           textposition='auto'),
                    row=1, col=1
                )

            # 第一行，第二列：计算时间
            if computation_times:
                fig_metrics.add_trace(
                    go.Bar(name='时间', x=algorithms, y=computation_times,
                           marker_color=colors, text=[f'{time:.1f}s' for time in computation_times],
                           textposition='auto'),
                    row=1, col=2
                )

            # 第一行，第三列：年发电量
            if annual_generation and any(annual_generation):
                fig_metrics.add_trace(
                    go.Bar(name='发电量', x=algorithms, y=annual_generation,
                           marker_color=colors, text=[f'{gen:.1f}' for gen in annual_generation],
                           textposition='auto'),
                    row=1, col=3
                )

            # 第二行，第一列：容量因数
            if capacity_factors and any(capacity_factors):
                fig_metrics.add_trace(
                    go.Bar(name='容量', x=algorithms, y=capacity_factors,
                           marker_color=colors, text=[f'{cf:.1f}%' for cf in capacity_factors],
                           textposition='auto'),
                    row=2, col=1
                )

            # 第二行，第二列：平均风速
            if avg_wind_speeds and any(avg_wind_speeds):
                fig_metrics.add_trace(
                    go.Bar(name='风速', x=algorithms, y=avg_wind_speeds,
                           marker_color=colors, text=[f'{ws:.2f}' for ws in avg_wind_speeds],
                           textposition='auto'),
                    row=2, col=2
                )

            # 第二行，第三列：布局效率
            if layout_efficiency_list and any(layout_efficiency_list):
                fig_metrics.add_trace(
                    go.Bar(name='效率', x=algorithms, y=layout_efficiency_list,
                           marker_color=colors, text=[f'{eff:.3f}' for eff in layout_efficiency_list],
                           textposition='auto'),
                    row=2, col=3
                )

            fig_metrics.update_layout(
                height=700,
                showlegend=False,
                title_text="综合算法性能指标"
            )

            # 更新y轴标签
            if fitness_scores:
                fig_metrics.update_yaxes(title_text="得分", row=1, col=1)
            if computation_times:
                fig_metrics.update_yaxes(title_text="秒", row=1, col=2)
            if annual_generation and any(annual_generation):
                fig_metrics.update_yaxes(title_text="GWh", row=1, col=3)
            if capacity_factors and any(capacity_factors):
                fig_metrics.update_yaxes(title_text="百分比", row=2, col=1)
            if avg_wind_speeds and any(avg_wind_speeds):
                fig_metrics.update_yaxes(title_text="m/s", row=2, col=2)
            if layout_efficiency_list and any(layout_efficiency_list):
                fig_metrics.update_yaxes(title_text="效率", row=2, col=3)

            st.plotly_chart(fig_metrics, use_container_width=True)

            # ========== 详细算法推荐 ==========
            st.markdown("---")
            st.markdown("#### 💡 详细算法推荐")

            if fitness_scores and computation_times and annual_generation:
                # 计算性能得分
                performance_scores = {}
                for i, algo in enumerate(algorithms):
                    # 归一化得分（0-1）
                    norm_fitness = (fitness_scores[i] - min(fitness_scores)) / (
                            max(fitness_scores) - min(fitness_scores)) if max(fitness_scores) != min(
                        fitness_scores) else 0.5
                    norm_speed = 1 - (computation_times[i] - min(computation_times)) / (
                            max(computation_times) - min(computation_times)) if max(computation_times) != min(
                        computation_times) else 0.5
                    norm_generation = (annual_generation[i] - min(annual_generation)) / (
                            max(annual_generation) - min(annual_generation)) if max(annual_generation) != min(
                        annual_generation) else 0.5

                    # 加权综合得分（可根据需要调整权重）
                    overall_score = 0.5 * norm_fitness + 0.3 * norm_generation + 0.2 * norm_speed
                    performance_scores[algo] = overall_score

                # 找到最佳综合算法
                best_overall_algo = max(performance_scores, key=performance_scores.get)

                # 创建推荐列
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("🏆 最佳综合", best_overall_algo,
                              f"得分: {performance_scores[best_overall_algo]:.3f}")

                with col2:
                    best_fitness_algo = algorithms[fitness_scores.index(max(fitness_scores))]
                    st.metric("🎯 最佳性能", best_fitness_algo,
                              f"适应度: {max(fitness_scores):.4f}")

                with col3:
                    fastest_algo = algorithms[computation_times.index(min(computation_times))]
                    st.metric("⚡ 最快", fastest_algo,
                              f"时间: {min(computation_times):.1f}s")

                # 详细推荐文本
                st.markdown("**📝 推荐详情:**")

                if best_overall_algo == best_fitness_algo == fastest_algo:
                    st.success(f"""
                    **强烈推荐: {best_overall_algo}**
                    - 🏆 在所有指标上表现最佳
                    - 🎯 最高的风电场布局质量适应度得分
                    - ⚡ 最快的计算时间，优化效率高
                    - 💡 适用于大多数场景的理想选择
                    """)
                elif best_overall_algo == best_fitness_algo:
                    st.info(f"""
                    **性能优先推荐: {best_overall_algo}**
                    - 🎯 提供最佳的风电场布局质量
                    - ⚖️ 在性能和速度之间取得良好平衡
                    - 💡 当布局质量是首要考虑因素时推荐使用
                    - ⏱️ 速度替代方案: {fastest_algo} ({min(computation_times):.1f}s)
                    """)
                else:
                    st.warning(f"""
                    **平衡推荐: {best_overall_algo}**
                    - ⚖️ 在考虑所有因素时提供最佳整体价值
                    - 🎯 良好性能: {fitness_scores[algorithms.index(best_overall_algo)]:.4f} 适应度
                    - ⚡ 合理速度: {computation_times[algorithms.index(best_overall_algo)]:.1f}s
                    - 💡 在同时考虑质量和效率时的最佳选择
                    """)

                # 性能改进分析
                st.markdown("**📊 性能改进分析**")
                improvement_data = []
                for algo in algorithms:
                    idx = algorithms.index(algo)
                    improvement_data.append({
                        '算法': algo,
                        '适应度vs最佳': f"{(fitness_scores[idx] / max(fitness_scores) * 100 - 100):+.1f}%" if max(
                            fitness_scores) > 0 else "N/A",
                        '时间vs最快': f"{(computation_times[idx] / min(computation_times) * 100 - 100):+.1f}%" if min(
                            computation_times) > 0 else "N/A",
                        '发电量vs最佳': f"{(annual_generation[idx] / max(annual_generation) * 100 - 100):+.1f}%" if max(
                            annual_generation) > 0 else "N/A"
                    })

                improvement_df = pd.DataFrame(improvement_data)
                st.dataframe(improvement_df, use_container_width=True)
            else:
                st.warning("数据不足生成详细推荐分析")

        else:
            st.warning("没有成功的算法结果可供对比显示。")


def process_imported_data():
    """处理从数据导入页面导入的数据"""
    if 'dataset' not in st.session_state:
        st.error("❌ 没有找到导入的数据")
        return

    df = st.session_state['dataset'].copy()

    # 添加必要的列
    if "predicted_wind_speed" in df.columns:
        df["wind_power_density"] = 0.5 * 1.225 * (df["predicted_wind_speed"] ** 3)

        # 计算风能利用率指标
        df["wind_utilization_rate"] = calculate_wind_utilization(df["predicted_wind_speed"])

        # 计算综合评分（使用默认权重）
        max_wind_speed = df["predicted_wind_speed"].max()
        max_utilization = df["wind_utilization_rate"].max()

        # 归一化处理
        df["normalized_wind_speed"] = df["predicted_wind_speed"] / max_wind_speed
        df["normalized_utilization"] = df["wind_utilization_rate"] / max_utilization

        # 综合评分（使用默认权重0.6和0.4）
        df["composite_score"] = (
                0.6 * df["normalized_wind_speed"] +
                0.4 * df["normalized_utilization"]
        )

    # 过滤边界内的点（如果有边界数据）
    base_map = create_maale_gilboa_base_map()
    if base_map:
        # 创建几何点并检查是否在边界内
        geometries = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

        # 过滤边界内的点
        within_boundary = gdf.within(base_map['geometry'])
        df = df[within_boundary].copy().reset_index(drop=True)

    # 设置有效点位
    df["valid"] = (
            (df["predicted_wind_speed"] >= 5.0) &
            (df["slope"] <= 35) &
            (df["elevation"] >= 150) & (df["elevation"] <= 1600) &
            (df["composite_score"] >= 0.4)  # 综合评分阈值
    )

    st.session_state["windfarm_data"] = df

    # 重定向到风能分布页面
    st.session_state.current_page = "wind"


# ======================================================
# 🚀 运行 Streamlit
# ======================================================
if __name__ == "__main__":
    strategy_optimization_page()