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
    st.markdown("### 🌬️ 风电场选址优化系统")
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
            st.success("✅ 数据已就绪")
            df = st.session_state["windfarm_data"]

            # 显示数据基本信息
            valid_count = df['valid'].sum() if 'valid' in df.columns else 0
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
                with st.spinner(f"正在计算{n_farms}个风电场的最优布局..."):
                    try:
                        if compare_algorithms:
                            # 运行多算法对比（不包含PuLP）
                            run_algorithm_comparison(df, algorithm_params, n_farms)
                        else:
                            # 单个算法优化
                            result = call_optimize_function_with_all_strategies(df, algo, algorithm_params)
                            st.session_state["optimization_result"] = result
                            st.success(f"🎯 {n_farms}个风电场优化完成")
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

        # 显示算法对比结果（如果有）
        if st.session_state.algorithm_comparison_results:
            display_algorithm_comparison()


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
    """显示算法对比分析结果"""
    st.markdown("---")
    st.markdown("#### 📈 优化算法对比分析")

    # 添加对比范围说明
    st.info("""
    🔍 **对比范围说明**：
    - ✅ 包含：遗传算法、模拟退火算法、粒子群优化算法
    - ⚠️ 未包含：PuLP优化求解器（数学规划方法，适用场景不同）
    - 📊 对比指标：适应度得分、计算时间、发电性能
    """)

    comparison_results = st.session_state.algorithm_comparison_results

    # 创建对比表格
    comparison_data = []
    for algo, data in comparison_results.items():
        if data['result']:
            power_results = data['result'].get('power_results', {})
            comparison_data.append({
                '算法': algo,
                '适应度得分': f"{data['fitness']:.3f}",
                '计算时间(秒)': f"{data['computation_time']:.2f}",
                '年发电量(GWh)': f"{power_results.get('total_annual_generation_gwh', 0):.2f}" if power_results else "N/A",
                '容量因数(%)': f"{power_results.get('average_capacity_factor', 0) * 100:.1f}" if power_results else "N/A",
                '状态': '✅ 成功'
            })
        else:
            comparison_data.append({
                '算法': algo,
                '适应度得分': '0.000',
                '计算时间(秒)': '0.00',
                '年发电量(GWh)': 'N/A',
                '容量因数(%)': 'N/A',
                '状态': f'❌ 失败: {data.get("error", "未知错误")}'
            })

    # 显示对比表格
    import pandas as pd
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # 性能对比图表
    st.markdown("**📊 算法性能对比**")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 准备数据
    algorithms = []
    fitness_scores = []
    computation_times = []

    for algo, data in comparison_results.items():
        if data['result']:
            algorithms.append(algo)
            fitness_scores.append(data['fitness'])
            computation_times.append(data['computation_time'])

    if algorithms:
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('适应度得分对比', '计算时间对比(秒)'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # 适应度得分
        fig.add_trace(
            go.Bar(name='适应度得分', x=algorithms, y=fitness_scores,
                   marker_color='lightblue', text=[f'{score:.3f}' for score in fitness_scores],
                   textposition='auto'),
            row=1, col=1
        )

        # 计算时间
        fig.add_trace(
            go.Bar(name='计算时间', x=algorithms, y=computation_times,
                   marker_color='lightcoral', text=[f'{time:.1f}s' for time in computation_times],
                   textposition='auto'),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="优化算法性能对比"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 算法推荐
        st.markdown("**💡 算法推荐**")

        # 根据性能指标给出推荐
        best_fitness_algo = max(zip(fitness_scores, algorithms))[1]
        fastest_algo = min(zip(computation_times, algorithms))[1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("最佳效果算法", best_fitness_algo)
        with col2:
            st.metric("最快算法", fastest_algo)

        # 综合推荐
        if best_fitness_algo == fastest_algo:
            st.success(f"🎯 推荐使用 {best_fitness_algo} - 既高效又快速")
        else:
            st.info(f"⚖️ 平衡选择: 追求效果选 {best_fitness_algo}, 追求速度选 {fastest_algo}")


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