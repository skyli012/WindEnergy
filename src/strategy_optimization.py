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
        st.session_state.n_farms = 1
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
                    algorithm_params['initial_temp'] = st.slider("初始温度", 100, 5000, 3000, 100)
                with col16:
                    algorithm_params['cooling_rate'] = st.slider("降温速率", 0.85, 0.99, 0.85, 0.01)
                with col17:
                    algorithm_params['iterations_per_temp'] = st.slider("每温度迭代次数", 10, 200, 100)

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


from src.plotting_functions import create_convergence_chart, create_wind_farm_tables, create_wind_resource_tables, \
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
        # 🔧 新增：算法参数与性能指标表格
        st.markdown("#### ⚙️ 算法参数与性能指标")
        create_algorithm_parameters_table(result, wind_farm_fengjie, power_results)

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


def create_algorithm_parameters_table(result, wind_farm_df, power_results):
    """
    创建算法参数与性能指标表格
    """
    # 收集算法参数
    algorithm_params = {}

    # 算法基本信息
    algorithm_name = result.get('algorithm', '未知算法')
    algorithm_params['算法名称'] = algorithm_name

    # 获取算法超参数
    params = result.get('algorithm_params', {})
    if params:
        for key, value in params.items():
            if isinstance(value, (int, float)):
                algorithm_params[f'参数_{key}'] = f"{value:.4f}" if isinstance(value, float) else str(value)

    # 如果没有明确的参数，尝试从结果中提取
    if not algorithm_params:
        # 尝试提取常见的算法参数
        common_params = ['population_size', 'max_iterations', 'learning_rate', 'crossover_rate',
                         'mutation_rate', 'elite_size', 'temperature', 'cooling_rate']

        for param in common_params:
            if param in result:
                value = result[param]
                if isinstance(value, (int, float, str)):
                    algorithm_params[f'参数_{param}'] = str(value)

    # 计算性能指标
    performance_metrics = {}

    # 适应度得分
    fitness_value = result.get('best_fitness') or result.get('fitness') or result.get('best_score') or 0
    performance_metrics['适应度得分'] = f"{fitness_value:.4f}"

    # 计算时间
    execution_time = result.get('execution_time') or result.get('computation_time') or result.get('time', 0)
    if execution_time:
        if execution_time < 60:
            performance_metrics['计算时间'] = f"{execution_time:.2f} 秒"
        elif execution_time < 3600:
            performance_metrics['计算时间'] = f"{execution_time / 60:.2f} 分钟"
        else:
            performance_metrics['计算时间'] = f"{execution_time / 3600:.2f} 小时"
    else:
        performance_metrics['计算时间'] = "N/A"

    # 年发电量
    if power_results and 'total_annual_generation_gwh' in power_results:
        annual_generation = power_results['total_annual_generation_gwh']
        performance_metrics['年发电量'] = f"{annual_generation:.2f} GWh"
    else:
        performance_metrics['年发电量'] = "N/A"

    # 平均风速
    if 'predicted_wind_speed' in wind_farm_df.columns:
        avg_wind_speed = wind_farm_df["predicted_wind_speed"].mean()
        performance_metrics['平均风速'] = f"{avg_wind_speed:.2f} m/s"
    else:
        performance_metrics['平均风速'] = "N/A"

    # 风能利用率
    if 'predicted_wind_speed' in wind_farm_df.columns:
        wind_speed_series = wind_farm_df["predicted_wind_speed"]
        utilization_rate = calculate_wind_utilization(wind_speed_series)
        performance_metrics['风能利用率'] = f"{utilization_rate:.2%}"
    else:
        performance_metrics['风能利用率'] = "N/A"

    # 迭代次数
    fitness_history = result.get("fitness_history") or result.get("convergence_history") or []
    if fitness_history:
        performance_metrics['迭代次数'] = len(fitness_history)
    else:
        performance_metrics['迭代次数'] = result.get('max_iterations', 'N/A')

    # 风机数量
    performance_metrics['风机数量'] = len(wind_farm_df)

    # 创建表格
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 算法超参数")
        if algorithm_params:
            params_df = pd.DataFrame(list(algorithm_params.items()), columns=['参数名称', '参数值'])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        else:
            st.info("无算法参数信息")

    with col2:
        st.markdown("##### 性能指标")
        metrics_df = pd.DataFrame(list(performance_metrics.items()), columns=['指标名称', '指标值'])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
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


# src/visualization/storage_schedule_display.py

from plotly.subplots import make_subplots
import plotly.express as px


def display_storage_schedule_analysis(result, df):
    """
    显示储能调度详细分析 - 支持多风场分析
    """
    if 'storage_results' not in result:
        st.warning("⚠️ 没有找到储能调度数据")
        return

    storage_results = result['storage_results']
    best_strategy = result.get('best_strategy', '未知')

    # 检查是否为多风场数据
    if isinstance(storage_results, list):
        # 多风场情况
        display_multi_farm_storage_analysis(result, df)
    else:
        # 单风场情况
        display_single_farm_storage_analysis(result, df)


def display_single_farm_storage_analysis(result, df):
    """
    显示单风场储能调度分析 - 改为垂直布局
    """
    storage_results = result['storage_results']
    best_strategy = result.get('best_strategy', '未知')

    st.markdown("### ⚡ 储能调度详细分析")

    # 不再使用标签页，改为垂直顺序排列
    # 1. 功率平衡分析
    st.markdown("#### 📊 功率平衡分析")
    display_power_balance_analysis(storage_results, best_strategy, farm_name="主风场")

    st.markdown("---")  # 添加分割线

    # 2. 储能状态分析
    st.markdown("#### 🔋 储能状态分析")
    display_storage_state_analysis(storage_results, best_strategy, farm_name="主风场")

    st.markdown("---")  # 添加分割线

    # 3. 详细充放电状态数据
    st.markdown("#### 📈 详细充放电状态")
    display_detailed_storage_status(storage_results, best_strategy, farm_name="主风场")

    st.markdown("---")  # 添加分割线

    # 4. 调度性能指标
    st.markdown("#### 🎯 调度性能指标")
    display_scheduling_performance_metrics(storage_results, best_strategy, farm_name="主风场")

    st.markdown("---")  # 添加分割线

    # 5. 策略效果对比（如果有多个策略）
    if 'strategy_comparison' in result:
        st.markdown("#### 🔄 策略效果对比")
        display_strategy_effect_comparison(result)


def display_multi_farm_storage_analysis(result, df):
    """
    显示多风场储能调度分析 - 改为标签页切换
    """
    # 获取风场信息
    n_farms = result.get('n_farms', 1)
    storage_results_list = result['storage_results']
    best_strategy = result.get('best_strategy', '未知')

    st.markdown(f"### ⚡ 多风场储能调度分析 ({best_strategy}策略)")

    # 显示风场数量统计
    st.info(f"🏭 共优化了 {n_farms} 个风场，选择标签页查看各风场详细数据")

    # 为每个风场创建标签页
    farm_tabs = st.tabs([f"🏭 风场 {i + 1}" for i in range(n_farms)])

    for i, tab in enumerate(farm_tabs):
        with tab:
            if i < len(storage_results_list):
                farm_storage_results = storage_results_list[i]
                st.markdown(f"#### 风场 {i + 1} - {best_strategy}策略详细分析")

                # 1. 功率平衡分析
                st.markdown("##### 📊 功率平衡分析")
                display_power_balance_analysis(farm_storage_results, best_strategy, f"风场 {i + 1}")

                st.markdown("---")  # 添加分割线

                # 2. 储能状态分析
                st.markdown("##### 🔋 储能状态分析")
                display_storage_state_analysis(farm_storage_results, best_strategy, f"风场 {i + 1}")

                st.markdown("---")  # 添加分割线

                # 3. 详细充放电状态数据
                st.markdown("##### 📈 详细充放电状态")
                display_detailed_storage_status(farm_storage_results, best_strategy, f"风场 {i + 1}")

                st.markdown("---")  # 添加分割线

                # 4. 调度性能指标
                st.markdown("##### 🎯 调度性能指标")
                display_scheduling_performance_metrics(farm_storage_results, best_strategy, f"风场 {i + 1}")
            else:
                st.warning(f"⚠️ 风场 {i + 1} 暂无储能调度数据")

    # 在所有风场标签页之后，添加综合对比分析
    st.markdown("---")
    st.markdown("### 📊 多风场综合对比分析")

    # 创建两个标签页：性能对比和策略效果
    comparison_tabs = st.tabs(["📈 风场性能对比", "🔄 策略效果对比"])

    with comparison_tabs[0]:
        display_multi_farm_comparison(result)

    with comparison_tabs[1]:
        if 'strategy_comparison' in result:
            display_strategy_effect_comparison(result)
        else:
            st.info("暂无策略比较数据")


def display_multi_farm_comparison(result):
    """
    显示多风场综合对比
    """
    st.markdown("#### 📊 多风场性能对比")

    if 'storage_results' not in result or not isinstance(result['storage_results'], list):
        st.info("暂无多风场数据")
        return

    storage_results_list = result['storage_results']
    n_farms = len(storage_results_list)

    # 收集各风场性能指标
    farm_metrics = []
    for i, farm_storage in enumerate(storage_results_list):
        metrics = calculate_scheduling_performance(farm_storage)
        metrics['风场编号'] = i + 1

        # 添加储能参数信息
        if 'storage_params' in farm_storage:
            storage_params = farm_storage['storage_params']
            metrics.update({
                '储能容量_kWh': storage_params.get('storage_capacity', 0),
                '储能功率_kW': storage_params.get('storage_power', 0),
                '策略类型': storage_params.get('storage_strategy', '未知')
            })
        farm_metrics.append(metrics)

    # 创建对比表格
    comparison_data = []
    for metrics in farm_metrics:
        comparison_data.append({
            '风场': f"风场 {metrics['风场编号']}",
            '储能容量(kWh)': f"{metrics.get('储能容量_kWh', 0):.0f}",
            '储能功率(kW)': f"{metrics.get('储能功率_kW', 0):.0f}",
            '策略类型': metrics.get('策略类型', '未知'),
            '平滑效果 (%)': f"{metrics['smoothing_effect']:.1f}",
            '储能利用率 (%)': f"{metrics['storage_utilization']:.1f}",
            '弃风率 (%)': f"{metrics['curtailment_rate']:.1f}",
            '系统效率 (%)': f"{metrics['system_efficiency']:.1f}",
            'SOC保持率 (%)': f"{metrics['soc_maintenance']:.1f}"
        })

    comparison_df = pd.DataFrame(comparison_data)

    # 使用条件格式突出显示性能指标
    def color_performance(val):
        try:
            num_val = float(str(val).replace('%', ''))
            if num_val >= 80:
                return 'background-color: #d4edda; color: #155724;'  # 优秀 - 绿色
            elif num_val >= 60:
                return 'background-color: #fff3cd; color: #856404;'  # 良好 - 黄色
            else:
                return 'background-color: #f8d7da; color: #721c24;'  # 待改进 - 红色
        except:
            return ''

    # 应用样式
    styled_df = comparison_df.style.applymap(color_performance, subset=[
        '平滑效果 (%)', '储能利用率 (%)', '系统效率 (%)', 'SOC保持率 (%)'
    ])

    st.dataframe(styled_df, use_container_width=True)

    # 创建性能对比图表
    fig = go.Figure()

    farms = [f"风场 {i + 1}" for i in range(n_farms)]
    metrics_to_compare = ['smoothing_effect', 'storage_utilization', 'system_efficiency', 'soc_maintenance']
    metric_names = ['平滑效果', '储能利用率', '系统效率', 'SOC保持率']

    for metric, name in zip(metrics_to_compare, metric_names):
        values = [farm_metrics[i][metric] for i in range(n_farms)]
        fig.add_trace(go.Bar(
            name=name,
            x=farms,
            y=values,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto'
        ))

    fig.update_layout(
        title="多风场储能性能对比",
        barmode='group',
        height=400,
        yaxis_title="性能指标值 (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 添加储能参数对比图表
    if all('储能容量_kWh' in metrics for metrics in farm_metrics):
        fig2 = go.Figure()

        capacities = [metrics.get('储能容量_kWh', 0) for metrics in farm_metrics]
        powers = [metrics.get('储能功率_kW', 0) for metrics in farm_metrics]

        fig2.add_trace(go.Bar(
            name='储能容量 (kWh)',
            x=farms,
            y=capacities,
            text=[f"{c:.0f}" for c in capacities],
            textposition='auto'
        ))

        fig2.add_trace(go.Bar(
            name='储能功率 (kW)',
            x=farms,
            y=powers,
            text=[f"{p:.0f}" for p in powers],
            textposition='auto'
        ))

        fig2.update_layout(
            title="多风场储能配置对比",
            barmode='group',
            height=400,
            yaxis_title="储能参数值",
            template="plotly_white"
        )

        st.plotly_chart(fig2, use_container_width=True)


def display_multi_farm_comparison(result):
    """
    显示多风场综合对比
    """
    st.markdown("#### 📊 多风场性能对比")

    if 'storage_results' not in result or not isinstance(result['storage_results'], list):
        return

    storage_results_list = result['storage_results']
    n_farms = len(storage_results_list)

    # 收集各风场性能指标
    farm_metrics = []
    for i, farm_storage in enumerate(storage_results_list):
        metrics = calculate_scheduling_performance(farm_storage)
        metrics['风场编号'] = i + 1
        farm_metrics.append(metrics)

    # 创建对比表格
    comparison_data = []
    for metrics in farm_metrics:
        comparison_data.append({
            '风场': f"风场 {metrics['风场编号']}",
            '平滑效果 (%)': f"{metrics['smoothing_effect']:.1f}",
            '储能利用率 (%)': f"{metrics['storage_utilization']:.1f}",
            '弃风率 (%)': f"{metrics['curtailment_rate']:.1f}",
            '系统效率 (%)': f"{metrics['system_efficiency']:.1f}",
            'SOC保持率 (%)': f"{metrics['soc_maintenance']:.1f}"
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # 创建性能对比图表
    fig = go.Figure()

    farms = [f"风场 {i + 1}" for i in range(n_farms)]
    metrics_to_compare = ['smoothing_effect', 'storage_utilization', 'system_efficiency', 'soc_maintenance']
    metric_names = ['平滑效果', '储能利用率', '系统效率', 'SOC保持率']

    for metric, name in zip(metrics_to_compare, metric_names):
        values = [farm_metrics[i][metric] for i in range(n_farms)]
        fig.add_trace(go.Bar(
            name=name,
            x=farms,
            y=values,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto'
        ))

    fig.update_layout(
        title="多风场储能性能对比",
        barmode='group',
        height=400,
        yaxis_title="性能指标值 (%)",
        yaxis=dict(range=[0, 100])
    )

    st.plotly_chart(fig, use_container_width=True)


def display_power_balance_analysis(storage_results, strategy, farm_name="风场"):
    """
    显示功率平衡分析（只保留图表部分）
    """
    st.markdown(f"**🔌 功率平衡分析 - {farm_name} ({strategy}策略)**")

    # 检查数据结构
    if 'schedule_data' not in storage_results:
        st.error("❌ 调度数据格式错误")
        return

    schedule_data = storage_results['schedule_data']

    # 使用正确的时间列
    if 'time_index' in schedule_data.columns:
        time_data = schedule_data['time_index']
        time_label = "时间点"
    elif 'hour' in schedule_data.columns:
        time_data = schedule_data['hour']
        time_label = "时间 (小时)"
    else:
        # 如果没有时间列，创建索引
        time_data = range(len(schedule_data))
        time_label = "时间点"

    # 限制显示的数据点数量
    min_length = min(len(schedule_data), 144)  # 最多显示一天数据

    # 创建功率平衡图
    fig = go.Figure()

    # 添加风能功率曲线
    if 'wind_power' in schedule_data.columns:
        fig.add_trace(go.Scatter(
            x=time_data[:min_length],
            y=schedule_data['wind_power'][:min_length],
            mode='lines',
            name='风能功率',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))

    # 添加电网功率曲线
    if 'grid_power' in schedule_data.columns:
        fig.add_trace(go.Scatter(
            x=time_data[:min_length],
            y=schedule_data['grid_power'][:min_length],
            mode='lines',
            name='电网功率',
            line=dict(color='#2ca02c', width=2, dash='dash')
        ))

    # 添加储能功率曲线
    if 'battery_power' in schedule_data.columns:
        fig.add_trace(go.Scatter(
            x=time_data[:min_length],
            y=schedule_data['battery_power'][:min_length],
            mode='lines',
            name='储能功率',
            line=dict(color='#ff7f0e', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.1)'
        ))

    # 添加零线
    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    fig.update_layout(
        title=f"{farm_name}功率平衡分析 - {strategy}策略",
        xaxis_title=time_label,
        yaxis_title="功率 (kW)",
        hovermode="x unified",
        height=400,
        showlegend=True,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 功率统计信息
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'wind_power' in schedule_data.columns:
            avg_wind_power = schedule_data['wind_power'].mean()
            st.metric("平均风能功率", f"{avg_wind_power:.0f} kW")

    with col2:
        if 'wind_power' in schedule_data.columns:
            max_wind_power = schedule_data['wind_power'].max()
            st.metric("最大风能功率", f"{max_wind_power:.0f} kW")

    with col3:
        if 'grid_power' in schedule_data.columns:
            avg_grid_power = schedule_data['grid_power'].mean()
            st.metric("平均电网功率", f"{avg_grid_power:.0f} kW")

    with col4:
        if 'grid_power' in schedule_data.columns:
            grid_fluctuation = schedule_data['grid_power'].std()
            st.metric("电网波动", f"{grid_fluctuation:.0f} kW")


def display_detailed_storage_status(storage_results, strategy, farm_name="风场"):
    """
    专门显示详细充放电状态数据表格 - 显示完整数据
    """
    if 'schedule_data' not in storage_results:
        st.error("❌ 调度数据格式错误")
        return

    schedule_data = storage_results['schedule_data']

    # 显示完整数据
    st.info(f"📋 显示{farm_name}完整充放电状态数据（共{len(schedule_data)}个时间点）")

    # 创建状态标识
    def get_operation_status(row):
        """获取操作状态"""
        if 'battery_power' not in row:
            return "待机"

        battery_power = row['battery_power']
        if battery_power > 0:
            return "放电"
        elif battery_power < 0:
            return "充电"
        else:
            return "待机"

    def get_power_balance_status(row):
        """获取功率平衡状态"""
        if 'wind_power' not in row or 'grid_power' not in row:
            return "未知"

        wind_power = row['wind_power']
        grid_power = row['grid_power']

        if wind_power > grid_power:
            return "风电过剩"
        elif wind_power < grid_power:
            return "风电不足"
        else:
            return "平衡"

    # 使用完整数据
    table_data = []
    for i, (_, row) in enumerate(schedule_data.iterrows()):
        # 时间信息
        if 'timestamp' in row:
            time_str = str(row['timestamp'])
        elif 'hour' in row:
            hour = int(row['hour'])
            minute = int(row['minute']) if 'minute' in row else 0
            time_str = f"{hour:02d}:{minute:02d}"
        else:
            time_str = f"时间点 {i + 1}"

        # 获取状态
        operation_status = get_operation_status(row)
        balance_status = get_power_balance_status(row)

        # 功率数据
        wind_power = row.get('wind_power', 0)
        grid_power = row.get('grid_power', 0)
        battery_power = row.get('battery_power', 0)
        soc = row.get('storage_soc', 0) * 100 if 'storage_soc' in row else 0

        # 弃风功率
        wind_curtailment = row.get('wind_curtailment', 0)

        table_data.append({
            "序号": i + 1,
            "时间": time_str,
            "风电功率(kW)": f"{wind_power:.1f}",
            "电网功率(kW)": f"{grid_power:.1f}",
            "储能功率(kW)": f"{battery_power:+.1f}",  # 使用+号显示正负
            "储能状态": operation_status,
            "SOC(%)": f"{soc:.1f}",
            "功率平衡": balance_status,
            "弃风功率(kW)": f"{wind_curtailment:.1f}",
            "净功率(kW)": f"{(wind_power + battery_power):.1f}"
        })

    # 创建数据框
    status_df = pd.DataFrame(table_data)

    # 添加筛选功能
    st.markdown("**🔍 数据筛选**")
    col1, col2, col3 = st.columns(3)

    with col1:
        # 按状态筛选
        status_options = ["全部", "充电", "放电", "待机"]
        selected_status = st.selectbox("储能状态筛选", status_options, key=f"status_filter_{farm_name}")

    with col2:
        # 按功率平衡筛选
        balance_options = ["全部", "风电过剩", "风电不足", "平衡"]
        selected_balance = st.selectbox("功率平衡筛选", balance_options, key=f"balance_filter_{farm_name}")

    with col3:
        # 按时间范围筛选
        if 'hour' in schedule_data.columns:
            min_hour = int(schedule_data['hour'].min())
            max_hour = int(schedule_data['hour'].max())
            time_range = st.slider(
                "时间范围筛选(小时)",
                min_hour, max_hour,
                (min_hour, max_hour),
                key=f"time_filter_{farm_name}"
            )

    # 应用筛选
    filtered_df = status_df.copy()

    if selected_status != "全部":
        filtered_df = filtered_df[filtered_df["储能状态"] == selected_status]

    if selected_balance != "全部":
        filtered_df = filtered_df[filtered_df["功率平衡"] == selected_balance]

    if 'hour' in schedule_data.columns:
        # 需要从原始时间字符串提取小时
        def extract_hour(time_str):
            try:
                # 处理格式如 "08:30"
                if ":" in time_str:
                    return int(time_str.split(":")[0])
                return 0
            except:
                return 0

        filtered_df = filtered_df[filtered_df["时间"].apply(extract_hour).between(time_range[0], time_range[1])]

    # 显示筛选结果统计
    st.info(f"📊 显示 {len(filtered_df)} 条数据（共 {len(status_df)} 条）")

    # 使用条件格式突出显示重要信息
    def color_operation_status(val):
        """根据操作状态着色"""
        if val == "充电":
            return 'background-color: #d4edda; color: #155724;'  # 绿色
        elif val == "放电":
            return 'background-color: #f8d7da; color: #721c24;'  # 红色
        else:
            return 'background-color: #e2e3e5; color: #383d41;'  # 灰色

    def color_balance_status(val):
        """根据平衡状态着色"""
        if val == "风电过剩":
            return 'background-color: #fff3cd; color: #856404;'  # 黄色
        elif val == "风电不足":
            return 'background-color: #cce5ff; color: #004085;'  # 蓝色
        else:
            return ''

    # 应用样式
    styled_df = filtered_df.style.applymap(color_operation_status, subset=['储能状态'])
    styled_df = styled_df.applymap(color_balance_status, subset=['功率平衡'])

    # 添加分页功能
    page_size = 50  # 每页显示50条数据
    total_pages = max(1, (len(filtered_df) + page_size - 1) // page_size)

    # 页码选择
    current_page = st.selectbox(
        "选择页码",
        range(1, total_pages + 1),
        key=f"page_select_{farm_name}"
    )

    # 计算当前页数据
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered_df))
    page_df = filtered_df.iloc[start_idx:end_idx]

    # 显示当前页数据
    st.write(f"📄 第 {current_page} 页，显示第 {start_idx + 1} - {end_idx} 条数据")

    # 显示表格（不设置固定高度，让表格根据内容自适应）
    st.dataframe(
        page_df.style.applymap(color_operation_status, subset=['储能状态'])
        .applymap(color_balance_status, subset=['功率平衡']),
        use_container_width=True
    )

    # 添加数据导出功能
    st.markdown("**💾 数据导出**")
    col_export1, col_export2, col_export3 = st.columns(3)

    with col_export1:
        if st.button(f"📥 导出筛选数据 (CSV)", key=f"export_csv_{farm_name}"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="点击下载 CSV",
                data=csv,
                file_name=f"{farm_name}_储能调度数据.csv",
                mime="text/csv"
            )

    with col_export2:
        if st.button(f"📥 导出完整数据 (CSV)", key=f"export_full_csv_{farm_name}"):
            csv = status_df.to_csv(index=False)
            st.download_button(
                label="点击下载完整 CSV",
                data=csv,
                file_name=f"{farm_name}_完整储能调度数据.csv",
                mime="text/csv"
            )

    with col_export3:
        if st.button(f"📊 显示统计摘要", key=f"show_stats_{farm_name}"):
            show_data_statistics(filtered_df, farm_name)

    # 添加汇总统计
    st.markdown("**📈 充放电统计汇总**")

    if 'battery_power' in schedule_data.columns:
        # 计算统计信息
        charge_data = schedule_data[schedule_data['battery_power'] < 0]
        discharge_data = schedule_data[schedule_data['battery_power'] > 0]
        idle_data = schedule_data[schedule_data['battery_power'] == 0]

        total_time = len(schedule_data)

        col1, col2, col3 = st.columns(3)

        with col1:
            charge_time = len(charge_data)
            charge_ratio = (charge_time / total_time * 100) if total_time > 0 else 0
            avg_charge_power = abs(charge_data['battery_power'].mean()) if len(charge_data) > 0 else 0
            total_charge_energy = abs(charge_data['battery_power'].sum() * (10 / 60)) if len(charge_data) > 0 else 0
            st.metric(
                "充电统计",
                f"{charge_ratio:.1f}%",
                f"{charge_time}个时段，平均{avg_charge_power:.0f} kW，总能量{total_charge_energy:.0f} kWh"
            )

        with col2:
            discharge_time = len(discharge_data)
            discharge_ratio = (discharge_time / total_time * 100) if total_time > 0 else 0
            avg_discharge_power = discharge_data['battery_power'].mean() if len(discharge_data) > 0 else 0
            total_discharge_energy = discharge_data['battery_power'].sum() * (10 / 60) if len(discharge_data) > 0 else 0
            st.metric(
                "放电统计",
                f"{discharge_ratio:.1f}%",
                f"{discharge_time}个时段，平均{avg_discharge_power:.0f} kW，总能量{total_discharge_energy:.0f} kWh"
            )

        with col3:
            idle_time = len(idle_data)
            idle_ratio = (idle_time / total_time * 100) if total_time > 0 else 0
            st.metric(
                "待机统计",
                f"{idle_ratio:.1f}%",
                f"{idle_time}个时段"
            )


def show_data_statistics(data_df, farm_name):
    """
    显示数据统计摘要
    """
    if len(data_df) == 0:
        st.warning("没有数据可统计")
        return

    st.markdown(f"#### 📊 {farm_name}数据统计摘要")

    # 创建统计卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_records = len(data_df)
        st.metric("总记录数", f"{total_records}")

    with col2:
        charge_count = len(data_df[data_df["储能状态"] == "充电"])
        st.metric("充电次数", f"{charge_count}")

    with col3:
        discharge_count = len(data_df[data_df["储能状态"] == "放电"])
        st.metric("放电次数", f"{discharge_count}")

    with col4:
        idle_count = len(data_df[data_df["储能状态"] == "待机"])
        st.metric("待机次数", f"{idle_count}")

    # 更多统计信息
    if len(data_df) > 0:
        # 提取数值数据
        def extract_numeric(col_name):
            try:
                # 移除单位并转换为浮点数
                return data_df[col_name].str.replace('kW', '').str.replace('kWh', '').str.replace('%', '').str.replace(
                    '+', '').astype(float)
            except:
                return pd.Series([0] * len(data_df))

        wind_power = extract_numeric("风电功率(kW)")
        grid_power = extract_numeric("电网功率(kW)")
        battery_power = extract_numeric("储能功率(kW)")

        if len(wind_power) > 0:
            st.markdown("**功率统计**")
            stats_col1, stats_col2, stats_col3 = st.columns(3)

            with stats_col1:
                st.write("风电功率")
                st.write(f"- 平均: {wind_power.mean():.1f} kW")
                st.write(f"- 最大: {wind_power.max():.1f} kW")
                st.write(f"- 最小: {wind_power.min():.1f} kW")

            with stats_col2:
                st.write("电网功率")
                st.write(f"- 平均: {grid_power.mean():.1f} kW")
                st.write(f"- 最大: {grid_power.max():.1f} kW")
                st.write(f"- 最小: {grid_power.min():.1f} kW")

            with stats_col3:
                st.write("储能功率")
                st.write(f"- 平均: {battery_power.mean():.1f} kW")
                st.write(f"- 最大: {battery_power.max():.1f} kW")
                st.write(f"- 最小: {battery_power.min():.1f} kW")


def display_storage_state_analysis(storage_results, strategy, farm_name="风场"):
    """
    显示储能状态分析
    """
    st.markdown(f"**🔋 储能状态分析 - {farm_name} ({strategy}策略)**")

    if 'schedule_data' not in storage_results:
        st.error("❌ 调度数据格式错误")
        return

    schedule_data = storage_results['schedule_data']

    # 使用正确的时间列
    if 'time_index' in schedule_data.columns:
        time_data = schedule_data['time_index']
        time_label = "时间点"
    elif 'hour' in schedule_data.columns:
        time_data = schedule_data['hour']
        time_label = "时间 (小时)"
    else:
        time_data = range(len(schedule_data))
        time_label = "时间点"

    min_length = min(len(schedule_data), 144)

    # 创建子图：上图为SOC，下图为功率
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{farm_name}储能荷电状态 (SOC)', f'{farm_name}储能充放电功率'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # SOC曲线
    if 'storage_soc' in schedule_data.columns:
        fig.add_trace(go.Scatter(
            x=time_data[:min_length],
            y=schedule_data['storage_soc'][:min_length] * 100,  # 转换为百分比
            mode='lines',
            name='SOC',
            line=dict(color='#9467bd', width=3),
            fill='tozeroy',
            fillcolor='rgba(148, 103, 189, 0.1)'
        ), row=1, col=1)

    # 充放电功率曲线
    if 'battery_power' in schedule_data.columns:
        fig.add_trace(go.Scatter(
            x=time_data[:min_length],
            y=schedule_data['battery_power'][:min_length],
            mode='lines',
            name='储能功率',
            line=dict(color='#e377c2', width=2),
            fill='tozeroy',
            fillcolor='rgba(227, 119, 194, 0.1)'
        ), row=2, col=1)

        # 添加充放电区域标识
        battery_power = schedule_data['battery_power']
        fig.add_hrect(y0=0, y1=battery_power.max() if len(battery_power) > 0 else 0,
                      line_width=0, fillcolor="red", opacity=0.1, row=2, col=1)
        fig.add_hrect(y0=battery_power.min() if len(battery_power) > 0 else 0, y1=0,
                      line_width=0, fillcolor="green", opacity=0.1, row=2, col=1)

    fig.update_yaxes(title_text="SOC (%)", row=1, col=1)
    fig.update_yaxes(title_text="功率 (kW)", row=2, col=1)
    fig.update_xaxes(title_text=time_label, row=2, col=1)

    fig.update_layout(
        title=f"{farm_name}储能状态分析 - {strategy}策略",
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # SOC统计信息
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'storage_soc' in schedule_data.columns:
            avg_soc = schedule_data['storage_soc'].mean() * 100
            st.metric("平均SOC", f"{avg_soc:.1f}%")

    with col2:
        if 'storage_soc' in schedule_data.columns:
            min_soc = schedule_data['storage_soc'].min() * 100
            st.metric("最低SOC", f"{min_soc:.1f}%")

    with col3:
        if 'storage_soc' in schedule_data.columns:
            max_soc = schedule_data['storage_soc'].max() * 100
            st.metric("最高SOC", f"{max_soc:.1f}%")

    with col4:
        if 'storage_soc' in schedule_data.columns:
            soc_fluctuation = schedule_data['storage_soc'].std() * 100
            st.metric("SOC波动", f"{soc_fluctuation:.1f}%")


def display_strategy_effect_comparison(result):
    """
    显示策略效果对比分析
    """
    st.markdown("**📊 储能策略效果对比**")

    if 'strategy_comparison' not in result:
        st.info("暂无策略比较数据")
        return

    comparison_data = result['strategy_comparison']

    # 创建柱状图显示策略适应度对比
    strategies = [item['strategy'] for item in comparison_data]
    fitness_scores = [item['fitness'] for item in comparison_data]

    fig = px.bar(
        x=strategies,
        y=fitness_scores,
        title="储能策略适应度对比",
        labels={'x': '策略类型', 'y': '适应度得分'},
        color=fitness_scores,
        color_continuous_scale='viridis'
    )

    fig.update_layout(
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # 策略效果表格
    effect_data = {
        '策略': strategies,
        '适应度得分': [f"{score:.3f}" for score in fitness_scores],
        '计算时间(秒)': [f"{item.get('computation_time', 0):.1f}" for item in comparison_data],
        '质量评级': [item.get('quality_rating', '未知') for item in comparison_data]
    }

    effect_df = pd.DataFrame(effect_data)
    st.dataframe(effect_df, use_container_width=True)


def display_scheduling_performance_metrics(storage_results, strategy, farm_name="风场"):
    """
    显示调度性能指标
    """
    st.markdown(f"**🎯 调度性能指标 - {farm_name} ({strategy}策略)**")

    # 计算性能指标
    performance_metrics = calculate_scheduling_performance(storage_results)

    # 显示关键指标
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("平滑效果", f"{performance_metrics['smoothing_effect']:.1f}%")

    with col2:
        st.metric("储能利用率", f"{performance_metrics['storage_utilization']:.1f}%")

    with col3:
        st.metric("弃风率", f"{performance_metrics['curtailment_rate']:.1f}%")

    with col4:
        st.metric("系统效率", f"{performance_metrics['system_efficiency']:.1f}%")

    # 详细指标表格
    st.markdown("**详细性能指标**")

    metrics_df = pd.DataFrame([
        {"指标": "风能波动系数", "数值": f"{performance_metrics['wind_fluctuation']:.1f} kW", "说明": "数值越小越稳定"},
        {"指标": "电网波动系数", "数值": f"{performance_metrics['grid_fluctuation']:.1f} kW", "说明": "数值越小越稳定"},
        {"指标": "平滑改善率", "数值": f"{performance_metrics['smoothing_improvement']:.1f}%",
         "说明": "改善比例越高越好"},
        {"指标": "充电次数", "数值": f"{performance_metrics['charge_cycles']}次", "说明": "储能充放电循环次数"},
        {"指标": "总充电能量", "数值": f"{performance_metrics['total_charge_energy']:.0f} kWh", "说明": "储能总充电量"},
        {"指标": "总放电能量", "数值": f"{performance_metrics['total_discharge_energy']:.0f} kWh",
         "说明": "储能总放电量"},
        {"指标": "SOC保持率", "数值": f"{performance_metrics['soc_maintenance']:.1f}%",
         "说明": "SOC维持在合理范围的比例"},
    ])

    st.table(metrics_df)

    # 性能指标可视化
    fig = go.Figure(data=[
        go.Bar(name='性能指标',
               x=['平滑效果', '储能利用率', '系统效率', 'SOC保持率'],
               y=[performance_metrics['smoothing_effect'],
                  performance_metrics['storage_utilization'],
                  performance_metrics['system_efficiency'],
                  performance_metrics['soc_maintenance']])
    ])

    fig.update_layout(
        title=f"{farm_name}调度性能指标对比",
        yaxis_title="性能指标值 (%)",
        yaxis=dict(range=[0, 100]),
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)


def calculate_scheduling_performance(storage_results):
    """
    计算调度性能指标
    """
    if 'schedule_data' not in storage_results or 'performance_metrics' not in storage_results:
        return {
            'smoothing_effect': 0,
            'storage_utilization': 0,
            'curtailment_rate': 0,
            'system_efficiency': 0,
            'wind_fluctuation': 0,
            'grid_fluctuation': 0,
            'smoothing_improvement': 0,
            'charge_cycles': 0,
            'total_charge_energy': 0,
            'total_discharge_energy': 0,
            'soc_maintenance': 0
        }

    schedule_data = storage_results['schedule_data']
    perf_metrics = storage_results['performance_metrics']

    # 使用已有的性能指标
    smoothing_effect = perf_metrics.get('smoothing_effect', 0)
    storage_utilization = perf_metrics.get('storage_utilization', 0)
    curtailment_rate = perf_metrics.get('curtailment_rate', 0)
    system_efficiency = perf_metrics.get('system_efficiency', 0)
    total_charge_energy = perf_metrics.get('total_charge_energy', 0)
    total_discharge_energy = perf_metrics.get('total_discharge_energy', 0)

    # 计算风能波动性
    wind_fluctuation = schedule_data['wind_power'].std() if 'wind_power' in schedule_data.columns else 0

    # 计算电网波动性
    grid_fluctuation = schedule_data['grid_power'].std() if 'grid_power' in schedule_data.columns else 0

    # 计算平滑改善率
    smoothing_improvement = (
            (wind_fluctuation - grid_fluctuation) / wind_fluctuation * 100) if wind_fluctuation > 0 else 0

    # 计算充放电循环次数
    battery_power = schedule_data['battery_power'] if 'battery_power' in schedule_data.columns else pd.Series()
    charge_cycles = len(np.where(np.diff(np.sign(battery_power)) != 0)[0]) // 2 if len(battery_power) > 0 else 0

    # 计算SOC保持率（SOC在20%-80%之间的时间比例）
    storage_soc = schedule_data['storage_soc'] if 'storage_soc' in schedule_data.columns else pd.Series()
    soc_maintenance = ((storage_soc >= 0.2) & (storage_soc <= 0.8)).mean() * 100 if len(storage_soc) > 0 else 0

    return {
        'smoothing_effect': smoothing_effect,
        'storage_utilization': storage_utilization,
        'curtailment_rate': curtailment_rate,
        'system_efficiency': system_efficiency,
        'wind_fluctuation': wind_fluctuation,
        'grid_fluctuation': grid_fluctuation,
        'smoothing_improvement': smoothing_improvement,
        'charge_cycles': charge_cycles,
        'total_charge_energy': total_charge_energy,
        'total_discharge_energy': total_discharge_energy,
        'soc_maintenance': soc_maintenance
    }



# src/optimization/algorithm_convergence_curve.py

import pulp
import time
from scipy.spatial.distance import pdist


# ==============================
# 储能经济性计算函数
# ==============================

def calculate_storage_investment_cost(capacity_kwh, power_kw):
    """计算储能投资成本"""
    # 成本参数（元/kWh 和 元/kW）
    capacity_cost_per_kwh = 1500  # 元/kWh
    power_cost_per_kw = 1000  # 元/kW

    total_cost = capacity_kwh * capacity_cost_per_kwh + power_kw * power_cost_per_kw
    return total_cost


def calculate_storage_annual_revenue(selected_data, capacity_kwh, power_kw, constraints):
    """计算储能年收益"""
    strategy = constraints.get('storage_strategy', '平滑输出')
    electricity_price = 0.4  # 元/kWh

    if strategy == '经济调度' or strategy == '削峰填谷':
        # 峰谷套利收益
        peak_valley_diff = constraints.get('peak_valley_diff', 0.7)  # 峰谷电价差
        daily_cycles = 1.0  # 每天充放电次数
        efficiency = 0.85  # 充放电效率

        # 考虑容量和功率限制
        usable_capacity = capacity_kwh * 0.8  # 可用容量（考虑SOC范围）
        daily_revenue = min(usable_capacity * peak_valley_diff * efficiency * daily_cycles,
                            power_kw * 24 * peak_valley_diff * 0.5)  # 功率限制
        annual_revenue = daily_revenue * 365
    else:
        # 其他策略收益（平滑、削峰等）
        capacity_utilization = 0.3  # 假设30%的容量利用率
        annual_revenue = capacity_kwh * electricity_price * capacity_utilization * 365

    return annual_revenue


def calculate_storage_operation_cost(capacity_kwh, power_kw):
    """计算储能年运行维护成本"""
    # O&M成本（元/kWh/年）
    om_cost_per_kwh = 50
    return capacity_kwh * om_cost_per_kwh


def calculate_wind_utilization(wind_speed_series):
    """
    计算风能利用率指标
    基于风速的稳定性、可利用小时数等因素
    """
    if len(wind_speed_series) == 0:
        return 0

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


def calculate_composite_fitness_with_storage(positions, df, wind_speed_weight=0.6, utilization_weight=0.4,
                                             **constraints):
    """基于风速、风能利用率和储能经济性的综合适应度函数"""
    if len(positions) == 0:
        return 0

    # 获取选中的点位数据
    selected_data = df.loc[positions]

    # 1. 发电量收益（基于真实的风速数据）
    if 'predicted_wind_speed' in selected_data.columns:
        # 使用风功率公式: P = 0.5 * ρ * A * v³
        air_density = 1.225  # kg/m³
        rotor_diameter = 140  # 米
        rotor_area = np.pi * (rotor_diameter / 2) ** 2

        # 计算风速得分（归一化）
        wind_speeds = selected_data['predicted_wind_speed']
        max_wind_speed = df['predicted_wind_speed'].max()
        normalized_wind_speed = wind_speeds.sum() / (len(wind_speeds) * max_wind_speed) if max_wind_speed > 0 else 0

        # 计算风能利用率得分
        if 'wind_utilization_rate' in selected_data.columns:
            utilization_scores = selected_data['wind_utilization_rate']
            max_utilization = df['wind_utilization_rate'].max()
            normalized_utilization = utilization_scores.sum() / (
                    len(utilization_scores) * max_utilization) if max_utilization > 0 else 0
        else:
            # 如果没有预计算的利用率，实时计算
            utilization_rates = []
            for idx in positions:
                point_data = df.loc[idx]
                # 这里简化计算，实际应该使用时间序列数据
                utilization = calculate_wind_utilization(pd.Series([point_data.get('predicted_wind_speed', 0)]))
                utilization_rates.append(utilization)
            normalized_utilization = sum(utilization_rates) / len(utilization_rates) if utilization_rates else 0

        # 综合评分
        composite_score = (wind_speed_weight * normalized_wind_speed +
                           utilization_weight * normalized_utilization)

        # 基础发电量计算（用于约束惩罚的基准）
        power_benefit = 0.5 * air_density * rotor_area * (wind_speeds ** 3).sum()
    else:
        composite_score = 0
        power_benefit = 0

    # 2. 储能经济性计算
    storage_economic_score = 0
    if 'storage_capacity' in constraints and 'storage_power' in constraints:
        storage_capacity_kwh = constraints.get('storage_capacity', 0)
        storage_power_kw = constraints.get('storage_power', 0)

        if storage_capacity_kwh > 0 and storage_power_kw > 0:
            # 计算储能投资成本
            storage_investment = calculate_storage_investment_cost(storage_capacity_kwh, storage_power_kw)

            # 计算储能年收益
            storage_annual_revenue = calculate_storage_annual_revenue(
                selected_data, storage_capacity_kwh, storage_power_kw, constraints
            )

            # 计算储能运维成本
            storage_om_cost = calculate_storage_operation_cost(storage_capacity_kwh, storage_power_kw)

            # 计算储能净年收益
            storage_net_annual_benefit = storage_annual_revenue - storage_om_cost

            # 计算储能投资回收期（简化）
            storage_payback_years = storage_investment / storage_net_annual_benefit if storage_net_annual_benefit > 0 else float(
                'inf')

            # 储能经济性评分（回收期越短评分越高）
            if storage_payback_years < 5:
                storage_economic_score = 1.0
            elif storage_payback_years < 10:
                storage_economic_score = 0.8
            elif storage_payback_years < 15:
                storage_economic_score = 0.5
            elif storage_payback_years < 20:
                storage_economic_score = 0.3
            else:
                storage_economic_score = 0.1

            # 添加到适应度中（适当权重）
            storage_weight = constraints.get('storage_weight', 0.3)  # 储能经济性权重
            composite_score += storage_economic_score * storage_weight

    # 3. 成本惩罚（基于真实的约束条件）
    cost_penalty = 0

    # 坡度约束惩罚
    if 'slope' in selected_data.columns:
        max_slope = constraints.get('max_slope', 15)
        slope_violation = selected_data[selected_data['slope'] > max_slope]['slope'].sum()
        cost_penalty += slope_violation * 10

    # 道路距离约束惩罚
    if 'road_distance' in selected_data.columns:
        max_road_distance = constraints.get('max_road_distance', 1000)
        road_violation = selected_data[selected_data['road_distance'] > max_road_distance]['road_distance'].sum()
        cost_penalty += road_violation * 0.1

    # 居民区距离约束惩罚
    if 'residential_distance' in selected_data.columns:
        min_residential_distance = constraints.get('min_residential_distance', 600)
        residential_violation = selected_data[selected_data['residential_distance'] < min_residential_distance]
        if len(residential_violation) > 0:
            violation_amount = (min_residential_distance - residential_violation['residential_distance']).sum()
            cost_penalty += violation_amount * 5

    # 文化遗产距离约束惩罚
    if 'heritage_distance' in selected_data.columns:
        min_heritage_distance = constraints.get('min_heritage_distance', 700)
        heritage_violation = selected_data[selected_data['heritage_distance'] < min_heritage_distance]
        if len(heritage_violation) > 0:
            violation_amount = (min_heritage_distance - heritage_violation['heritage_distance']).sum()
            cost_penalty += violation_amount * 8

    # 地质距离约束惩罚
    if 'geology_distance' in selected_data.columns:
        min_geology_distance = constraints.get('min_geology_distance', 800)
        geology_violation = selected_data[selected_data['geology_distance'] < min_geology_distance]
        if len(geology_violation) > 0:
            violation_amount = (min_geology_distance - geology_violation['geology_distance']).sum()
            cost_penalty += violation_amount * 6

    # 水体距离约束惩罚
    if 'water_distance' in selected_data.columns:
        min_water_distance = constraints.get('min_water_distance', 1000)
        water_violation = selected_data[selected_data['water_distance'] < min_water_distance]
        if len(water_violation) > 0:
            violation_amount = (min_water_distance - water_violation['water_distance']).sum()
            cost_penalty += violation_amount * 7

    # 建设成本
    if 'cost' in selected_data.columns:
        construction_cost = selected_data['cost'].sum() * 0.01
        cost_penalty += construction_cost

    # 风场间距约束惩罚
    if 'min_farm_distance' in constraints and len(positions) > 1:
        coords = selected_data[['lat', 'lon']].values
        if len(coords) > 1:
            distances = pdist(coords) * 111000  # 转换为米（近似）
            min_distance = distances.min() if len(distances) > 0 else 0
            min_required = constraints['min_farm_distance']
            if min_distance < min_required:
                cost_penalty += (min_required - min_distance) * 100

    cost_weight = constraints.get('cost_weight', 0.5)

    # 最终适应度 = 综合评分 - 成本惩罚
    fitness = composite_score * 1000 - cost_weight * cost_penalty  # 缩放综合评分

    return max(fitness, 0)  # 确保适应度非负


def calculate_composite_fitness(positions, df, wind_speed_weight=0.6, utilization_weight=0.4, **constraints):
    """兼容旧版本的适应度函数，可选择是否包含储能优化"""
    if constraints.get('enable_storage_optimization', False):
        return calculate_composite_fitness_with_storage(positions, df, wind_speed_weight, utilization_weight,
                                                        **constraints)
    else:
        # 原版的适应度计算（不含储能优化）
        if len(positions) == 0:
            return 0

        selected_data = df.loc[positions]

        if 'predicted_wind_speed' in selected_data.columns:
            wind_speeds = selected_data['predicted_wind_speed']
            max_wind_speed = df['predicted_wind_speed'].max()
            normalized_wind_speed = wind_speeds.sum() / (len(wind_speeds) * max_wind_speed) if max_wind_speed > 0 else 0

            if 'wind_utilization_rate' in selected_data.columns:
                utilization_scores = selected_data['wind_utilization_rate']
                max_utilization = df['wind_utilization_rate'].max()
                normalized_utilization = utilization_scores.sum() / (
                        len(utilization_scores) * max_utilization) if max_utilization > 0 else 0
            else:
                normalized_utilization = 0

            composite_score = (wind_speed_weight * normalized_wind_speed +
                               utilization_weight * normalized_utilization)
        else:
            composite_score = 0

        cost_penalty = 0
        if 'slope' in selected_data.columns:
            max_slope = constraints.get('max_slope', 15)
            slope_violation = selected_data[selected_data['slope'] > max_slope]['slope'].sum()
            cost_penalty += slope_violation * 10

        cost_weight = constraints.get('cost_weight', 0.5)
        fitness = composite_score * 1000 - cost_weight * cost_penalty

        return max(fitness, 0)



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
    utilization_rates = []

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

        # 风能利用率
        if 'wind_utilization_rate' in turbine:
            utilization_rates.append(turbine['wind_utilization_rate'])
        else:
            # 简化计算利用率
            utilization = 1.0 if 3 <= wind_speed <= 25 else 0.5
            utilization_rates.append(utilization)

    total_annual_generation = sum(annual_generation_per_turbine)
    avg_capacity_factor = np.mean(capacity_factors) if capacity_factors else 0
    avg_utilization_rate = np.mean(utilization_rates) if utilization_rates else 0
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
        'average_utilization_rate': avg_utilization_rate,
        'equivalent_full_load_hours': equivalent_full_load_hours,
        'annual_generation_per_turbine': annual_generation_per_turbine,
        'capacity_factors': capacity_factors,
        'utilization_rates': utilization_rates,
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


def real_genetic_algorithm_with_storage(df, n_turbines, pop_size=50, generations=100,
                                        mutation_rate=0.1, crossover_rate=0.8, **kwargs):
    """包含储能优化的遗传算法"""
    start_time = time.time()

    # 是否启用储能优化
    enable_storage_opt = kwargs.get('enable_storage_optimization', False)

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    n_points = len(valid_points)
    fitness_history = []
    best_fitness_history = []

    if enable_storage_opt:
        # 扩展个体编码：包含储能容量和功率
        # 前n_turbines个基因是风机位置，后2个基因是储能容量和功率
        individual_length = n_turbines + 2
    else:
        individual_length = n_turbines

    # 初始化种群
    population = []
    for _ in range(pop_size):
        if enable_storage_opt:
            # 风机位置（离散）
            turbine_genes = np.random.choice(valid_points.index, n_turbines, replace=False)
            # 储能容量和功率（连续）
            storage_capacity = np.random.uniform(kwargs.get('min_storage_capacity', 10000),
                                                 kwargs.get('max_storage_capacity', 200000))
            storage_power = np.random.uniform(kwargs.get('min_storage_power', 5000),
                                              kwargs.get('max_storage_power', 100000))
            individual = np.concatenate([turbine_genes, [storage_capacity, storage_power]])
        else:
            individual = np.random.choice(valid_points.index, n_turbines, replace=False)
        population.append(individual)

    best_fitness = -float('inf')
    best_individual = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    for generation in range(generations):
        # 计算适应度
        fitness_scores = []
        for individual in population:
            if enable_storage_opt:
                turbine_positions = individual[:n_turbines].astype(int)
                storage_capacity = individual[n_turbines]
                storage_power = individual[n_turbines + 1]

                # 添加储能参数到约束中
                current_constraints = kwargs.copy()
                current_constraints['storage_capacity'] = storage_capacity
                current_constraints['storage_power'] = storage_power
                current_constraints['enable_storage_optimization'] = True

                fitness = calculate_composite_fitness_with_storage(
                    turbine_positions, df, **current_constraints
                )
            else:
                fitness = calculate_composite_fitness(individual, df, **kwargs)
            fitness_scores.append(fitness)

        # 记录历史
        current_best_fitness = max(fitness_scores)
        best_fitness_history.append(current_best_fitness)
        avg_fitness = np.mean(fitness_scores)
        fitness_history.append(avg_fitness)

        # 更新全局最优
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[np.argmax(fitness_scores)].copy()

        # 选择（轮盘赌选择）
        fitness_scores = np.array(fitness_scores)
        if fitness_scores.min() < 0:
            fitness_scores = fitness_scores - fitness_scores.min() + 1e-6
        selection_probs = fitness_scores / fitness_scores.sum()

        new_population = []
        for _ in range(pop_size):
            parent_idx = np.random.choice(len(population), p=selection_probs)
            new_population.append(population[parent_idx].copy())

        # 交叉
        for i in range(0, len(new_population), 2):
            if i + 1 < len(new_population) and np.random.random() < crossover_rate:
                parent1 = new_population[i]
                parent2 = new_population[i + 1]

                # 对风机位置进行交叉
                if enable_storage_opt:
                    crossover_point = np.random.randint(1, n_turbines - 1)
                    child1_genes = np.concatenate([parent1[:crossover_point], parent2[crossover_point:n_turbines]])
                    child2_genes = np.concatenate([parent2[:crossover_point], parent1[crossover_point:n_turbines]])

                    # 对储能参数进行算术交叉
                    alpha = np.random.random()
                    storage1 = alpha * parent1[n_turbines:] + (1 - alpha) * parent2[n_turbines:]
                    storage2 = alpha * parent2[n_turbines:] + (1 - alpha) * parent1[n_turbines:]

                    child1 = np.concatenate([child1_genes, storage1])
                    child2 = np.concatenate([child2_genes, storage2])
                else:
                    crossover_point = np.random.randint(1, n_turbines - 1)
                    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

                # 确保风机位置不重复
                if enable_storage_opt:
                    turbine_genes1 = child1[:n_turbines].astype(int)
                    turbine_genes2 = child2[:n_turbines].astype(int)

                    # 去重并补充
                    unique1 = np.unique(turbine_genes1)
                    while len(unique1) < n_turbines:
                        new_gene = np.random.choice(valid_points.index)
                        if new_gene not in unique1:
                            unique1 = np.append(unique1, new_gene)

                    unique2 = np.unique(turbine_genes2)
                    while len(unique2) < n_turbines:
                        new_gene = np.random.choice(valid_points.index)
                        if new_gene not in unique2:
                            unique2 = np.append(unique2, new_gene)

                    child1[:n_turbines] = unique1[:n_turbines]
                    child2[:n_turbines] = unique2[:n_turbines]
                else:
                    child1 = np.unique(child1)
                    child2 = np.unique(child2)

                    while len(child1) < n_turbines:
                        new_gene = np.random.choice(valid_points.index)
                        if new_gene not in child1:
                            child1 = np.append(child1, new_gene)

                    while len(child2) < n_turbines:
                        new_gene = np.random.choice(valid_points.index)
                        if new_gene not in child2:
                            child2 = np.append(child2, new_gene)

                new_population[i] = child1[:individual_length]
                new_population[i + 1] = child2[:individual_length]

        # 变异
        for i in range(len(new_population)):
            if np.random.random() < mutation_rate:
                individual = new_population[i]
                if enable_storage_opt:
                    # 随机选择变异类型：风机位置变异或储能参数变异
                    if np.random.random() < 0.7:  # 70%概率变异风机位置
                        mutation_point = np.random.randint(n_turbines)
                        new_gene = np.random.choice(valid_points.index)
                        while new_gene in individual[:n_turbines]:
                            new_gene = np.random.choice(valid_points.index)
                        individual[mutation_point] = new_gene
                    else:  # 30%概率变异储能参数
                        mutation_point = n_turbines + np.random.randint(2)
                        if mutation_point == n_turbines:  # 变异容量
                            min_cap = kwargs.get('min_storage_capacity', 10000)
                            max_cap = kwargs.get('max_storage_capacity', 200000)
                            individual[mutation_point] = np.random.uniform(min_cap, max_cap)
                        else:  # 变异功率
                            min_pow = kwargs.get('min_storage_power', 5000)
                            max_pow = kwargs.get('max_storage_power', 100000)
                            individual[mutation_point] = np.random.uniform(min_pow, max_pow)
                else:
                    mutation_point = np.random.randint(n_turbines)
                    new_gene = np.random.choice(valid_points.index)
                    while new_gene in individual:
                        new_gene = np.random.choice(valid_points.index)
                    individual[mutation_point] = new_gene

        population = new_population

        # 更新进度
        progress = (generation + 1) / generations
        progress_bar.progress(progress)
        if enable_storage_opt:
            status_text.text(
                f"储能优化遗传算法进度: {generation + 1}/{generations} 代, 当前最优适应度: {current_best_fitness:.2f}")
        else:
            status_text.text(
                f"遗传算法进度: {generation + 1}/{generations} 代, 当前最优适应度: {current_best_fitness:.2f}")

    progress_bar.empty()
    status_text.empty()

    computation_time = time.time() - start_time

    # 提取最优解
    if enable_storage_opt and best_individual is not None:
        best_turbine_positions = best_individual[:n_turbines].astype(int).tolist()
        best_storage_capacity = best_individual[n_turbines]
        best_storage_power = best_individual[n_turbines + 1]
    else:
        best_turbine_positions = best_individual.tolist() if best_individual is not None else []
        best_storage_capacity = kwargs.get('storage_capacity', 0)
        best_storage_power = kwargs.get('storage_power', 0)

    # 计算真实的最优位置数据
    best_positions_data = df.loc[best_turbine_positions] if len(best_turbine_positions) > 0 else pd.DataFrame()

    # 计算真实的发电量
    power_results = calculate_real_power_generation(best_positions_data)

    # 计算储能经济性
    storage_economic_analysis = {}
    if enable_storage_opt:
        storage_investment = calculate_storage_investment_cost(best_storage_capacity, best_storage_power)
        storage_annual_revenue = calculate_storage_annual_revenue(
            best_positions_data, best_storage_capacity, best_storage_power, kwargs
        )
        storage_om_cost = calculate_storage_operation_cost(best_storage_capacity, best_storage_power)
        storage_net_benefit = storage_annual_revenue - storage_om_cost
        storage_payback = storage_investment / storage_net_benefit if storage_net_benefit > 0 else float('inf')

        storage_economic_analysis = {
            'storage_capacity_kwh': best_storage_capacity,
            'storage_power_kw': best_storage_power,
            'storage_investment': storage_investment,
            'storage_annual_revenue': storage_annual_revenue,
            'storage_om_cost': storage_om_cost,
            'storage_net_benefit': storage_net_benefit,
            'storage_payback_years': storage_payback
        }

    # 添加权重信息到结果中
    result = {
        'best_positions': best_turbine_positions,
        'best_positions_data': best_positions_data,
        'best_fitness': best_fitness,
        'fitness_history': best_fitness_history,
        'algorithm': '遗传算法（含储能优化）' if enable_storage_opt else '遗传算法',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs),
        'optimization_weights': {
            'wind_speed_weight': kwargs.get('wind_speed_weight', 0.6),
            'utilization_weight': kwargs.get('utilization_weight', 0.4),
            'storage_weight': kwargs.get('storage_weight', 0.3) if enable_storage_opt else 0
        },
        'n_farms': kwargs.get('n_farms', 1),
        'n_turbines_per_farm': n_turbines // kwargs.get('n_farms', 1),
        'enable_storage_optimization': enable_storage_opt,
        'storage_economic_analysis': storage_economic_analysis
    }

    return result


def real_genetic_algorithm(df, n_turbines, pop_size=50, generations=100,
                           mutation_rate=0.1, crossover_rate=0.8, **kwargs):
    """原始的遗传算法实现 - 调用新版本的函数但不启用储能优化"""
    kwargs['enable_storage_optimization'] = False
    return real_genetic_algorithm_with_storage(df, n_turbines, pop_size, generations,
                                               mutation_rate, crossover_rate, **kwargs)


def real_simulated_annealing(df, n_turbines, **kwargs):
    """真实的模拟退火算法 - 使用综合适应度函数"""
    start_time = time.time()

    enable_storage_opt = kwargs.get('enable_storage_optimization', False)

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    # 初始解
    if enable_storage_opt:
        # 初始解包含储能参数
        current_turbine_solution = np.random.choice(valid_points.index, n_turbines, replace=False)
        current_storage_capacity = np.random.uniform(kwargs.get('min_storage_capacity', 10000),
                                                     kwargs.get('max_storage_capacity', 200000))
        current_storage_power = np.random.uniform(kwargs.get('min_storage_power', 5000),
                                                  kwargs.get('max_storage_power', 100000))
        current_solution = (current_turbine_solution, current_storage_capacity, current_storage_power)
    else:
        current_solution = np.random.choice(valid_points.index, n_turbines, replace=False)

    # 计算初始适应度
    if enable_storage_opt:
        current_constraints = kwargs.copy()
        current_constraints['storage_capacity'] = current_storage_capacity
        current_constraints['storage_power'] = current_storage_power
        current_constraints['enable_storage_optimization'] = True
        current_fitness = calculate_composite_fitness_with_storage(
            current_turbine_solution, df, **current_constraints
        )
    else:
        current_fitness = calculate_composite_fitness(current_solution, df, **kwargs)

    best_solution = current_solution
    best_fitness = current_fitness

    initial_temp = kwargs.get('initial_temp', 1000)
    cooling_rate = kwargs.get('cooling_rate', 0.95)
    iterations_per_temp = kwargs.get('iterations_per_temp', 50)

    temperature = initial_temp
    fitness_history = [current_fitness]

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_iterations = int(np.log(0.01) / np.log(cooling_rate)) * iterations_per_temp
    current_iteration = 0

    while temperature > 1e-3:
        for _ in range(iterations_per_temp):
            if enable_storage_opt:
                # 生成邻域解
                current_turbines, current_capacity, current_power = current_solution
                neighbor_turbines = current_turbines.copy()

                # 随机决定变异类型
                if np.random.random() < 0.7:  # 70%概率变异风机位置
                    mutation_point = np.random.randint(n_turbines)
                    new_gene = np.random.choice(valid_points.index)
                    while new_gene in neighbor_turbines:
                        new_gene = np.random.choice(valid_points.index)
                    neighbor_turbines[mutation_point] = new_gene
                    neighbor_capacity = current_capacity
                    neighbor_power = current_power
                else:  # 30%概率变异储能参数
                    neighbor_turbines = current_turbines.copy()
                    if np.random.random() < 0.5:  # 变异容量
                        neighbor_capacity = current_capacity + np.random.normal(0, current_capacity * 0.1)
                        neighbor_capacity = max(kwargs.get('min_storage_capacity', 10000),
                                                min(kwargs.get('max_storage_capacity', 200000), neighbor_capacity))
                        neighbor_power = current_power
                    else:  # 变异功率
                        neighbor_power = current_power + np.random.normal(0, current_power * 0.1)
                        neighbor_power = max(kwargs.get('min_storage_power', 5000),
                                             min(kwargs.get('max_storage_power', 100000), neighbor_power))
                        neighbor_capacity = current_capacity

                neighbor_solution = (neighbor_turbines, neighbor_capacity, neighbor_power)

                # 计算邻域解适应度
                neighbor_constraints = kwargs.copy()
                neighbor_constraints['storage_capacity'] = neighbor_capacity
                neighbor_constraints['storage_power'] = neighbor_power
                neighbor_constraints['enable_storage_optimization'] = True
                neighbor_fitness = calculate_composite_fitness_with_storage(
                    neighbor_turbines, df, **neighbor_constraints
                )
            else:
                # 生成邻域解
                neighbor = current_solution.copy()
                mutation_point = np.random.randint(n_turbines)
                new_gene = np.random.choice(valid_points.index)
                while new_gene in neighbor:
                    new_gene = np.random.choice(valid_points.index)
                neighbor[mutation_point] = new_gene

                neighbor_fitness = calculate_composite_fitness(neighbor, df, **kwargs)
                neighbor_solution = neighbor

            # 决定是否接受新解
            if neighbor_fitness > current_fitness:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                if neighbor_fitness > best_fitness:
                    best_solution = neighbor_solution
                    best_fitness = neighbor_fitness
            else:
                delta = neighbor_fitness - current_fitness
                acceptance_prob = np.exp(delta / temperature)
                if np.random.random() < acceptance_prob:
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness

            fitness_history.append(current_fitness)
            current_iteration += 1

            # 更新进度
            if current_iteration % 10 == 0:
                progress = min(1.0, current_iteration / total_iterations)
                progress_bar.progress(progress)
                status_text.text(
                    f"模拟退火进度: {current_iteration}/{total_iterations}, 温度: {temperature:.2f}, 最优适应度: {best_fitness:.2f}")

        temperature *= cooling_rate

    progress_bar.empty()
    status_text.empty()

    computation_time = time.time() - start_time

    # 提取最优解
    if enable_storage_opt:
        best_turbines, best_capacity, best_power = best_solution
        best_positions_data = df.loc[best_turbines]

        # 计算储能经济性
        storage_investment = calculate_storage_investment_cost(best_capacity, best_power)
        storage_annual_revenue = calculate_storage_annual_revenue(
            best_positions_data, best_capacity, best_power, kwargs
        )
        storage_om_cost = calculate_storage_operation_cost(best_capacity, best_power)
        storage_net_benefit = storage_annual_revenue - storage_om_cost
        storage_payback = storage_investment / storage_net_benefit if storage_net_benefit > 0 else float('inf')

        storage_economic_analysis = {
            'storage_capacity_kwh': best_capacity,
            'storage_power_kw': best_power,
            'storage_investment': storage_investment,
            'storage_annual_revenue': storage_annual_revenue,
            'storage_om_cost': storage_om_cost,
            'storage_net_benefit': storage_net_benefit,
            'storage_payback_years': storage_payback
        }
    else:
        best_turbines = best_solution
        best_positions_data = df.loc[best_turbines]
        storage_economic_analysis = {}

    power_results = calculate_real_power_generation(best_positions_data)

    return {
        'best_positions': best_turbines.tolist(),
        'best_positions_data': best_positions_data,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'algorithm': '模拟退火算法（含储能优化）' if enable_storage_opt else '模拟退火算法',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs),
        'optimization_weights': {
            'wind_speed_weight': kwargs.get('wind_speed_weight', 0.6),
            'utilization_weight': kwargs.get('utilization_weight', 0.4),
            'storage_weight': kwargs.get('storage_weight', 0.3) if enable_storage_opt else 0
        },
        'enable_storage_optimization': enable_storage_opt,
        'storage_economic_analysis': storage_economic_analysis
    }


def real_particle_swarm(df, n_turbines, pop_size=30, generations=100,
                        w=0.7, c1=1.5, c2=1.5, **kwargs):
    """真实的粒子群优化算法 - 使用综合适应度函数"""
    start_time = time.time()

    enable_storage_opt = kwargs.get('enable_storage_optimization', False)

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    n_points = len(valid_points)

    if enable_storage_opt:
        # 扩展粒子维度：包含储能容量和功率
        dim = n_turbines + 2  # 风机位置 + 储能容量 + 储能功率
        # 定义边界
        bounds = []
        # 风机位置边界
        for _ in range(n_turbines):
            bounds.append([0, n_points - 1])
        # 储能容量边界
        bounds.append([kwargs.get('min_storage_capacity', 10000),
                       kwargs.get('max_storage_capacity', 200000)])
        # 储能功率边界
        bounds.append([kwargs.get('min_storage_power', 5000),
                       kwargs.get('max_storage_power', 100000)])
    else:
        dim = n_turbines
        bounds = [[0, n_points - 1] for _ in range(n_turbines)]

    # 初始化粒子群
    particles = []
    velocities = []
    personal_best_positions = []
    personal_best_fitnesses = []

    for _ in range(pop_size):
        # 初始化粒子位置
        if enable_storage_opt:
            position = []
            # 随机选择风机位置（离散）
            turbine_indices = np.random.choice(valid_points.index, n_turbines, replace=False)
            position.extend(turbine_indices)
            # 随机初始化储能参数
            position.append(np.random.uniform(bounds[n_turbines][0], bounds[n_turbines][1]))
            position.append(np.random.uniform(bounds[n_turbines + 1][0], bounds[n_turbines + 1][1]))
            position = np.array(position)
        else:
            position = np.random.choice(valid_points.index, n_turbines, replace=False)

        particles.append(position)
        velocities.append(np.zeros(dim))
        personal_best_positions.append(position.copy())

        # 计算适应度
        if enable_storage_opt:
            turbine_positions = position[:n_turbines].astype(int)
            storage_capacity = position[n_turbines]
            storage_power = position[n_turbines + 1]

            current_constraints = kwargs.copy()
            current_constraints['storage_capacity'] = storage_capacity
            current_constraints['storage_power'] = storage_power
            current_constraints['enable_storage_optimization'] = True

            fitness = calculate_composite_fitness_with_storage(
                turbine_positions, df, **current_constraints
            )
        else:
            fitness = calculate_composite_fitness(position, df, **kwargs)

        personal_best_fitnesses.append(fitness)

    # 全局最优
    global_best_idx = np.argmax(personal_best_fitnesses)
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_fitness = personal_best_fitnesses[global_best_idx]

    fitness_history = [global_best_fitness]

    progress_bar = st.progress(0)
    status_text = st.empty()

    for generation in range(generations):
        for i in range(pop_size):
            # 更新粒子位置
            for d in range(dim):
                # PSO速度更新公式
                r1, r2 = np.random.random(), np.random.random()
                velocities[i][d] = (w * velocities[i][d] +
                                    c1 * r1 * (personal_best_positions[i][d] - particles[i][d]) +
                                    c2 * r2 * (global_best_position[d] - particles[i][d]))

                # 位置更新
                particles[i][d] = particles[i][d] + velocities[i][d]

                # 应用边界约束
                particles[i][d] = max(bounds[d][0], min(bounds[d][1], particles[i][d]))

            # 确保风机位置不重复（仅对前n_turbines维度）
            if enable_storage_opt:
                turbine_positions = particles[i][:n_turbines].copy()
                # 将连续值转换为离散索引
                discrete_indices = []
                for j in range(n_turbines):
                    idx = int(np.clip(turbine_positions[j], 0, n_points - 1))
                    discrete_indices.append(idx)

                # 去重处理
                unique_indices = np.unique(discrete_indices)
                while len(unique_indices) < n_turbines:
                    new_idx = np.random.randint(0, n_points)
                    if new_idx not in unique_indices:
                        unique_indices = np.append(unique_indices, new_idx)

                particles[i][:n_turbines] = unique_indices[:n_turbines]

            # 计算适应度
            if enable_storage_opt:
                turbine_positions = particles[i][:n_turbines].astype(int)
                storage_capacity = particles[i][n_turbines]
                storage_power = particles[i][n_turbines + 1]

                current_constraints = kwargs.copy()
                current_constraints['storage_capacity'] = storage_capacity
                current_constraints['storage_power'] = storage_power
                current_constraints['enable_storage_optimization'] = True

                current_fitness = calculate_composite_fitness_with_storage(
                    turbine_positions, df, **current_constraints
                )
            else:
                # 确保位置是整数
                int_positions = particles[i].astype(int)
                # 去重处理
                unique_positions = np.unique(int_positions)
                while len(unique_positions) < n_turbines:
                    new_idx = np.random.randint(0, n_points)
                    if new_idx not in unique_positions:
                        unique_positions = np.append(unique_positions, new_idx)

                particles[i] = unique_positions[:n_turbines]
                current_fitness = calculate_composite_fitness(particles[i], df, **kwargs)

            # 更新个体最优
            if current_fitness > personal_best_fitnesses[i]:
                personal_best_positions[i] = particles[i].copy()
                personal_best_fitnesses[i] = current_fitness

                # 更新全局最优
                if current_fitness > global_best_fitness:
                    global_best_position = particles[i].copy()
                    global_best_fitness = current_fitness

        fitness_history.append(global_best_fitness)

        # 更新进度
        progress = (generation + 1) / generations
        progress_bar.progress(progress)
        status_text.text(f"粒子群进度: {generation + 1}/{generations}, 最优适应度: {global_best_fitness:.2f}")

    progress_bar.empty()
    status_text.empty()

    computation_time = time.time() - start_time

    # 提取最优解
    if enable_storage_opt:
        best_turbine_positions = global_best_position[:n_turbines].astype(int).tolist()
        best_storage_capacity = global_best_position[n_turbines]
        best_storage_power = global_best_position[n_turbines + 1]
        best_positions_data = df.loc[best_turbine_positions]

        # 计算储能经济性
        storage_investment = calculate_storage_investment_cost(best_storage_capacity, best_storage_power)
        storage_annual_revenue = calculate_storage_annual_revenue(
            best_positions_data, best_storage_capacity, best_storage_power, kwargs
        )
        storage_om_cost = calculate_storage_operation_cost(best_storage_capacity, best_storage_power)
        storage_net_benefit = storage_annual_revenue - storage_om_cost
        storage_payback = storage_investment / storage_net_benefit if storage_net_benefit > 0 else float('inf')

        storage_economic_analysis = {
            'storage_capacity_kwh': best_storage_capacity,
            'storage_power_kw': best_storage_power,
            'storage_investment': storage_investment,
            'storage_annual_revenue': storage_annual_revenue,
            'storage_om_cost': storage_om_cost,
            'storage_net_benefit': storage_net_benefit,
            'storage_payback_years': storage_payback
        }
    else:
        best_turbine_positions = global_best_position.tolist()
        best_positions_data = df.loc[best_turbine_positions]
        storage_economic_analysis = {}

    power_results = calculate_real_power_generation(best_positions_data)

    return {
        'best_positions': best_turbine_positions,
        'best_positions_data': best_positions_data,
        'best_fitness': global_best_fitness,
        'fitness_history': fitness_history,
        'algorithm': '粒子群优化算法（含储能优化）' if enable_storage_opt else '粒子群优化算法',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs),
        'optimization_weights': {
            'wind_speed_weight': kwargs.get('wind_speed_weight', 0.6),
            'utilization_weight': kwargs.get('utilization_weight', 0.4),
            'storage_weight': kwargs.get('storage_weight', 0.3) if enable_storage_opt else 0
        },
        'enable_storage_optimization': enable_storage_opt,
        'storage_economic_analysis': storage_economic_analysis
    }


def real_pulp_optimization(df, n_turbines, solver_type="CBC", time_limit=60, **kwargs):
    """使用PuLP进行数学规划优化 - 使用综合评分"""
    start_time = time.time()

    enable_storage_opt = kwargs.get('enable_storage_optimization', False)

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    # 创建问题
    prob = pulp.LpProblem("WindFarm_Optimization", pulp.LpMaximize)

    # 决策变量：是否选择该点位
    x = pulp.LpVariable.dicts("x", valid_points.index, cat='Binary')

    # 目标函数：最大化综合评分
    composite_terms = []
    cost_terms = []

    wind_speed_weight = kwargs.get('wind_speed_weight', 0.6)
    utilization_weight = kwargs.get('utilization_weight', 0.4)

    # 预计算最大值为归一化
    max_wind_speed = valid_points['predicted_wind_speed'].max() if 'predicted_wind_speed' in valid_points.columns else 1
    max_utilization = valid_points[
        'wind_utilization_rate'].max() if 'wind_utilization_rate' in valid_points.columns else 1

    for idx, point in valid_points.iterrows():
        # 风速得分
        wind_speed = point.get('predicted_wind_speed', 0)
        wind_score = (wind_speed / max_wind_speed) * wind_speed_weight if max_wind_speed > 0 else 0

        # 利用率得分
        if 'wind_utilization_rate' in point:
            utilization_score = (point[
                                     'wind_utilization_rate'] / max_utilization) * utilization_weight if max_utilization > 0 else 0
        else:
            utilization_score = 0

        composite_score = wind_score + utilization_score
        composite_terms.append(composite_score * x[idx])

        # 成本项
        cost_value = 0
        if point.get('slope', 0) > kwargs.get('max_slope', 15):
            cost_value += point['slope'] * 10
        cost_terms.append(cost_value * x[idx])

    # 目标函数
    cost_weight = kwargs.get('cost_weight', 0.5)
    prob += pulp.lpSum(composite_terms) - cost_weight * pulp.lpSum(cost_terms)

    # 约束：选择恰好n_turbines个点位
    prob += pulp.lpSum([x[i] for i in valid_points.index]) == n_turbines

    # 求解
    if solver_type == "CBC":
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit)
    elif solver_type == "GLPK":
        solver = pulp.GLPK_CMD(timeLimit=time_limit)
    else:
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit)

    prob.solve(solver)

    # 提取结果
    selected_positions = []
    for idx in valid_points.index:
        if pulp.value(x[idx]) == 1:
            selected_positions.append(idx)

    best_fitness = pulp.value(prob.objective)
    computation_time = time.time() - start_time

    best_positions_data = df.loc[selected_positions]
    power_results = calculate_real_power_generation(best_positions_data)

    # 对于PuLP优化，储能优化需要单独处理（因为PuLP主要处理离散变量）
    storage_economic_analysis = {}
    if enable_storage_opt:
        # 可以在后续步骤中优化储能参数
        st.info("PuLP求解器主要用于离散优化，储能优化建议使用遗传算法或粒子群算法")

    return {
        'best_positions': selected_positions,
        'best_positions_data': best_positions_data,
        'best_fitness': best_fitness if best_fitness else 0,
        'fitness_history': [best_fitness] if best_fitness else [0],
        'algorithm': 'PuLP优化求解器（含储能优化）' if enable_storage_opt else 'PuLP优化求解器',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs),
        'optimization_weights': {
            'wind_speed_weight': wind_speed_weight,
            'utilization_weight': utilization_weight,
            'storage_weight': kwargs.get('storage_weight', 0.3) if enable_storage_opt else 0
        },
        'enable_storage_optimization': enable_storage_opt,
        'storage_economic_analysis': storage_economic_analysis
    }


def check_constraints_violations(positions_data, constraints):
    """检查约束违反情况"""
    violations = {}

    if positions_data.empty:
        return violations

    if 'slope' in positions_data.columns and 'max_slope' in constraints:
        slope_violations = positions_data[positions_data['slope'] > constraints['max_slope']]
        violations['slope'] = len(slope_violations)

    if 'road_distance' in positions_data.columns and 'max_road_distance' in constraints:
        road_violations = positions_data[positions_data['road_distance'] > constraints['max_road_distance']]
        violations['road'] = len(road_violations)

    # 添加其他约束检查...

    return violations


def call_optimize_function(df, algo, algorithm_params):
    """调用真实优化函数"""

    # 参数映射和转换
    optimization_params = algorithm_params.copy()

    # 处理风场数量相关的参数
    if 'total_turbines' in optimization_params:
        # 多风场优化：使用总风机数量
        optimization_params['n_turbines'] = optimization_params['total_turbines']
    elif 'n_turbines_per_farm' in optimization_params:
        # 单风场优化：使用单场风机数量
        optimization_params['n_turbines'] = optimization_params['n_turbines_per_farm']

    # 移除可能冲突的参数
    optimization_params.pop('n_farms', None)
    optimization_params.pop('n_turbines_per_farm', None)
    optimization_params.pop('total_turbines', None)
    optimization_params.pop('min_farm_distance', None)

    try:
        if algo == "遗传算法":
            result = real_genetic_algorithm_with_storage(df, **optimization_params)
        elif algo == "模拟退火算法":
            result = real_simulated_annealing(df, **optimization_params)
        elif algo == "粒子群优化算法":
            result = real_particle_swarm(df, **optimization_params)
        elif algo == "PuLP优化求解器":
            result = real_pulp_optimization(df, **optimization_params)
        else:
            result = real_genetic_algorithm_with_storage(df, **optimization_params)

        return result

    except Exception as e:
        st.error(f"优化算法执行错误: {str(e)}")
        # 回退到基础遗传算法
        st.info("尝试使用基础参数重新计算...")
        base_params = {
            'n_turbines': optimization_params.get('n_turbines', 5),
            'pop_size': 30,
            'generations': 50,
            'enable_storage_optimization': False
        }
        return real_genetic_algorithm(df, **base_params)


def call_optimize_function_with_all_strategies(df, algo, algorithm_params):
    """
    调用优化函数并测试所有储能策略 - 支持多风场
    """
    try:
        # 测试不同的储能策略
        strategies = ['平滑输出', '削峰填谷', '混合模式']
        strategy_results = []

        best_result = None
        best_fitness = -1
        best_strategy = None

        # 创建进度条和状态文本
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, strategy in enumerate(strategies):
            # 更新进度状态
            current_progress = (i + 1) / len(strategies)
            status_text.text(f"🔄 正在测试储能策略: {strategy} ({i + 1}/{len(strategies)})")
            progress_bar.progress(current_progress)

            # 更新策略参数
            current_params = algorithm_params.copy()
            current_params['storage_strategy'] = strategy

            # 调用优化函数
            result = call_optimize_function(df, algo, current_params)

            # 为每个风场生成独立的储能调度数据
            if 'best_positions' in result and len(result['best_positions']) > 0:
                # 获取风场数量
                n_farms = algorithm_params.get('n_farms', 1)
                n_turbines_per_farm = algorithm_params.get('n_turbines_per_farm',
                                                           len(result['best_positions']) // n_farms)

                # 分割风场数据
                farm_storage_results = []
                for farm_idx in range(n_farms):
                    # 计算当前风场的风机位置
                    start_idx = farm_idx * n_turbines_per_farm
                    end_idx = start_idx + n_turbines_per_farm
                    farm_positions = result['best_positions'][start_idx:end_idx]

                    # 为当前风场生成储能调度数据
                    storage_params = {
                        'storage_capacity': result.get('storage_economic_analysis', {}).get('storage_capacity_kwh',
                                                                                            60000),
                        'storage_power': result.get('storage_economic_analysis', {}).get('storage_power_kw', 30000),
                        'grid_capacity': current_params.get('grid_capacity', 20000),
                        'storage_strategy': strategy
                    }

                    farm_storage = generate_storage_schedule_data(df, farm_positions, storage_params)
                    farm_storage_results.append(farm_storage)

                # 存储所有风场的储能结果
                result['storage_results'] = farm_storage_results
                result['n_farms'] = n_farms

            # 记录策略结果
            fitness = result.get('best_fitness', 0)
            strategy_results.append({
                'strategy': strategy,
                'fitness': fitness,
                'computation_time': result.get('computation_time', 0),
                'quality_rating': evaluate_solution_quality(fitness)
            })

            # 更新最佳结果
            if fitness > best_fitness:
                best_fitness = fitness
                best_result = result
                best_strategy = strategy

        # 清理进度显示
        progress_bar.empty()
        status_text.empty()

        # 将策略比较结果添加到最佳结果中
        if best_result:
            best_result['strategy_comparison'] = strategy_results
            best_result['best_strategy'] = best_strategy
            best_result['best_fitness'] = best_fitness

            st.success(f"🏆 最佳储能策略: {best_strategy} (适应度: {best_fitness:.3f})")

        return best_result

    except Exception as e:
        st.error(f"多策略优化失败: {str(e)}")
        return call_optimize_function(df, algo, algorithm_params)


def evaluate_solution_quality(fitness):
    """
    简单评估解决方案质量
    """
    if fitness >= 900:
        return "🎯 优秀"
    elif fitness >= 800:
        return "🟢 良好"
    elif fitness >= 700:
        return "🟡 一般"
    else:
        return "🔴 需要改进"


def generate_storage_schedule_data(df, selected_positions, storage_params):
    """
    生成储能调度数据用于可视化
    """
    try:
        # 获取选中的风电场数据
        selected_data = df.loc[selected_positions]

        # 计算总风电功率时间序列
        time_series_data = calculate_wind_power_time_series(df, selected_data)

        # 应用储能调度策略
        storage_results = apply_storage_strategy(time_series_data, storage_params)

        return storage_results
    except Exception as e:
        st.error(f"生成储能调度数据时出错: {str(e)}")
        # 返回空的调度数据
        return {
            'schedule_data': pd.DataFrame(),
            'performance_metrics': {},
            'storage_params': storage_params,
            'strategy': storage_params.get('storage_strategy', '未知')
        }


def calculate_wind_power_time_series(df, selected_data):
    """
    基于原始数据计算风电功率时间序列
    """

    # 风速转功率函数
    def wind_speed_to_power(wind_speed):
        cut_in, rated, cut_out = 3.0, 12.5, 25.0
        rated_power = 2500  # kW
        if wind_speed < cut_in or wind_speed > cut_out:
            return 0
        elif wind_speed < rated:
            return rated_power * ((wind_speed - cut_in) / (rated - cut_in)) ** 3
        else:
            return rated_power

    # 获取选中的坐标点
    selected_points = selected_data['point_id'].unique()

    # 按时间聚合计算总功率
    time_series = []

    # 假设每个坐标点有4台风机
    turbines_per_point = 4

    # 按时间戳分组计算
    df_sorted = df.sort_values('timestamp')

    for timestamp in df_sorted['timestamp'].unique():
        # 获取该时间点所有选中坐标的数据
        time_data = df_sorted[
            (df_sorted['timestamp'] == timestamp) &
            (df_sorted['point_id'].isin(selected_points))
            ]

        if len(time_data) > 0:
            # 计算总风电功率
            total_power = 0
            for _, row in time_data.iterrows():
                power_per_turbine = wind_speed_to_power(row['predicted_wind_speed'])
                total_power += power_per_turbine * turbines_per_point

            # 提取时间信息
            hour = row['hour'] if 'hour' in row else 0
            minute = row['minute'] if 'minute' in row else 0

            time_series.append({
                'timestamp': timestamp,
                'hour': hour,
                'minute': minute,
                'time_index': len(time_series),
                'wind_power': total_power,
                'wind_speed_avg': time_data['predicted_wind_speed'].mean()
            })

    return pd.DataFrame(time_series)


def apply_storage_strategy(time_series_data, storage_params):
    """
    应用储能调度策略
    """
    storage_capacity = storage_params.get('storage_capacity', 60000)  # kWh
    max_power = storage_params.get('storage_power', 30000)  # kW
    grid_capacity = storage_params.get('grid_capacity', 20000)  # kW
    strategy = storage_params.get('storage_strategy', '平滑输出')

    # 初始化变量
    wind_power = time_series_data['wind_power'].values
    n_periods = len(wind_power)

    battery_power = np.zeros(n_periods)  # 正值放电，负值充电
    soc = np.zeros(n_periods)  # 荷电状态 (0-1)
    grid_power = np.zeros(n_periods)  # 并网功率
    wind_curtailment = np.zeros(n_periods)  # 弃风功率

    # 初始SOC
    soc[0] = 0.5  # 50%初始电量

    # 根据策略选择参数
    if strategy == '平滑输出':
        smoothing_factor = 0.8
        peak_threshold = 0.9
    elif strategy == '削峰填谷':
        smoothing_factor = 0.6
        peak_threshold = 0.8
    else:  # 混合模式
        smoothing_factor = 0.7
        peak_threshold = 0.85

    for t in range(n_periods):
        current_wind_power = wind_power[t]

        # 计算功率差额
        power_diff = current_wind_power - grid_capacity

        if strategy == '平滑输出':
            # 平滑输出策略
            if power_diff > 0:  # 风电过剩
                # 充电
                charge_power = min(power_diff, max_power,
                                   storage_capacity * (0.9 - soc[t - 1]) / 0.95 if t > 0 else max_power)
                battery_power[t] = -charge_power
                grid_power[t] = grid_capacity
                wind_curtailment[t] = power_diff - charge_power
            else:  # 风电不足
                # 放电
                discharge_power = min(-power_diff, max_power,
                                      (soc[t - 1] - 0.1) * storage_capacity * 0.95 if t > 0 else max_power)
                battery_power[t] = discharge_power
                grid_power[t] = current_wind_power + discharge_power
                wind_curtailment[t] = 0

        elif strategy == '削峰填谷':
            # 削峰填谷策略
            if current_wind_power > grid_capacity * peak_threshold:  # 高峰时段
                charge_power = min(current_wind_power - grid_capacity * peak_threshold, max_power,
                                   storage_capacity * (0.9 - soc[t - 1]) / 0.95 if t > 0 else max_power)
                battery_power[t] = -charge_power
                grid_power[t] = grid_capacity * peak_threshold
                wind_curtailment[t] = current_wind_power - grid_capacity * peak_threshold - charge_power
            elif current_wind_power < grid_capacity * 0.6:  # 低谷时段
                discharge_power = min(grid_capacity * 0.6 - current_wind_power, max_power,
                                      (soc[t - 1] - 0.1) * storage_capacity * 0.95 if t > 0 else max_power)
                battery_power[t] = discharge_power
                grid_power[t] = current_wind_power + discharge_power
                wind_curtailment[t] = 0
            else:  # 正常时段
                battery_power[t] = 0
                grid_power[t] = current_wind_power
                wind_curtailment[t] = 0

        else:  # 混合模式
            # 混合模式：结合平滑输出和削峰填谷的优点
            if power_diff > 0:  # 风电过剩
                # 根据SOC决定充电策略
                if t > 0 and soc[t - 1] < 0.7:  # SOC较低时多充电
                    charge_power = min(power_diff * smoothing_factor, max_power,
                                       storage_capacity * (0.9 - soc[t - 1]) / 0.95)
                else:  # SOC较高时少充电
                    charge_power = min(power_diff * 0.5, max_power,
                                       storage_capacity * (0.9 - soc[t - 1]) / 0.95)

                battery_power[t] = -charge_power
                grid_power[t] = current_wind_power - charge_power
                wind_curtailment[t] = max(0, power_diff - charge_power)

            else:  # 风电不足
                # 根据SOC决定放电策略
                if t > 0 and soc[t - 1] > 0.4:  # SOC较高时多放电
                    discharge_power = min(-power_diff, max_power,
                                          (soc[t - 1] - 0.1) * storage_capacity * 0.95)
                else:  # SOC较低时少放电
                    discharge_power = min(-power_diff * 0.5, max_power,
                                          (soc[t - 1] - 0.1) * storage_capacity * 0.95)

                battery_power[t] = discharge_power
                grid_power[t] = current_wind_power + discharge_power
                wind_curtailment[t] = 0

        # 更新SOC
        if t > 0:
            energy_change = -battery_power[t] * (10 / 60)  # 10分钟间隔，转换为kWh
            soc[t] = max(0.1, min(0.9, soc[t - 1] + energy_change / storage_capacity))
        else:
            energy_change = -battery_power[t] * (10 / 60)
            soc[t] = max(0.1, min(0.9, 0.5 + energy_change / storage_capacity))

    # 创建结果数据框
    result_df = time_series_data.copy()
    result_df['battery_power'] = battery_power
    result_df['grid_power'] = grid_power
    result_df['storage_soc'] = soc
    result_df['wind_curtailment'] = wind_curtailment
    result_df['net_power'] = result_df['wind_power'] + result_df['battery_power']

    # 计算性能指标
    performance_metrics = calculate_storage_performance(result_df, storage_params)

    return {
        'schedule_data': result_df,
        'performance_metrics': performance_metrics,
        'storage_params': storage_params,
        'strategy': strategy
    }


def calculate_storage_performance(storage_data, storage_params):
    """
    计算储能系统性能指标
    """
    wind_power = storage_data['wind_power']
    grid_power = storage_data['grid_power']
    battery_power = storage_data['battery_power']
    storage_capacity = storage_params.get('storage_capacity', 60000)

    # 平滑效果
    wind_fluctuation = wind_power.std()
    grid_fluctuation = grid_power.std()
    smoothing_effect = ((wind_fluctuation - grid_fluctuation) / wind_fluctuation * 100) if wind_fluctuation > 0 else 0

    # 储能利用率
    total_charge = abs(storage_data[storage_data['battery_power'] < 0]['battery_power'].sum() * (10 / 60))
    total_discharge = storage_data[storage_data['battery_power'] > 0]['battery_power'].sum() * (10 / 60)
    storage_utilization = (total_charge + total_discharge) / (2 * storage_capacity) * 100

    # 弃风率
    total_wind_energy = wind_power.sum() * (10 / 60)
    total_curtailment = storage_data['wind_curtailment'].sum() * (10 / 60)
    curtailment_rate = (total_curtailment / total_wind_energy * 100) if total_wind_energy > 0 else 0

    # 系统效率（假设充放电效率为95%）
    system_efficiency = (total_discharge / total_charge * 100) if total_charge > 0 else 0

    return {
        'smoothing_effect': smoothing_effect,
        'storage_utilization': storage_utilization,
        'curtailment_rate': curtailment_rate,
        'system_efficiency': system_efficiency,
        'total_charge_energy': total_charge,
        'total_discharge_energy': total_discharge,
        'wind_fluctuation': wind_fluctuation,
        'grid_fluctuation': grid_fluctuation
    }



import numpy as np
import streamlit as st
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import Point
import os
import pandas as pd


def load_maale_gilboa_boundary():
    """加载Maale Gilboa区域边界数据"""
    geojson_path = r"C:\Users\lhl\Downloads\map (10).geojson"
    if not os.path.exists(geojson_path):
        return None

    try:
        gdf = gpd.read_file(geojson_path)
        return gdf
    except Exception as e:
        st.error(f"加载地图数据错误: {str(e)}")
        return None


def create_maale_gilboa_base_map():
    """创建Maale Gilboa基础地图"""
    maale_gilboa = load_maale_gilboa_boundary()
    if maale_gilboa is None:
        return None

    geometry = maale_gilboa.geometry.iloc[0]

    if geometry.geom_type == 'Polygon':
        polygons = [geometry]
    elif geometry.geom_type == 'MultiPolygon':
        polygons = list(geometry.geoms)
    else:
        return None

    centroid = geometry.centroid
    center_lat, center_lon = centroid.y, centroid.x

    # 计算边界框以确定合适的缩放级别
    bounds = geometry.bounds
    min_lon, min_lat, max_lon, max_lat = bounds

    return {
        'polygons': polygons,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'geometry': geometry,
        'bounds': bounds,
        'gdf': maale_gilboa  # 保留原始GeoDataFrame
    }


def preprocess_wind_data(df):
    """
    预处理风速数据，计算每个坐标点的24小时平均风速

    Parameters:
    - df: 原始数据框，包含24小时记录

    Returns:
    - df_avg: 包含每个坐标点平均风速的数据框
    """
    # 检查必要列是否存在
    required_columns = ['lat', 'lon', 'predicted_wind_speed', 'hour']
    if not all(col in df.columns for col in required_columns):
        st.error(f"数据缺少必要的列: {required_columns}")
        return None

    try:
        # 计算每个坐标点的平均风速
        df_avg = df.groupby(['lat', 'lon']).agg({
            'predicted_wind_speed': 'mean',
            'elevation': 'first',
            'slope': 'first',
            'grid_proximity': 'first',
            'road_distance': 'first',
            'residential_distance': 'first',
            'heritage_distance': 'first',
            'geology_distance': 'first',
            'water_distance': 'first',
            'cost': 'first'
        }).reset_index()

        # 重命名风速列为平均风速
        df_avg = df_avg.rename(columns={'predicted_wind_speed': 'avg_wind_speed'})

        return df_avg

    except Exception as e:
        st.error(f"数据预处理错误: {str(e)}")
        return None


def display_maale_gilboa_standalone_map(height=600):
    """显示Maale Gilboa基础地图"""
    base_map = create_maale_gilboa_base_map()
    if base_map is None:
        st.error("无法加载地图数据")
        return

    fig = go.Figure()

    # 添加边界线
    for polygon in base_map['polygons']:
        lats, lons = [], []
        for point in polygon.exterior.coords:
            lons.append(point[0])
            lats.append(point[1])

        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode='lines',
            line=dict(width=3, color='red'),
            name="Maale Gilboa边界",
            showlegend=True,
            hoverinfo='text',
            hovertext='Maale Gilboa区域边界'
        ))

    # 地图布局 - 默认使用OpenStreetMap
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",  # 固定使用OpenStreetMap
            center=dict(lat=base_map['center_lat'], lon=base_map['center_lon']),
            zoom=12,  # 调整缩放级别以适应Maale Gilboa区域
        ),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # 区域信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("区域名称", "Maale Gilboa")
    with col2:
        st.metric("所属地区", "以色列")
    with col3:
        area_km2 = base_map['geometry'].area * 10000
        st.metric("区域面积", f"{area_km2:.0f} km²")


def display_environment(df, height=600):
    """显示风能资源分布 - 使用平均风速"""
    base_map = create_maale_gilboa_base_map()
    if base_map is None:
        st.error("无法加载地图数据")
        return

    # 预处理数据，计算平均风速
    with st.spinner('正在计算平均风速...'):
        df_processed = preprocess_wind_data(df)

    if df_processed is None:
        return

    # 数据预处理 - 确保数据格式正确
    try:
        # 确保必要的列存在
        required_columns = ['lon', 'lat', 'avg_wind_speed']
        if not all(col in df_processed.columns for col in required_columns):
            st.error(f"处理后的数据缺少必要的列: {required_columns}")
            return

        # 空间数据处理
        gdf = gpd.GeoDataFrame(
            df_processed,
            geometry=gpd.points_from_xy(df_processed["lon"], df_processed["lat"]),
            crs="EPSG:4326"
        )

        gdf_maale_gilboa = gdf[gdf.within(base_map['geometry'])]
        if gdf_maale_gilboa.empty:
            st.warning("所选数据在Maale Gilboa区域内无有效点位")
            return

        fig = go.Figure()

        # 添加边界
        for polygon in base_map['polygons']:
            lats, lons = [], []
            for point in polygon.exterior.coords:
                lons.append(point[0])
                lats.append(point[1])

            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode='lines',
                line=dict(width=3, color='red'),
                name="区域边界",
                showlegend=True
            ))

        # 添加热力图 - 使用平均风速
        if not gdf_maale_gilboa.empty:
            fig.add_trace(go.Densitymapbox(
                lat=gdf_maale_gilboa["lat"],
                lon=gdf_maale_gilboa["lon"],
                z=gdf_maale_gilboa["avg_wind_speed"],
                radius=25,
                colorscale='Viridis',
                opacity=0.7,
                name="平均风速分布",
                showscale=True,
                hovertemplate=(
                    '<b>24小时平均风速</b>: %{z:.2f} m/s<br>'
                    '经纬度: (%{lat:.3f}, %{lon:.3f})<br>'
                    '<extra></extra>'
                ),
                colorbar=dict(
                    title="平均风速 (m/s)"
                )
            ))

        # 地图布局 - 默认使用OpenStreetMap
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",  # 固定使用OpenStreetMap
                center=dict(lat=base_map['center_lat'], lon=base_map['center_lon']),
                zoom=12,  # 调整缩放级别
            ),
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            title="Maale Gilboa区域 24小时平均风速分布图"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 数据统计
        if not gdf_maale_gilboa.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_value = gdf_maale_gilboa["avg_wind_speed"].mean()
                st.metric("平均风速", f"{avg_value:.2f} m/s")
            with col2:
                valid_count = len(gdf_maale_gilboa)
                st.metric("有效点位", f"{valid_count} 个")
            with col3:
                max_value = gdf_maale_gilboa["avg_wind_speed"].max()
                st.metric("最大平均风速", f"{max_value:.2f} m/s")
            with col4:
                min_value = gdf_maale_gilboa["avg_wind_speed"].min()
                st.metric("最小平均风速", f"{min_value:.2f} m/s")

            # 显示风速分布信息
            st.subheader("风速分布统计")
            col5, col6, col7 = st.columns(3)
            with col5:
                wind_std = gdf_maale_gilboa["avg_wind_speed"].std()
                st.metric("风速标准差", f"{wind_std:.2f} m/s")
            with col6:
                wind_median = gdf_maale_gilboa["avg_wind_speed"].median()
                st.metric("风速中位数", f"{wind_median:.2f} m/s")
            with col7:
                # 计算优质风能点位（假设平均风速 > 6 m/s 为优质点位）
                good_wind_points = len(gdf_maale_gilboa[gdf_maale_gilboa["avg_wind_speed"] > 6])
                st.metric("优质风能点位", f"{good_wind_points} 个")

    except Exception as e:
        st.error(f"数据处理错误: {str(e)}")
        st.info("请检查数据格式，确保包含经纬度坐标和风速数据")


def display_optimization_map(result, df, height=600):
    """在左侧地图上显示优化结果（风场位置） - 使用平均风速"""
    base_map = create_maale_gilboa_base_map()
    if base_map is None:
        st.error("无法加载地图数据")
        return

    # 预处理数据，计算平均风速
    with st.spinner('正在计算平均风速...'):
        df_processed = preprocess_wind_data(df)

    if df_processed is None:
        return

    # 兼容不同的结果格式
    try:
        # 尝试不同的键名来获取解决方案
        if "solution" in result:
            sol = result["solution"]
        elif "best_positions" in result:
            sol = result["best_positions"]
        elif "positions" in result:
            sol = result["positions"]
        elif "selected_indices" in result:
            sol = result["selected_indices"]
        else:
            # 如果没有明确的解决方案键，尝试使用第一个可迭代的值
            for key, value in result.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    sol = value
                    break
            else:
                st.error("❌ 无法找到有效的解决方案数据")
                return

        if not sol:
            st.error("❌ 没有找到有效的解决方案")
            return

        # 关键修改：处理索引映射问题
        if isinstance(sol, (list, np.ndarray)):
            # 方法1：如果sol是坐标索引
            if max(sol) < len(df_processed):
                # 直接使用预处理数据的索引
                valid_indices = [idx for idx in sol if idx < len(df_processed)]
                turbines = df_processed.iloc[valid_indices].copy().reset_index(drop=True)
            else:
                # 获取原始数据中的唯一坐标点
                unique_coords = df[['lat', 'lon']].drop_duplicates().reset_index(drop=True)

                # 找出被选中的坐标点在唯一坐标列表中的索引
                selected_coord_indices = []
                for idx in sol:
                    if idx < len(df):
                        # 获取原始数据中该索引的坐标
                        original_coord = (df.iloc[idx]['lat'], df.iloc[idx]['lon'])
                        # 在唯一坐标列表中查找这个坐标
                        for i, coord in enumerate(unique_coords.itertuples()):
                            if abs(coord.lat - original_coord[0]) < 0.0001 and abs(
                                    coord.lon - original_coord[1]) < 0.0001:
                                selected_coord_indices.append(i)
                                break

                # 去重
                selected_coord_indices = list(set(selected_coord_indices))

                if not selected_coord_indices:
                    st.error("❌ 无法映射索引到预处理数据")
                    return

                # 从预处理数据中获取对应的点
                turbines = df_processed.iloc[selected_coord_indices].copy().reset_index(drop=True)

        else:
            st.error(f"❌ 解决方案格式不正确: {type(sol)}")
            return

        # 修改：将选中的点位分组为风场
        # 假设每个风场包含固定数量的风机（根据界面设置）
        n_farms = st.session_state.get('n_farms', 2)  # 从session_state获取风场数量
        n_turbines_per_farm = st.session_state.get('n_turbines_per_farm', 4)  # 从session_state获取单场风机数

        # 将选中的点位分组到不同的风场
        farms = []
        for i in range(n_farms):
            start_idx = i * n_turbines_per_farm
            end_idx = start_idx + n_turbines_per_farm
            farm_turbines = turbines.iloc[start_idx:end_idx].copy().reset_index(drop=True)

            if len(farm_turbines) > 0:
                # 计算风场的中心位置
                center_lat = farm_turbines['lat'].mean()
                center_lon = farm_turbines['lon'].mean()

                # 计算风场的平均风速
                avg_wind_speed = farm_turbines[
                    'avg_wind_speed'].mean() if 'avg_wind_speed' in farm_turbines.columns else 0

                farms.append({
                    'farm_id': f"风场{i + 1}",
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'avg_wind_speed': avg_wind_speed,
                    'turbine_count': len(farm_turbines),
                    'turbines': farm_turbines  # 保留该风场的所有风机信息
                })

        # 保留Maale Gilboa区域内的风场
        farms_maale_gilboa = []
        for farm in farms:
            if Point(farm['center_lon'], farm['center_lat']).within(base_map['geometry']):
                farms_maale_gilboa.append(farm)

        if not farms_maale_gilboa:
            st.warning("⚠️ 优化结果中没有在Maale Gilboa区域内的风场位置")
            return

        fig = go.Figure()

        # 添加区域边界线
        for polygon in base_map['polygons']:
            lats, lons = [], []
            for point in polygon.exterior.coords:
                lons.append(point[0])
                lats.append(point[1])

            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode='lines',
                line=dict(width=3, color='red'),
                name="Maale Gilboa边界",
                showlegend=True
            ))

        # 添加风能热力图背景 - 使用平均风速
        gdf = gpd.GeoDataFrame(
            df_processed.copy(),
            geometry=gpd.points_from_xy(df_processed["lon"], df_processed["lat"]),
            crs="EPSG:4326"
        )
        gdf_maale_gilboa = gdf[gdf.within(base_map['geometry'])]

        if not gdf_maale_gilboa.empty and 'avg_wind_speed' in gdf_maale_gilboa.columns:
            fig.add_trace(go.Densitymapbox(
                lat=gdf_maale_gilboa["lat"],
                lon=gdf_maale_gilboa["lon"],
                z=gdf_maale_gilboa["avg_wind_speed"],
                radius=20,
                colorscale='Viridis',
                opacity=0.5,
                name="平均风速背景",
                showscale=True,
                hovertemplate='24小时平均风速: %{z:.2f} m/s',
                colorbar=dict(title="平均风速 (m/s)")
            ))

        # 修改：添加风场位置而不是单个风机
        if farms_maale_gilboa:
            # 为不同的风场使用不同的颜色
            colors = ['red', 'blue', 'green', 'orange', 'purple']

            for i, farm in enumerate(farms_maale_gilboa):
                color = colors[i % len(colors)]

                # 添加风场中心位置
                fig.add_trace(go.Scattermapbox(
                    lat=[farm['center_lat']],
                    lon=[farm['center_lon']],
                    mode="markers+text",
                    marker=dict(
                        color=color,
                        size=20,  # 风场标记比风机大
                        symbol="circle",
                        opacity=0.9
                    ),
                    text=[farm['farm_id']],
                    textposition="top center",
                    hovertext=[
                        f"<b>{farm['farm_id']}</b><br>"
                        f"中心经度: {farm['center_lon']:.3f}<br>"
                        f"中心纬度: {farm['center_lat']:.3f}<br>"
                        f"风机数量: {farm['turbine_count']} 台<br>"
                        + (f"平均风速: {farm['avg_wind_speed']:.2f} m/s<br>" if farm['avg_wind_speed'] > 0 else "")
                    ],
                    hoverinfo="text",
                    name=farm['farm_id'],
                    textfont=dict(size=12, color='black', weight='bold')
                ))

        # 地图布局 - 默认使用OpenStreetMap
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",  # 固定使用OpenStreetMap
                center=dict(lat=base_map['center_lat'], lon=base_map['center_lon']),
                zoom=12,  # 调整缩放级别
            ),
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            title=f"Maale Gilboa区域风场优化布局图 - 共{len(farms_maale_gilboa)}个风场"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 显示基本信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("优化风场数量", len(farms_maale_gilboa))
        with col2:
            total_turbines = sum(farm['turbine_count'] for farm in farms_maale_gilboa)
            st.metric("总风机数量", total_turbines)
        with col3:
            if farms_maale_gilboa:
                avg_speed = np.mean([farm['avg_wind_speed'] for farm in farms_maale_gilboa])
                st.metric("平均风速", f"{avg_speed:.2f} m/s")
            else:
                st.metric("平均风速", "N/A")
        with col4:
            # 计算风场间距
            if len(farms_maale_gilboa) > 1:
                from geopy.distance import geodesic
                min_distance = float('inf')
                for i in range(len(farms_maale_gilboa)):
                    for j in range(i + 1, len(farms_maale_gilboa)):
                        coord1 = (farms_maale_gilboa[i]['center_lat'], farms_maale_gilboa[i]['center_lon'])
                        coord2 = (farms_maale_gilboa[j]['center_lat'], farms_maale_gilboa[j]['center_lon'])
                        dist = geodesic(coord1, coord2).km
                        if dist < min_distance:
                            min_distance = dist
                st.metric("最小风场间距", f"{min_distance:.1f} km")
            else:
                st.metric("最小风场间距", "N/A")

    except Exception as e:
        st.error(f"优化结果显示错误: {str(e)}")
        # 显示调试信息
        with st.expander("🔍 调试信息"):
            st.write("结果字典的键:", list(result.keys()))
            st.write("结果类型:", type(result))
            st.write("错误详情:", str(e))
            import traceback
            st.write("完整错误跟踪:")
            st.code(traceback.format_exc())

# ======================================================
# 🚀 运行 Streamlit
# ======================================================
if __name__ == "__main__":
    strategy_optimization_page()