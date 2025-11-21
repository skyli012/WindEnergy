import geopandas as gpd
import streamlit as st
import pandas as pd
import numpy as np
from shapely.geometry import Point

from src.optimization.algorithm_convergence_curve import call_optimize_function
from src.utils.check_data import check_data_quality
from src.utils.create_map import display_maale_gilboa_standalone_map, display_environment, display_optimization_map, \
    create_maale_gilboa_base_map
from src.visualization.opt_result_show import display_optimization_result


# ======================================================
# 🌬️ 主页面：风电场选址优化系统
# ======================================================
def strategy_optimization_page():
    # 页面标题 - 更紧凑
    st.markdown("### 🌬️ 风电场选址优化与储能调度系统")
    st.caption("基于真实优化算法计算 · 奉节县风电场布局优化 · 储能消纳策略分析")

    # 初始化 session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "map"

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
                st.info("📁 请先上传风速预测数据以查看风能分布")

        elif st.session_state.current_page == "wind":
            if "windfarm_data" in st.session_state:
                display_environment(st.session_state["windfarm_data"])
                if "optimization_result" not in st.session_state:
                    st.info("⚙️ 数据已就绪，可点击'开始优化'进行风电场布局优化")
            else:
                st.warning("⚠️ 请先上传数据文件")
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
        col1, col2, col3 = st.columns(3)
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

        with col3:
            cost_weight = st.slider("成本权重", 0.1, 2.0, 1.0, 0.1, help="成本在优化中的重要性")

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


        # 储能系统参数
        st.markdown("**🔋 储能系统参数**")
        col6, col7, col8 = st.columns(3)
        with col6:
            # 根据风场数量动态调整储能容量
            base_storage = 40
            storage_per_farm = 20
            recommended_storage = base_storage + (n_farms - 1) * storage_per_farm
            storage_capacity = st.slider("储能容量 (MWh)", 1, 200, recommended_storage,
                                         help=f"推荐值: {recommended_storage}MWh ({n_farms}个风场)")
        with col7:
            base_power = 30
            power_per_farm = 15
            recommended_power = base_power + (n_farms - 1) * power_per_farm
            max_power = st.slider("最大功率 (MW)", 1, 80, recommended_power,
                                  help=f"推荐值: {recommended_power}MW ({n_farms}个风场)")
        with col8:
            base_grid = 50
            grid_per_farm = 25
            recommended_grid = base_grid + (n_farms - 1) * grid_per_farm
            grid_capacity = st.slider("电网容量 (MW)", 10, 150, 50,
                                      help=f"推荐值: {recommended_grid}MW ({n_farms}个风场)")

        # 功率变化率参数
        st.markdown("**📊 运行参数**")
        max_ramp_rate = st.slider("最大功率变化率 (MW/min)", 1, 30, 5 + n_farms,
                                  help="多风场运行时需要更高的变化率容限")

        # 固定约束条件值 - 使用合理的固定风场间距和风机间距
        algorithm_params = {
            'n_farms': n_farms,
            'n_turbines_per_farm': n_turbines,
            'total_turbines': total_turbines,
            'cost_weight': cost_weight,
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
            'storage_capacity': storage_capacity * 1000,  # 转换为kWh
            'max_power': max_power * 1000,  # 转换为kW
            'grid_capacity': grid_capacity * 1000,  # 转换为kW
            'max_ramp_rate': max_ramp_rate,
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
                base_map = create_maale_gilboa_base_map()
                if base_map:
                    # 创建几何点并检查是否在边界内
                    geometries = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
                    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

                    # 过滤边界内的点
                    within_boundary = gdf.within(base_map['geometry'])
                    df = df[within_boundary].copy().reset_index(drop=True)

                    st.info(f"🗺️ 过滤后：{len(df)} 个点在奉节县边界内")

                # 然后设置有效点位
                df["valid"] = (
                        (df["predicted_wind_speed"] >= 5.0) &
                        (df["slope"] <= 35) &
                        (df["elevation"] >= 150) & (df["elevation"] <= 1600)
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
            if valid_count < total_turbines:
                st.error(f"❌ 有效点位数量({valid_count})少于目标风机数量({total_turbines})")
                st.info("💡 建议：减少风场数量或单场风机数，或检查数据约束条件")
            else:
                st.success(f"✅ 有效点位数量({valid_count})满足目标风机数量({total_turbines})")

            if st.button("🚀 开始优化计算", use_container_width=True, type="primary"):
                with st.spinner(f"正在计算{n_farms}个风电场的最优布局..."):
                    try:
                        # 使用真实优化函数调用
                        result = call_optimize_function(df, algo, algorithm_params)
                        st.session_state["optimization_result"] = result
                        st.success(f"🎯 {n_farms}个风电场优化完成")
                        st.session_state.current_page = "result"
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 优化计算失败: {str(e)}")
                        st.info("💡 建议：尝试减少风场数量或使用更宽松的约束条件")
        else:
            st.button("🚀 开始优化计算", use_container_width=True, disabled=True)

    # ========== 优化结果详情展示在页面下端 ==========
    if st.session_state.current_page == "result" and "optimization_result" in st.session_state:
        st.markdown("---")
        st.markdown("#### 📊 多风场优化结果分析")

        result = st.session_state["optimization_result"]
        df = st.session_state["windfarm_data"]

        # 显示多风场特定的分析结果
        display_optimization_result(result, df)


# ======================================================
# 🚀 运行 Streamlit
# ======================================================
if __name__ == "__main__":
    strategy_optimization_page()