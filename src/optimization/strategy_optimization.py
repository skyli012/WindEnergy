import geopandas as gpd  # 正确导入 geopandas
import streamlit as st
import pandas as pd
from shapely.geometry import Point  # 明确从shapely导入

from src.optimization.algorithm_convergence_curve import call_optimize_function
from src.utils.check_data import check_data_quality
from src.utils.create_map import display_fengjie_standalone_map, display_environment, display_optimization_map, \
    create_fengjie_base_map
from src.visualization.opt_result_show import display_optimization_result

# ======================================================
# 🌬️ 主页面：风电场选址优化系统
# ======================================================
def strategy_optimization_page():
    # 页面标题 - 更紧凑
    st.markdown("### 🌬️ 风电场选址优化系统")
    st.caption("基于真实优化算法计算 · 奉节县风机布局优化")

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

        # 固定约束条件值
        algorithm_params = {
            'n_turbines': n_turbines,
            'cost_weight': cost_weight,
            'max_slope': 35,
            'max_road_distance': 100,
            'min_residential_distance': 60,
            'min_heritage_distance': 70,
            'min_geology_distance': 80,
            'min_water_distance': 100
        }

        # 算法高级参数（可选）
        st.markdown("**🔧 算法高级参数（可选）**")
        with st.expander("算法高级参数设置", expanded=False):
            if algo == "遗传算法":
                # 遗传算法参数 - 2行2列布局
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
                # 模拟退火参数 - 同一行布局
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
                # 粒子群优化参数 - 2行2列布局
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
                # PuLP求解器参数
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
                        (df["predicted_wind_speed"] >= 3.0) &  # 降低风速要求
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

        # 调试信息
        with st.expander("🔍 调试信息"):
            st.json({
                "算法参数": algorithm_params,
                "最终适应度": result.get('best_fitness', '未知'),
                "数据点数": len(df),
                "有效点数": df['valid'].sum() if 'valid' in df.columns else '未知',
                "优化模式": "真实算法计算"
            })


# ======================================================
# 🚀 运行 Streamlit
# ======================================================
if __name__ == "__main__":
    strategy_optimization_page()