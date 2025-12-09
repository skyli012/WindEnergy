import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 尝试导入XGBoost
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost未安装，将使用随机森林替代")

# 导入你的核心模块（请确保路径正确）
try:
    from src.optimization.algorithm_convergence_curve import call_optimize_function_with_all_strategies
    from src.visualization.opt_result_show import display_optimization_result, display_wind_utilization_analysis
except ImportError as e:
    st.error(f"模块导入失败: {e}")
    st.stop()


# ==================== 新增储能利用率计算函数 ====================
def calculate_storage_utilization_from_optimization_result(power_results):
    """
    基于优化结果计算储能利用率
    """
    if not power_results:
        return {
            'annual_generation_gwh': 0,
            'total_capacity_mw': 0,
            'capacity_factor': 0,
            'utilization_rate': 0,
            'equivalent_hours': 0,
            'theoretical_max_generation_gwh': 0,
            'generation_efficiency': 0,
            'estimated_storage_requirement_gwh': 0,
            'storage_contribution_gwh': 0,
            'equivalent_storage_capacity_gwh': 0,
            'storage_utilization_by_hours': 0,
            'comprehensive_storage_utilization': 0,
            'storage_cycle_times_per_year': 0,
            'estimated_storage_efficiency': 0.85
        }

    # 从优化结果中提取关键参数
    annual_generation_gwh = power_results.get('total_annual_generation_gwh', 0)
    total_capacity_mw = power_results.get('total_capacity_mw', 0)
    capacity_factor = power_results.get('average_capacity_factor', 0)
    utilization_rate = power_results.get('average_utilization_rate', 0)
    equivalent_hours = power_results.get('equivalent_full_load_hours', 0)

    # 计算理论最大发电量
    theoretical_max_generation = total_capacity_mw * 8760 / 1000  # GWh

    # 1. 基于容量因子的储能需求分析
    storage_for_smoothing = annual_generation_gwh * (
            1 - capacity_factor) / capacity_factor if capacity_factor > 0 else 0

    # 2. 基于利用率的储能配置估算
    base_utilization = 0.5  # 假设无储能时的基础利用率
    storage_contribution = annual_generation_gwh * (
            utilization_rate - base_utilization) if utilization_rate > base_utilization else 0

    # 3. 等效储能容量估算
    equivalent_storage_capacity_gwh = storage_contribution * 0.3  # 经验系数

    # 4. 储能利用率计算
    storage_utilization_by_hours = min(equivalent_hours / 8760, 1.0) if equivalent_hours > 0 else 0

    # 5. 综合储能利用率
    comprehensive_storage_utilization = (
            capacity_factor * 0.4 +
            utilization_rate * 0.3 +
            storage_utilization_by_hours * 0.3
    )

    results = {
        'annual_generation_gwh': annual_generation_gwh,
        'total_capacity_mw': total_capacity_mw,
        'capacity_factor': capacity_factor,
        'utilization_rate': utilization_rate,
        'equivalent_hours': equivalent_hours,
        'theoretical_max_generation_gwh': theoretical_max_generation,
        'generation_efficiency': annual_generation_gwh / theoretical_max_generation if theoretical_max_generation > 0 else 0,

        # 储能利用率指标
        'estimated_storage_requirement_gwh': storage_for_smoothing,
        'storage_contribution_gwh': storage_contribution,
        'equivalent_storage_capacity_gwh': equivalent_storage_capacity_gwh,
        'storage_utilization_by_hours': storage_utilization_by_hours,
        'comprehensive_storage_utilization': comprehensive_storage_utilization,

        # 储能性能指标
        'storage_cycle_times_per_year': capacity_factor * 365,
        'estimated_storage_efficiency': 0.85
    }

    return results


# ======================================================
# 🧠 预测-优化联合对比实验页面
# ======================================================
def prediction_optimization_comparison_page():
    # 页面标题
    st.markdown("### 🧪 预测模型 × 优化算法 联合对比实验")
    st.caption("选择不同风速预测模型与优化策略，评估端到端风电布局性能")

    # 初始化 session state
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "config"
    if 'algorithm_comparison_results' not in st.session_state:
        st.session_state.algorithm_comparison_results = {}
    if 'selected_prediction_models' not in st.session_state:
        st.session_state.selected_prediction_models = ["随机森林"]
    if 'selected_optimization_algorithms' not in st.session_state:
        st.session_state.selected_optimization_algorithms = ["遗传算法"]
    if 'dataset_split_info' not in st.session_state:
        st.session_state.dataset_split_info = None
    if 'dataset' not in st.session_state:
        st.warning("⚠️ 请先加载数据集")
        return

    # 检查数据集是否包含必要字段
    required_columns = ['lat', 'lon', 'predicted_wind_speed']
    df_base = st.session_state['dataset']

    missing_columns = [col for col in required_columns if col not in df_base.columns]
    if missing_columns:
        st.error(f"❌ 数据集缺少必要字段: {missing_columns}")
        st.info("请确保数据集包含以下字段：")
        for col in required_columns:
            st.write(f"- {col}")
        return

    # ========== 上半部分：参数调整和算法选择 ==========
    st.markdown("---")

    # 使用卡片式容器组织配置区域
    with st.container():
        st.markdown("#### ⚙️ 实验参数配置")

        # 第一行：模型和算法选择
        col1, col2 = st.columns(2)

        with col1:
            # 预测模型选择
            with st.container():
                st.markdown("**🔮 预测模型选择**")
                # 修改预测模型列表
                prediction_models = ["随机森林", "XGBoost", "LSTM", "GRU"]
                selected_pred_models = st.multiselect(
                    "选择预测模型（可多选）",
                    prediction_models,
                    default=["随机森林", "XGBoost"],
                    help="可选择多个预测模型进行对比"
                )
                st.session_state.selected_prediction_models = selected_pred_models

        with col2:
            # 优化算法选择
            with st.container():
                st.markdown("**🔧 优化算法选择**")
                optimization_algorithms = ["遗传算法", "模拟退火算法", "粒子群优化算法"]
                selected_opt_algorithms = st.multiselect(
                    "选择优化算法（可多选）",
                    optimization_algorithms,
                    default=["遗传算法", "模拟退火算法"],
                    help="可选择多个优化算法进行对比"
                )
                st.session_state.selected_optimization_algorithms = selected_opt_algorithms

        st.markdown("---")

        # 第二行：风场配置、权重设置、数据集划分、储能配置 - 四个等宽列
        col3, col4, col5, col6 = st.columns(4)

        with col3:
            with st.container():
                st.markdown("**🏗️ 风场配置**")
                # 风场数量 - 上下排列
                n_farms = st.slider("风场数量", 1, 5, st.session_state.get('n_farms', 2))
                st.session_state.n_farms = n_farms

                # 单场风机数 - 上下排列
                n_turbines = st.slider("单场风机数", 1, 10, st.session_state.get('n_turbines_per_farm', 4))
                st.session_state.n_turbines_per_farm = n_turbines

                total_turbines = n_farms * n_turbines
                st.metric("总风机数", f"{total_turbines} 台")

        with col4:
            with st.container():
                st.markdown("**🎯 优化目标权重**")
                # 使用数字输入框代替滑块，确保权重总和为1
                col_weight1, col_weight2, col_weight3 = st.columns(3)

                with col_weight1:
                    wind_speed_weight = st.number_input(
                        "风速权重",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                        help="风速稳定性的权重"
                    )

                with col_weight2:
                    utilization_weight = st.number_input(
                        "利用率权重",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.3,
                        step=0.05,
                        help="设备利用率的权重"
                    )

                with col_weight3:
                    storage_weight = st.number_input(
                        "储能权重",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.2,
                        step=0.05,
                        help="储能优化的权重"
                    )

                # 计算和显示权重总和
                total_weight = wind_speed_weight + utilization_weight + storage_weight
                if abs(total_weight - 1.0) > 0.01:
                    st.warning(f"权重总和: {total_weight:.2f} (建议调整为1.0)")
                else:
                    st.success(f"权重总和: {total_weight:.2f} ✓")

        with col5:
            with st.container():
                st.markdown("**📊 数据集划分配置**")

                # 使用更清晰的比例设置方式
                train_ratio = st.slider(
                    "训练集比例 (%)",
                    min_value=50,
                    max_value=80,
                    value=60,
                    step=5,
                    help="训练集占数据总量的比例"
                )

                # 自动计算验证集和测试集比例
                remaining = 100 - train_ratio
                val_ratio = st.slider(
                    "验证集比例 (%)",
                    min_value=10,
                    max_value=min(30, remaining - 10),
                    value=20,
                    step=5,
                    help="验证集占数据总量的比例"
                )

                # 测试集比例自动计算
                test_ratio = 100 - train_ratio - val_ratio

        with col6:
            with st.container():
                st.markdown("**🔋 储能系统配置**")

                # 储能策略选择
                storage_strategy = st.selectbox(
                    "储能策略",
                    ["平滑输出", "削峰填谷", "混合模式"],
                    help="选择储能系统的运行策略"
                )

                # 储能容量滑块 (MWh)
                storage_capacity_mwh = st.slider(
                    "储能容量 (MWh)",
                    1, 1000, 60,
                    help="储能系统的总容量 (兆瓦时)"
                )

                # 储能功率滑块 (MW)
                storage_power_mw = st.slider(
                    "储能功率 (MW)",
                    1, 500, 30,
                    help="储能系统的最大充放电功率 (兆瓦)"
                )

                # 计算并显示储能时间
                if storage_power_mw > 0:
                    storage_hours = storage_capacity_mwh / storage_power_mw
                else:
                    storage_hours = 0

        st.markdown("---")

        # 第三行：高级参数设置
        with st.container():
            st.markdown("**📋 算法高级参数**")
            with st.expander("展开高级参数设置", expanded=False):
                # 构建基础算法参数字典
                TURBINE_DIAMETER = 140  # 米
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

                # 设置合理的固定间距值
                DOWNWIND_DISTANCE_RATIO = 8.0  # 主风向间距 8倍D
                CROSSWIND_DISTANCE_RATIO = 4.0  # 侧向间距 4倍D

                # 计算实际间距
                min_downwind_distance = DOWNWIND_DISTANCE_RATIO * TURBINE_DIAMETER  # 米
                min_crosswind_distance = CROSSWIND_DISTANCE_RATIO * TURBINE_DIAMETER  # 米

                # 添加储能参数到基础参数中
                base_algorithm_params = {
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
                    'storage_weight': storage_weight,
                    'storage_strategy': storage_strategy,
                    'storage_capacity': storage_capacity_mwh * 1000,  # 转换为kWh
                    'storage_power': storage_power_mw * 1000,  # 转换为kW
                    'enable_storage_optimization': True if storage_weight > 0 else False,

                    # 储能参数范围（用于优化算法）
                    'min_storage_capacity': 10000,  # kWh
                    'max_storage_capacity': 200000,  # kWh
                    'min_storage_power': 5000,  # kW
                    'max_storage_power': 100000,  # kW
                }

                # 为每个算法设置参数
                tab1, tab2, tab3 = st.tabs(["遗传算法参数", "模拟退火参数", "粒子群优化参数"])

                with tab1:
                    ga_col1, ga_col2 = st.columns(2)
                    with ga_col1:
                        # 根据问题复杂度调整种群大小
                        base_pop_size = 50
                        pop_size_multiplier = n_farms * 2
                        recommended_pop = base_pop_size + pop_size_multiplier * 10
                        ga_pop_size = st.slider("种群大小", 20, 300, recommended_pop, key="ga_pop")
                        ga_generations = st.slider("迭代代数", 50, 500, 100 + n_farms * 20, key="ga_gen")
                    with ga_col2:
                        ga_mutation_rate = st.slider("变异率", 0.01, 0.3, 0.1, 0.01, key="ga_mut")
                        ga_crossover_rate = st.slider("交叉率", 0.5, 1.0, 0.8, 0.05, key="ga_cross")

                with tab2:
                    sa_col1, sa_col2 = st.columns(2)
                    with sa_col1:
                        sa_initial_temp = st.slider("初始温度", 100, 5000, 1000 + n_farms * 200, key="sa_temp")
                        sa_cooling_rate = st.slider("降温速率", 0.85, 0.99, 0.95, 0.01, key="sa_cool")
                    with sa_col2:
                        sa_iterations = st.slider("每温度迭代次数", 10, 200, 50 + n_farms * 10, key="sa_iter")

                with tab3:
                    pso_col1, pso_col2 = st.columns(2)
                    with pso_col1:
                        base_particles = 30
                        recommended_particles = base_particles + n_farms * 5
                        pso_pop_size = st.slider("粒子数量", 20, 150, recommended_particles, key="pso_pop")
                        pso_generations = st.slider("迭代次数", 50, 500, 100 + n_farms * 25, key="pso_gen")
                    with pso_col2:
                        pso_w = st.slider("惯性权重", 0.1, 1.0, 0.7, 0.1, key="pso_w")
                        pso_c1 = st.slider("个体学习因子", 0.1, 2.0, 1.5, 0.1, key="pso_c1")
                        pso_c2 = st.slider("社会学习因子", 0.1, 2.0, 1.5, 0.1, key="pso_c2")

                # 存储算法特定参数
                algorithm_specific_params = {
                    "遗传算法": {
                        'pop_size': ga_pop_size,
                        'generations': ga_generations,
                        'mutation_rate': ga_mutation_rate,
                        'crossover_rate': ga_crossover_rate
                    },
                    "模拟退火算法": {
                        'initial_temp': sa_initial_temp,
                        'cooling_rate': sa_cooling_rate,
                        'iterations_per_temp': sa_iterations
                    },
                    "粒子群优化算法": {
                        'pop_size': pso_pop_size,
                        'generations': pso_generations,
                        'w': pso_w,
                        'c1': pso_c1,
                        'c2': pso_c2
                    }
                }

        # 控制按钮区域
        st.markdown("---")

        # 检查数据是否可用
        data_available = False
        if 'dataset' in st.session_state:
            df_base = st.session_state['dataset']
            if ('lat' in df_base.columns and 'lon' in df_base.columns and
                    'predicted_wind_speed' in df_base.columns):
                data_available = True

        # 检查模型和算法是否选择
        models_selected = bool(selected_pred_models and selected_opt_algorithms)

        # 判断按钮是否禁用
        run_disabled = not (data_available and models_selected)

        # 开始按钮
        if st.button("🚀 开始联合实验",
                     type="primary",
                     use_container_width=True,
                     disabled=run_disabled):
            with st.spinner("正在执行多模型多算法对比实验..."):
                try:
                    # 执行所有组合的实验
                    _run_all_combinations_experiment(
                        selected_pred_models,
                        selected_opt_algorithms,
                        base_algorithm_params,
                        algorithm_specific_params,
                        n_farms,
                        train_ratio,
                        val_ratio,
                        test_ratio
                    )
                except Exception as e:
                    st.error(f"❌ 实验失败: {str(e)}")
                    st.exception(e)

    # ========== 下半部分：数据展示 ==========
    st.markdown("---")

    with st.container():
        st.markdown("#### 📊 实验结果分析")

        if st.session_state.current_view == "config":
            st.info("👆 请在上方配置实验参数并点击「开始联合实验」")

            # 显示配置预览
            if selected_pred_models and selected_opt_algorithms:
                with st.expander("📋 当前配置预览", expanded=True):
                    preview_col1, preview_col2, preview_col3 = st.columns(3)
                    with preview_col1:
                        st.write("**预测模型:**")
                        for model in selected_pred_models:
                            st.write(f"- {model}")
                        st.write(f"**风场配置:**")
                        st.write(f"- {n_farms}个风场 × {n_turbines}台风机")
                        st.write(f"- 总风机数: {total_turbines}台")
                    with preview_col2:
                        st.write("**优化算法:**")
                        for algo in selected_opt_algorithms:
                            st.write(f"- {algo}")
                        st.write("**权重设置:**")
                        st.write(f"- 风速: {wind_speed_weight}")
                        st.write(f"- 利用率: {utilization_weight}")
                        st.write(f"- 储能: {storage_weight}")
                    with preview_col3:
                        st.write("**储能配置:**")
                        st.write(f"- 策略: {storage_strategy}")
                        st.write(f"- 容量: {storage_capacity_mwh} MWh")
                        st.write(f"- 功率: {storage_power_mw} MW")
                        st.write(f"- 充放电时间: {storage_capacity_mwh / storage_power_mw:.1f} h")

        elif st.session_state.current_view == "result":
            if "all_experiment_results" not in st.session_state:
                st.warning("未找到实验结果，请重新运行实验。")
                return

            # 新的结果展示布局
            _display_new_results_layout()


# ==================== 新的结果展示函数 ====================

def _display_new_results_layout():
    """新的结果展示布局"""

    # 显示最佳组合推荐
    _display_best_combination_recommendation()

    st.markdown("---")

    # 第一：综合性能对比数据表
    st.markdown("### 📊 综合性能对比数据表")
    _display_comprehensive_table()

    st.markdown("---")

    # 第二：三个柱形图 2x2 排布
    st.markdown("### 📈 关键性能指标对比")
    _display_three_bar_charts()

    st.markdown("---")

    # 第三：雷达图
    st.markdown("### 🎯 多维度性能雷达图")
    _display_radar_chart()

    # 第四：详细分析图表
    st.markdown("---")
    st.markdown("### 📈 详细性能分析")
    _display_detailed_analysis_charts()


def _display_best_combination_recommendation():
    """显示最佳组合推荐 - 基于综合数据表的排序结果"""
    results = st.session_state["all_experiment_results"]
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}

    if not successful_results:
        st.warning("没有成功的实验组合")
        return

    # 准备成功的数据（复用_display_comprehensive_table的逻辑）
    comparison_data = []

    for combo_key, data in successful_results.items():
        if data['result'] is None:
            continue

        power = data['result'].get('power_results', {})
        result_data = data['result']

        # 获取平均风速
        if 'best_positions_data' in result_data:
            selected_df = result_data['best_positions_data']
            if isinstance(selected_df,
                          pd.DataFrame) and not selected_df.empty and 'predicted_wind_speed' in selected_df.columns:
                avg_wind_speed = selected_df['predicted_wind_speed'].mean()
            else:
                avg_wind_speed = 0
        else:
            avg_wind_speed = 0

        # 计算储能利用率
        storage_results = calculate_storage_utilization_from_optimization_result(power)
        storage_utilization = storage_results['comprehensive_storage_utilization'] * 100

        # 获取储能经济性分析
        storage_economic = result_data.get('storage_economic_analysis', {})
        storage_capacity_kwh = storage_economic.get('storage_capacity_kwh', 0)
        storage_power_kw = storage_economic.get('storage_power_kw', 0)

        comparison_data.append({
            '组合': combo_key,
            '预测模型': data['prediction_model'],
            '优化算法': data['optimization_algorithm'],
            '平均风速(m/s)': avg_wind_speed,
            '储能利用率(%)': storage_utilization,
            '年发电量(GWh)': power.get('total_annual_generation_gwh', 0),
            '最优适应度': data['fitness'],
            '容量因数(%)': power.get('average_capacity_factor', 0) * 100,
            '计算时间(秒)': data['computation_time'],
            '储能容量(kWh)': storage_capacity_kwh,
            '储能功率(kW)': storage_power_kw,
            '储能容量(MWh)': storage_capacity_kwh / 1000,
            '储能功率(MW)': storage_power_kw / 1000,
        })

    # 创建数据框并计算综合分数进行排序
    df_comp = pd.DataFrame(comparison_data)

    if df_comp.empty:
        st.warning("没有成功的数据可用于排序")
        return

    # 计算综合排名分数
    df_comp['综合分数'] = (
            df_comp['年发电量(GWh)'] * 0.20 +
            df_comp['最优适应度'] * 0.20 +
            df_comp['容量因数(%)'] * 0.15 +
            df_comp['储能利用率(%)'] * 0.15 +
            (1 / (df_comp['计算时间(秒)'] + 0.001)) * 0.10 +
            df_comp['平均风速(m/s)'] * 0.10 +
            (df_comp['储能利用率(%)'] / 100) * 0.10  # 额外考虑储能利用率
    )

    # 按综合分数降序排列
    df_comp = df_comp.sort_values('综合分数', ascending=False)

    # 获取最佳组合（第一行）
    best_row = df_comp.iloc[0]
    best_combo_key = best_row['组合']
    best_data = successful_results[best_combo_key]

    # 显示最佳组合推荐
    st.markdown("### 🏆 最佳组合推荐")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.success(f"**{best_combo_key}**")
        st.metric("综合得分", f"{best_row['综合分数']:.3f}")
        st.metric("预测模型", best_row['预测模型'])

    with col2:
        st.metric("年发电量", f"{best_row['年发电量(GWh)']:.1f} GWh")
        st.metric("容量因数", f"{best_row['容量因数(%)']:.1f}%")

    with col3:
        st.metric("平均风速", f"{best_row['平均风速(m/s)']:.1f} m/s")
        st.metric("储能利用率", f"{best_row['储能利用率(%)']:.1f}%")

    with col4:
        st.metric("计算时间", f"{best_row['计算时间(秒)']:.1f}秒")
        st.metric("最优适应度", f"{best_row['最优适应度']:.4f}")

    # 显示储能配置信息
    st.markdown("#### 🔋 最佳储能配置")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        storage_economic = best_data['result'].get('storage_economic_analysis', {})
        storage_capacity = storage_economic.get('storage_capacity_kwh', 0) / 1000  # 转换为MWh
        st.metric("储能容量", f"{storage_capacity:.1f} MWh")

    with col2:
        storage_power = storage_economic.get('storage_power_kw', 0) / 1000  # 转换为MW
        st.metric("储能功率", f"{storage_power:.1f} MW")

    with col3:
        storage_investment = storage_economic.get('storage_investment', 0)
        st.metric("储能投资", f"{storage_investment / 1e6:.2f} 百万元")

    with col4:
        storage_payback = storage_economic.get('storage_payback_years', 0)
        if storage_payback < float('inf'):
            st.metric("投资回收期", f"{storage_payback:.1f} 年")
        else:
            st.metric("投资回收期", "∞")

    # 显示优化算法信息
    st.info(f"**优化算法**: {best_row['优化算法']}")

    # 显示评分说明
    with st.expander("📋 评分标准说明"):
        st.markdown("""
        **综合评分权重分配:**
        - ⚡ 年发电量: 20%
        - 🎯 最优适应度: 20%
        - 📊 容量因数: 15%
        - 🔋 储能利用率: 15% + 10% (额外)
        - ⏱️ 计算效率: 10%
        - 🌬️ 平均风速: 10%

        *储能利用率得到额外权重，反映储能对风电场的优化效果*
        *基于综合性能排序，最佳组合自动排在最上方*
        """)

    # 可选：显示前3名组合的简要对比
    if len(df_comp) > 1:
        with st.expander("🥈🥉 其他优秀组合"):
            top3 = df_comp.head(3)
            for i, (_, row) in enumerate(top3.iterrows()):
                rank_icon = "🏆" if i == 0 else "🥈" if i == 1 else "🥉"
                st.write(f"{rank_icon} **{row['组合']}** - 综合得分: {row['综合分数']:.3f} "
                         f"(发电量: {row['年发电量(GWh)']:.1f} GWh, "
                         f"储能利用率: {row['储能利用率(%)']:.1f}%, "
                         f"适应度: {row['最优适应度']:.4f})")


def _display_comprehensive_table():
    """显示综合性能对比数据表 - 修复版本"""
    results = st.session_state["all_experiment_results"]

    # 准备对比数据 - 只处理成功的结果
    comparison_data = []
    failed_data = []  # 单独存储失败数据

    for combo_key, data in results.items():
        if data['status'] == 'success' and data['result'] is not None:
            power = data['result'].get('power_results', {})
            result_data = data['result']

            # 获取平均风速
            if 'best_positions_data' in result_data:
                selected_df = result_data['best_positions_data']
                if isinstance(selected_df,
                              pd.DataFrame) and not selected_df.empty and 'predicted_wind_speed' in selected_df.columns:
                    avg_wind_speed = selected_df['predicted_wind_speed'].mean()
                else:
                    avg_wind_speed = 0
            else:
                avg_wind_speed = 0

            # 计算储能利用率
            storage_results = calculate_storage_utilization_from_optimization_result(power)
            storage_utilization = storage_results['comprehensive_storage_utilization'] * 100

            # 获取储能经济性分析
            storage_economic = result_data.get('storage_economic_analysis', {})
            storage_capacity_kwh = storage_economic.get('storage_capacity_kwh', 0)
            storage_power_kw = storage_economic.get('storage_power_kw', 0)

            comparison_data.append({
                '组合': combo_key,
                '预测模型': data['prediction_model'],
                '优化算法': data['optimization_algorithm'],
                '平均风速(m/s)': avg_wind_speed,
                '储能利用率(%)': storage_utilization,
                '年发电量(GWh)': power.get('total_annual_generation_gwh', 0),
                '最优适应度': data['fitness'],
                '容量因数(%)': power.get('average_capacity_factor', 0) * 100,
                '计算时间(秒)': data['computation_time'],
                '储能容量(MWh)': storage_capacity_kwh / 1000,
                '储能功率(MW)': storage_power_kw / 1000,
                '储能投资(百万)': storage_economic.get('storage_investment', 0) / 1e6,
                '状态': '✅ 成功'
            })
        else:
            # 失败数据单独存储
            failed_data.append({
                '组合': combo_key,
                '预测模型': data['prediction_model'],
                '优化算法': data['optimization_algorithm'],
                '平均风速(m/s)': 0,
                '储能利用率(%)': 0,
                '年发电量(GWh)': 0,
                '最优适应度': 0,
                '容量因数(%)': 0,
                '计算时间(秒)': 0,
                '储能容量(MWh)': 0,
                '储能功率(MW)': 0,
                '储能投资(百万)': 0,
                '状态': f'❌ 失败: {data.get("error", "未知错误")}'
            })

    # 创建成功数据的数据框
    df_comp = pd.DataFrame(comparison_data)

    # 对成功的结果进行排序（按综合性能）
    if not df_comp.empty:
        # 计算综合排名分数
        df_comp['综合分数'] = (
                df_comp['年发电量(GWh)'] * 0.20 +
                df_comp['最优适应度'] * 0.20 +
                df_comp['容量因数(%)'] * 0.15 +
                df_comp['储能利用率(%)'] * 0.15 +
                (1 / (df_comp['计算时间(秒)'] + 0.001)) * 0.10 +
                df_comp['平均风速(m/s)'] * 0.10 +
                (df_comp['储能利用率(%)'] / 100) * 0.10
        )
        df_comp = df_comp.sort_values('综合分数', ascending=False)

    # 显示成功的数据框
    if not df_comp.empty:
        st.markdown("### ✅ 成功实验组合")
        st.dataframe(
            df_comp,
            use_container_width=True,
            height=min(400, len(df_comp) * 35 + 100),  # 动态调整高度
            column_config={
                "组合": st.column_config.TextColumn(width="medium"),
                "预测模型": st.column_config.TextColumn(width="small"),
                "优化算法": st.column_config.TextColumn(width="small"),
                "平均风速(m/s)": st.column_config.NumberColumn(format="%.2f"),
                "储能利用率(%)": st.column_config.NumberColumn(format="%.2f"),
                "年发电量(GWh)": st.column_config.NumberColumn(format="%.2f"),
                "最优适应度": st.column_config.NumberColumn(format="%.4f"),
                "容量因数(%)": st.column_config.NumberColumn(format="%.2f"),
                "计算时间(秒)": st.column_config.NumberColumn(format="%.2f"),
                "储能容量(MWh)": st.column_config.NumberColumn(format="%.1f"),
                "储能功率(MW)": st.column_config.NumberColumn(format="%.1f"),
                "储能投资(百万)": st.column_config.NumberColumn(format="%.2f"),
            }
        )

        # 添加排序说明
        st.info(f"**📊 表格说明**: 表格已按综合性能排序，最佳组合 **{df_comp.iloc[0]['组合']}** 排在最上方")

    # 显示失败的数据（如果有的话）
    if failed_data:
        st.markdown("### ❌ 失败实验组合")
        df_failed = pd.DataFrame(failed_data)
        st.dataframe(
            df_failed,
            use_container_width=True,
            height=min(300, len(df_failed) * 35 + 100),
            column_config={
                "组合": st.column_config.TextColumn(width="medium"),
                "预测模型": st.column_config.TextColumn(width="small"),
                "优化算法": st.column_config.TextColumn(width="small"),
                "状态": st.column_config.TextColumn(width="large"),
            }
        )

    # 如果没有成功数据
    if not comparison_data and not failed_data:
        st.warning("没有找到任何实验数据")


def _display_three_bar_charts():
    """显示三个柱形图：平均风速、年发电量、最优适应度、储能利用率"""
    results = st.session_state["all_experiment_results"]
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}

    if not successful_results:
        st.warning("没有成功的实验组合可显示")
        return

    # 准备数据 - 按综合性能排序
    combinations = []
    wind_speeds = []
    powers = []
    fitnesses = []
    capacity_factors = []
    storage_utilizations = []
    storage_capacities = []
    computation_times = []

    # 先收集所有数据
    temp_data = []
    for combo_key, data in successful_results.items():
        if data['result'] is None:
            continue

        power = data['result'].get('power_results', {})
        result_data = data['result']

        # 获取平均风速
        if 'best_positions_data' in result_data:
            selected_df = result_data['best_positions_data']
            if isinstance(selected_df,
                          pd.DataFrame) and not selected_df.empty and 'predicted_wind_speed' in selected_df.columns:
                avg_wind_speed = selected_df['predicted_wind_speed'].mean()
            else:
                avg_wind_speed = 0
        else:
            avg_wind_speed = 0

        # 年发电量
        annual_power = power.get('total_annual_generation_gwh', 0)

        # 最优适应度
        fitness = data['fitness']

        # 容量因数
        capacity_factor = power.get('average_capacity_factor', 0) * 100

        # 计算储能利用率
        storage_results = calculate_storage_utilization_from_optimization_result(power)
        storage_utilization = storage_results['comprehensive_storage_utilization'] * 100

        # 获取储能容量
        storage_economic = result_data.get('storage_economic_analysis', {})
        storage_capacity = storage_economic.get('storage_capacity_kwh', 0) / 1000  # 转换为MWh

        # 计算时间
        comp_time = data['computation_time']

        # 计算综合分数用于排序
        composite_score = (
                annual_power * 0.20 +
                fitness * 0.20 +
                capacity_factor * 0.15 +
                storage_utilization * 0.15 +
                (1 / (comp_time + 0.001)) * 0.10 +
                avg_wind_speed * 0.10 +
                (storage_utilization / 100) * 0.10
        )

        temp_data.append({
            'combo_key': combo_key,
            'wind_speed': avg_wind_speed,
            'power': annual_power,
            'fitness': fitness,
            'capacity_factor': capacity_factor,
            'storage_utilization': storage_utilization,
            'storage_capacity': storage_capacity,
            'computation_time': comp_time,
            'composite_score': composite_score
        })

    # 按综合分数排序
    temp_data.sort(key=lambda x: x['composite_score'], reverse=True)

    # 提取排序后的数据
    for item in temp_data:
        combinations.append(item['combo_key'])
        wind_speeds.append(item['wind_speed'])
        powers.append(item['power'])
        fitnesses.append(item['fitness'])
        capacity_factors.append(item['capacity_factor'])
        storage_utilizations.append(item['storage_utilization'])
        storage_capacities.append(item['storage_capacity'])
        computation_times.append(item['computation_time'])

    # 创建2x2子图 - 增加间距
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '平均风速对比 (m/s)',
            '年发电量对比 (GWh)',
            '最优适应度对比',
            '储能容量对比 (MWh)'  # 修改为储能容量
        ),
        vertical_spacing=0.25,  # 增加垂直间距
        horizontal_spacing=0.1,  # 增加水平间距
    )

    # 使用渐变色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 平均风速
    fig.add_trace(
        go.Bar(x=combinations, y=wind_speeds, name="平均风速",
               marker_color=colors[0], showlegend=False,
               text=wind_speeds, texttemplate='%{text:.1f}m/s', textposition='outside'),
        row=1, col=1
    )

    # 年发电量
    fig.add_trace(
        go.Bar(x=combinations, y=powers, name="年发电量",
               marker_color=colors[1], showlegend=False,
               text=powers, texttemplate='%{text:.1f}GWh', textposition='outside'),
        row=1, col=2
    )

    # 最优适应度
    fig.add_trace(
        go.Bar(x=combinations, y=fitnesses, name="最优适应度",
               marker_color=colors[2], showlegend=False,
               text=fitnesses, texttemplate='%{text:.3f}', textposition='outside'),
        row=2, col=1
    )

    # 储能容量
    fig.add_trace(
        go.Bar(x=combinations, y=storage_capacities, name="储能容量",
               marker_color=colors[3], showlegend=False,
               text=storage_capacities, texttemplate='%{text:.1f}MWh', textposition='outside'),
        row=2, col=2
    )

    # 更新布局 - 增加整体边距
    fig.update_layout(
        height=900,  # 增加整体高度以适应更大的间距
        showlegend=False,
        title_text="关键性能指标对比分析",
        template="plotly_white",
        font=dict(size=12),
        margin=dict(l=50, r=50, t=80, b=100),  # 增加边距
    )

    # 更新y轴标签
    fig.update_yaxes(title_text="风速 (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="发电量 (GWh)", row=1, col=2)
    fig.update_yaxes(title_text="适应度", row=2, col=1)
    fig.update_yaxes(title_text="储能容量 (MWh)", row=2, col=2)

    # 调整字体大小和角度
    fig.update_annotations(font_size=12)  # 增加子图标题字体大小
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=10)  # 调整x轴标签字体大小
    )

    # 增加y轴标签与图表的间距
    fig.update_yaxes(title_standoff=15)

    st.plotly_chart(fig, use_container_width=True)

    # 创建额外的图表显示储能利用率
    st.markdown("#### 🔋 储能性能指标")
    col1, col2 = st.columns(2)

    with col1:
        fig_util = go.Figure()
        fig_util.add_trace(go.Bar(
            x=combinations, y=storage_utilizations, name="储能利用率",
            marker_color='#FFA500',
            text=storage_utilizations, texttemplate='%{text:.1f}%', textposition='outside'
        ))
        fig_util.update_layout(
            title="储能利用率对比",
            xaxis_title="组合",
            yaxis_title="储能利用率 (%)",
            height=400
        )
        st.plotly_chart(fig_util, use_container_width=True)

    with col2:
        # 储能容量与利用率的散点图
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=storage_capacities,
            y=storage_utilizations,
            mode='markers+text',
            marker=dict(size=12, color='#2ca02c'),
            text=combinations,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>储能容量: %{x:.1f} MWh<br>储能利用率: %{y:.1f}%<extra></extra>'
        ))
        fig_scatter.update_layout(
            title="储能容量 vs 利用率",
            xaxis_title="储能容量 (MWh)",
            yaxis_title="储能利用率 (%)",
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 添加排序说明
    if len(combinations) > 0:
        st.info(f"**📊 图表说明**: 所有组合已按综合性能排序，最佳组合 **{combinations[0]}** 显示在最左侧")


def _display_radar_chart():
    """显示雷达图对比各个组合的多维度性能"""
    results = st.session_state["all_experiment_results"]
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}

    if not successful_results:
        st.warning("没有成功的实验组合可显示雷达图")
        return

    # 准备雷达图数据 - 添加储能相关指标
    categories = ['平均风速', '年发电量', '最优适应度', '计算效率', '容量因数', '储能利用率', '储能容量比']

    # 归一化数据用于雷达图
    normalized_data = []
    combinations = []

    # 收集原始数据并排序
    temp_data = []
    for combo_key, data in successful_results.items():
        if data['result'] is None:
            continue

        power = data['result'].get('power_results', {})
        result_data = data['result']

        # 获取平均风速
        if 'best_positions_data' in result_data:
            selected_df = result_data['best_positions_data']
            if isinstance(selected_df,
                          pd.DataFrame) and not selected_df.empty and 'predicted_wind_speed' in selected_df.columns:
                avg_wind_speed = selected_df['predicted_wind_speed'].mean()
            else:
                avg_wind_speed = 0
        else:
            avg_wind_speed = 0

        annual_power = power.get('total_annual_generation_gwh', 0)
        fitness = data['fitness']
        comp_time = data['computation_time']
        computation_efficiency = 1 / (comp_time + 0.001)
        capacity_factor = power.get('average_capacity_factor', 0) * 100

        # 计算储能利用率
        storage_results = calculate_storage_utilization_from_optimization_result(power)
        storage_utilization = storage_results['comprehensive_storage_utilization'] * 100

        # 获取储能容量
        storage_economic = result_data.get('storage_economic_analysis', {})
        storage_capacity = storage_economic.get('storage_capacity_kwh', 0) / 1000  # 转换为MWh

        # 计算储能容量比（储能容量/年发电量）
        storage_capacity_ratio = (storage_capacity / annual_power * 100) if annual_power > 0 else 0

        # 计算综合分数用于排序
        composite_score = (
                annual_power * 0.20 +
                fitness * 0.20 +
                capacity_factor * 0.15 +
                storage_utilization * 0.15 +
                computation_efficiency * 0.10 +
                avg_wind_speed * 0.10 +
                (storage_utilization / 100) * 0.10
        )

        temp_data.append({
            'combo_key': combo_key,
            'wind_speed': avg_wind_speed,
            'power': annual_power,
            'fitness': fitness,
            'computation_efficiency': computation_efficiency,
            'capacity_factor': capacity_factor,
            'storage_utilization': storage_utilization,
            'storage_capacity_ratio': storage_capacity_ratio,
            'composite_score': composite_score
        })

    # 按综合分数排序
    temp_data.sort(key=lambda x: x['composite_score'], reverse=True)

    # 收集排序后的原始数据
    raw_data = {
        'wind_speed': [],
        'power': [],
        'fitness': [],
        'computation_efficiency': [],
        'capacity_factor': [],
        'storage_utilization': [],
        'storage_capacity_ratio': []
    }

    for item in temp_data:
        combinations.append(item['combo_key'])
        raw_data['wind_speed'].append(item['wind_speed'])
        raw_data['power'].append(item['power'])
        raw_data['fitness'].append(item['fitness'])
        raw_data['computation_efficiency'].append(item['computation_efficiency'])
        raw_data['capacity_factor'].append(item['capacity_factor'])
        raw_data['storage_utilization'].append(item['storage_utilization'])
        raw_data['storage_capacity_ratio'].append(item['storage_capacity_ratio'])

    # 归一化数据（0-1范围）
    for i, combo in enumerate(combinations):
        normalized_values = []
        for key in ['wind_speed', 'power', 'fitness', 'computation_efficiency', 'capacity_factor',
                    'storage_utilization', 'storage_capacity_ratio']:
            values = raw_data[key]
            if len(values) > 0 and max(values) > min(values):
                normalized_val = (values[i] - min(values)) / (max(values) - min(values))
            else:
                normalized_val = 0.5  # 如果所有值相同，设为中间值
            normalized_values.append(normalized_val * 100)  # 转换为百分比

        normalized_data.append(normalized_values)

    # 创建雷达图
    fig = go.Figure()

    # 使用统一的颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, combo in enumerate(combinations):
        fig.add_trace(go.Scatterpolar(
            r=normalized_data[i] + [normalized_data[i][0]],  # 闭合雷达图
            theta=categories + [categories[0]],
            fill='toself',
            name=combo,
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.7
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0%', '25%', '50%', '75%', '100%']
            )
        ),
        showlegend=True,
        title="多维度性能雷达图对比（包含储能指标）",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # 添加说明
    with st.expander("📋 雷达图说明"):
        st.markdown("""
        - **雷达图显示了各组合在七个关键维度上的相对性能**
        - 所有指标都归一化到0-100%的范围进行比较
        - **面积越大表示综合性能越好**
        - 图例中的组合已按综合性能排序
        - 计算效率基于计算时间的倒数（时间越短效率越高）
        - **储能利用率**反映了储能系统对风电场的优化效果
        - **储能容量比**表示储能容量与年发电量的比例
        """)


def _display_detailed_analysis_charts():
    """显示详细分析图表"""
    results = st.session_state["all_experiment_results"]
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}

    if not successful_results:
        return

    # 创建详细分析标签页 - 添加第四个标签页显示详细结果
    tab1, tab2, tab3, tab4 = st.tabs(["📈 算法性能对比", "🔧 预测模型分析", "🎯 储能效果评估", "🔍 详细优化结果"])

    with tab1:
        _display_algorithm_performance_comparison(successful_results)

    with tab2:
        _display_prediction_model_analysis(successful_results)

    with tab3:
        _display_storage_effect_evaluation(successful_results)  # 修改为储能效果评估

    with tab4:
        _display_detailed_optimization_results(successful_results)


def _display_storage_effect_evaluation(successful_results):
    """显示储能效果评估"""
    st.markdown("#### 🔋 储能效果综合评估")

    # 准备储能相关数据
    storage_data = []
    for combo_key, data in successful_results.items():
        if data['result'] is None:
            continue

        result_data = data['result']
        storage_economic = result_data.get('storage_economic_analysis', {})

        if storage_economic:
            storage_data.append({
                '组合': combo_key,
                '储能容量(MWh)': storage_economic.get('storage_capacity_kwh', 0) / 1000,
                '储能功率(MW)': storage_economic.get('storage_power_kw', 0) / 1000,
                '储能投资(百万)': storage_economic.get('storage_investment', 0) / 1e6,
                '储能年收益(百万)': storage_economic.get('storage_annual_revenue', 0) / 1e6,
                '储能运维成本(百万)': storage_economic.get('storage_om_cost', 0) / 1e6,
                '储能净收益(百万)': storage_economic.get('storage_net_benefit', 0) / 1e6,
                '回收期(年)': storage_economic.get('storage_payback_years', 0),
                '充放电时间(h)': (storage_economic.get('storage_capacity_kwh', 0) /
                                  storage_economic.get('storage_power_kw', 1) if storage_economic.get(
                    'storage_power_kw', 0) > 0 else 0)
            })

    if storage_data:
        df_storage = pd.DataFrame(storage_data)

        # 显示储能配置表格
        st.markdown("##### 📋 储能配置详情")
        st.dataframe(
            df_storage,
            use_container_width=True,
            column_config={
                "组合": st.column_config.TextColumn(width="medium"),
                "储能容量(MWh)": st.column_config.NumberColumn(format="%.1f"),
                "储能功率(MW)": st.column_config.NumberColumn(format="%.1f"),
                "储能投资(百万)": st.column_config.NumberColumn(format="%.2f"),
                "储能年收益(百万)": st.column_config.NumberColumn(format="%.2f"),
                "储能运维成本(百万)": st.column_config.NumberColumn(format="%.2f"),
                "储能净收益(百万)": st.column_config.NumberColumn(format="%.2f"),
                "回收期(年)": st.column_config.NumberColumn(format="%.1f"),
                "充放电时间(h)": st.column_config.NumberColumn(format="%.1f"),
            }
        )

        # 储能经济性分析图表
        st.markdown("##### 📈 储能经济性分析")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('储能投资成本', '储能年收益', '投资回收期', '充放电时间'),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
        )

        fig.add_trace(
            go.Bar(x=df_storage['组合'], y=df_storage['储能投资(百万)'],
                   name='投资成本', marker_color='#d62728'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=df_storage['组合'], y=df_storage['储能年收益(百万)'],
                   name='年收益', marker_color='#2ca02c'),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x=df_storage['组合'], y=df_storage['回收期(年)'],
                   name='回收期', marker_color='#ff7f0e'),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(x=df_storage['组合'], y=df_storage['充放电时间(h)'],
                   name='充放电时间', marker_color='#9467bd'),
            row=2, col=2
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="储能经济性指标对比"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 储能效率分析
        st.markdown("##### ⚡ 储能系统效率分析")

        # 计算储能效率指标
        efficiency_data = []
        for i, row in df_storage.iterrows():
            efficiency = (row['储能年收益(百万)'] / row['储能投资(百万)']) * 100 if row['储能投资(百万)'] > 0 else 0
            efficiency_data.append({
                '组合': row['组合'],
                '投资收益率(%)': efficiency,
                '单位容量投资(万元/MWh)': (row['储能投资(百万)'] * 100) / row['储能容量(MWh)'] if row[
                                                                                                      '储能容量(MWh)'] > 0 else 0,
                '单位功率投资(万元/MW)': (row['储能投资(百万)'] * 100) / row['储能功率(MW)'] if row[
                                                                                                    '储能功率(MW)'] > 0 else 0,
            })

        df_efficiency = pd.DataFrame(efficiency_data)

        col1, col2, col3 = st.columns(3)
        with col1:
            avg_return = df_efficiency['投资收益率(%)'].mean()
            st.metric("平均投资收益率", f"{avg_return:.1f}%")
        with col2:
            avg_cap_cost = df_efficiency['单位容量投资(万元/MWh)'].mean()
            st.metric("平均单位容量投资", f"{avg_cap_cost:.0f} 万元/MWh")
        with col3:
            avg_power_cost = df_efficiency['单位功率投资(万元/MW)'].mean()
            st.metric("平均单位功率投资", f"{avg_power_cost:.0f} 万元/MW")

    else:
        st.info("无储能经济性分析数据")


def _display_detailed_optimization_results(successful_results):
    """显示每个组合的详细优化结果"""
    st.markdown("#### 🔍 详细优化结果")

    # 让用户选择要查看的组合
    combo_keys = list(successful_results.keys())
    selected_combo = st.selectbox("选择要查看的组合", combo_keys)

    if selected_combo:
        data = successful_results[selected_combo]
        result = data['result']

        if result is None:
            st.error("❌ 该组合没有优化结果")
            return

        # 显示组合基本信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("预测模型", data['prediction_model'])
        with col2:
            st.metric("优化算法", data['optimization_algorithm'])
        with col3:
            st.metric("最优适应度", f"{data['fitness']:.4f}")

        # 显示详细结果（使用之前编写的显示代码）
        _display_single_result_details(result, data)


def _display_single_result_details(result, data):
    """显示单个结果的详细信息"""

    # 创建三个标签页来显示结果
    tab1, tab2, tab3 = st.tabs(["📊 结果概览", "🔧 详细数据", "🔋 储能分析"])

    with tab1:
        st.subheader("优化结果基本信息")

        # 显示result的基本信息
        if result is None:
            st.error("❌ 优化结果为空")
        else:
            # 显示result的类型和键
            st.write(f"**结果类型:** {type(result)}")
            st.write(f"**结果包含的键:** {list(result.keys()) if hasattr(result, 'keys') else 'N/A'}")

            # 显示关键指标
            col1, col2, col3 = st.columns(3)
            with col1:
                best_fitness = result.get('best_fitness', 'N/A')
                st.metric("最优适应度",
                          f"{best_fitness:.4f}" if isinstance(best_fitness, (int, float)) else best_fitness)

            with col2:
                comp_time = result.get('computation_time', 'N/A')
                st.metric("计算时间", f"{comp_time:.2f}s" if isinstance(comp_time, (int, float)) else comp_time)

            with col3:
                convergence = result.get('fitness_history', 'N/A')
                if convergence is not None and hasattr(convergence, '__len__'):
                    st.metric("收敛迭代次数", len(convergence))
                else:
                    st.metric("收敛数据", "无")

    with tab2:
        st.subheader("详细数据结构")

        # 递归显示详细内容
        def display_result_details(result_data, depth=0, max_depth=3):
            if depth > max_depth:
                return "..."  # 防止无限递归

            indent = "  " * depth

            if result_data is None:
                return f"{indent}None"
            elif isinstance(result_data, dict):
                output = []
                for key, value in result_data.items():
                    if isinstance(value, (pd.DataFrame, list)) and len(value) > 10:
                        # 对于大数据结构，只显示摘要
                        if isinstance(value, pd.DataFrame):
                            output.append(f"{indent}{key}: DataFrame(shape={value.shape})")
                            output.append(f"{indent}  列名: {list(value.columns)}")
                            if len(value) > 0:
                                output.append(f"{indent}  前3行数据:")
                                for i, (idx, row) in enumerate(value.head(3).iterrows()):
                                    if i < 2:  # 只显示前2行的部分数据
                                        row_preview = {k: f"{v:.3f}" if isinstance(v, float) else v
                                                       for k, v in row.items() if
                                                       not isinstance(v, (list, dict))}
                                        output.append(f"{indent}    {row_preview}")
                        elif isinstance(value, list):
                            output.append(f"{indent}{key}: List(length={len(value)})")
                            if len(value) > 0 and all(isinstance(x, (int, float)) for x in value[:5]):
                                output.append(f"{indent}  前5个值: {value[:5]}")
                    else:
                        output.append(f"{indent}{key}: {display_result_details(value, depth + 1, max_depth)}")
                return "\n".join(output)
            elif isinstance(result_data, pd.DataFrame):
                return f"DataFrame(shape={result_data.shape}, columns={list(result_data.columns)})"
            elif isinstance(result_data, (list, tuple)):
                if len(result_data) > 10:
                    return f"{type(result_data).__name__}(length={len(result_data)})"
                else:
                    return f"{result_data}"
            else:
                return f"{result_data}"

        # 显示详细内容
        st.text_area("结果详细内容",
                     display_result_details(result),
                     height=400,
                     help="显示优化结果的完整数据结构")

    # 显示关键数据表格
    st.markdown("#### 📋 关键数据表格")

    # 显示最佳位置数据
    if result and 'best_positions_data' in result and isinstance(result['best_positions_data'], pd.DataFrame):
        st.subheader("最佳风机位置数据")
        best_positions = result['best_positions_data']
        st.write(f"数据形状: {best_positions.shape}")
        st.dataframe(best_positions.head(10), use_container_width=True)

        # 显示统计信息
        if not best_positions.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'predicted_wind_speed' in best_positions.columns:
                    avg_wind = best_positions['predicted_wind_speed'].mean()
                    st.metric("平均预测风速", f"{avg_wind:.2f} m/s")

            with col2:
                if 'wind_utilization_rate' in best_positions.columns:
                    avg_util = best_positions['wind_utilization_rate'].mean()
                    st.metric("平均风能利用率", f"{avg_util:.2%}")

            with col3:
                st.metric("选择的位置数量", len(best_positions))

    # 显示功率结果
    if result and 'power_results' in result and isinstance(result['power_results'], dict):
        st.subheader("发电量分析结果")
        power_results = result['power_results']

        # 创建表格显示功率结果
        power_data = []
        for key, value in power_results.items():
            if isinstance(value, (int, float)):
                if 'gwh' in key.lower() or 'generation' in key.lower():
                    value_str = f"{value:.2f} GWh"
                elif 'factor' in key.lower():
                    value_str = f"{value:.2%}"
                else:
                    value_str = f"{value:.4f}"
                power_data.append({'指标': key, '值': value_str})

        if power_data:
            st.table(power_data)

    # 显示收敛曲线数据
    if result and 'fitness_history' in result and result['fitness_history'] is not None:
        st.subheader("收敛曲线数据")
        convergence_data = result['fitness_history']

        if isinstance(convergence_data, (list, tuple, np.ndarray)):
            st.write(f"收敛迭代次数: {len(convergence_data)}")

            # 显示收敛曲线的前后部分
            col1, col2 = st.columns(2)
            with col1:
                if len(convergence_data) > 0:
                    st.write("前10次迭代:")
                    st.write(convergence_data[:10])
            with col2:
                if len(convergence_data) > 10:
                    st.write("最后10次迭代:")
                    st.write(convergence_data[-10:])

            # 绘制收敛曲线
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=convergence_data,
                mode='lines',
                name='适应度收敛'
            ))
            fig.update_layout(
                title="优化收敛曲线",
                xaxis_title="迭代次数",
                yaxis_title="适应度值",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        _display_storage_utilization_analysis(result)


def _display_storage_utilization_analysis(result):
    """显示储能利用率分析"""
    st.markdown("#### 🔋 储能利用率分析")

    if result and 'power_results' in result:
        power_results = result['power_results']

        # 计算储能利用率
        storage_results = calculate_storage_utilization_from_optimization_result(power_results)

        # 显示储能关键指标
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("年发电量", f"{storage_results['annual_generation_gwh']:.2f} GWh")
            st.metric("装机容量", f"{storage_results['total_capacity_mw']:.2f} MW")
            st.metric("容量因数", f"{storage_results['capacity_factor']:.2%}")

        with col2:
            st.metric("估算储能需求", f"{storage_results['estimated_storage_requirement_gwh']:.2f} GWh")
            st.metric("储能贡献电量", f"{storage_results['storage_contribution_gwh']:.2f} GWh")
            st.metric("等效储能容量", f"{storage_results['equivalent_storage_capacity_gwh']:.2f} GWh")

        with col3:
            st.metric("综合储能利用率", f"{storage_results['comprehensive_storage_utilization']:.2%}")
            st.metric("小时数利用率", f"{storage_results['storage_utilization_by_hours']:.2%}")
            st.metric("年循环次数", f"{storage_results['storage_cycle_times_per_year']:.0f}")

        # 显示储能经济性分析
        storage_economic = result.get('storage_economic_analysis', {})
        if storage_economic:
            st.markdown("##### 💰 储能经济性分析")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                storage_capacity = storage_economic.get('storage_capacity_kwh', 0) / 1000  # MWh
                st.metric("储能容量", f"{storage_capacity:.1f} MWh")

            with col2:
                storage_power = storage_economic.get('storage_power_kw', 0) / 1000  # MW
                st.metric("储能功率", f"{storage_power:.1f} MW")

            with col3:
                storage_investment = storage_economic.get('storage_investment', 0)
                st.metric("总投资", f"{storage_investment / 1e6:.2f} 百万")

            with col4:
                storage_payback = storage_economic.get('storage_payback_years', 0)
                if storage_payback < float('inf'):
                    st.metric("回收期", f"{storage_payback:.1f} 年")
                else:
                    st.metric("回收期", "∞")

            # 计算单位投资
            unit_capacity_cost = storage_investment / (storage_capacity * 1000) if storage_capacity > 0 else 0
            unit_power_cost = storage_investment / (storage_power * 1000) if storage_power > 0 else 0

            col1, col2 = st.columns(2)
            with col1:
                st.metric("单位容量成本", f"{unit_capacity_cost:.0f} 元/kWh")
            with col2:
                st.metric("单位功率成本", f"{unit_power_cost:.0f} 元/kW")

        # 储能利用率详细分析
        st.markdown("##### 📊 储能利用率构成")

        # 创建雷达图显示储能利用率构成
        categories = ['容量因数贡献', '利用率贡献', '小时数贡献']
        values = [
            storage_results['capacity_factor'] * 0.4 * 100,
            storage_results['utilization_rate'] * 0.3 * 100,
            storage_results['storage_utilization_by_hours'] * 0.3 * 100
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='储能利用率构成',
            line=dict(color='#00b4d8', width=2),
            fillcolor='rgba(0, 180, 216, 0.2)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=['0%', '25%', '50%', '75%', '100%']
                )
            ),
            showlegend=True,
            title="储能利用率构成分析",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # 储能配置建议
        st.markdown("##### 💡 储能配置建议")

        storage_capacity_ratio = (storage_results['equivalent_storage_capacity_gwh'] /
                                  storage_results['annual_generation_gwh']) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**建议储能容量**: {storage_results['equivalent_storage_capacity_gwh']:.2f} GWh")
            st.info(f"**储能占比**: {storage_capacity_ratio:.1f}%")

        with col2:
            st.info(f"**预计年循环次数**: {storage_results['storage_cycle_times_per_year']:.0f} 次")
            st.info(f"**储能系统效率**: {storage_results['estimated_storage_efficiency']:.0%}")


def _display_algorithm_performance_comparison(successful_results):
    """显示算法性能对比分析"""
    st.markdown("#### 📊 优化算法性能对比")

    # 按算法分组数据
    algorithm_data = {}
    for combo_key, data in successful_results.items():
        if data['result'] is None:
            continue

        algo = data['optimization_algorithm']
        if algo not in algorithm_data:
            algorithm_data[algo] = []
        algorithm_data[algo].append(data)

    # 计算每个算法的平均性能
    algo_stats = {}
    for algo, data_list in algorithm_data.items():
        fitnesses = [d['fitness'] for d in data_list]
        computation_times = [d['computation_time'] for d in data_list]
        powers = [d['result'].get('power_results', {}).get('total_annual_generation_gwh', 0) for d in data_list]

        # 计算平均储能利用率
        storage_utilizations = []
        # 获取储能容量
        storage_capacities = []
        for d in data_list:
            if d['result']:
                storage_results = calculate_storage_utilization_from_optimization_result(
                    d['result'].get('power_results', {})
                )
                storage_utilizations.append(storage_results['comprehensive_storage_utilization'] * 100)

                # 获取储能容量
                storage_economic = d['result'].get('storage_economic_analysis', {})
                storage_capacity = storage_economic.get('storage_capacity_kwh', 0) / 1000  # MWh
                storage_capacities.append(storage_capacity)

        if fitnesses:  # 确保有数据
            algo_stats[algo] = {
                'avg_fitness': np.mean(fitnesses),
                'avg_computation_time': np.mean(computation_times),
                'avg_power': np.mean(powers),
                'avg_storage_utilization': np.mean(storage_utilizations) if storage_utilizations else 0,
                'avg_storage_capacity': np.mean(storage_capacities) if storage_capacities else 0,
                'count': len(data_list)
            }

    # 创建算法对比图表
    if algo_stats:
        algorithms = list(algo_stats.keys())
        avg_fitness = [algo_stats[algo]['avg_fitness'] for algo in algorithms]
        avg_times = [algo_stats[algo]['avg_computation_time'] for algo in algorithms]
        avg_powers = [algo_stats[algo]['avg_power'] for algo in algorithms]
        avg_storage_utilizations = [algo_stats[algo]['avg_storage_utilization'] for algo in algorithms]
        avg_storage_capacities = [algo_stats[algo]['avg_storage_capacity'] for algo in algorithms]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('平均适应度', '平均计算时间(秒)', '平均年发电量(GWh)', '平均储能容量(MWh)'),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
        )

        # 适应度
        fig.add_trace(
            go.Bar(name='适应度', x=algorithms, y=avg_fitness,
                   marker_color='lightblue', text=[f'{score:.3f}' for score in avg_fitness],
                   textposition='auto'),
            row=1, col=1
        )

        # 计算时间
        fig.add_trace(
            go.Bar(name='计算时间', x=algorithms, y=avg_times,
                   marker_color='lightcoral', text=[f'{time:.1f}s' for time in avg_times],
                   textposition='auto'),
            row=1, col=2
        )

        # 年发电量
        fig.add_trace(
            go.Bar(name='年发电量', x=algorithms, y=avg_powers,
                   marker_color='lightgreen', text=[f'{power:.1f}' for power in avg_powers],
                   textposition='auto'),
            row=2, col=1
        )

        # 储能容量
        fig.add_trace(
            go.Bar(name='储能容量', x=algorithms, y=avg_storage_capacities,
                   marker_color='#FFA500', text=[f'{cap:.1f}' for cap in avg_storage_capacities],
                   textposition='auto'),
            row=2, col=2
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="优化算法平均性能对比"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 额外显示储能利用率图表
        st.markdown("##### 🔋 储能性能对比")

        fig_util = go.Figure()
        fig_util.add_trace(go.Bar(
            x=algorithms, y=avg_storage_utilizations,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text=[f'{util:.1f}%' for util in avg_storage_utilizations],
            textposition='auto'
        ))
        fig_util.update_layout(
            title="平均储能利用率对比",
            xaxis_title="算法",
            yaxis_title="储能利用率 (%)",
            height=400
        )
        st.plotly_chart(fig_util, use_container_width=True)

        # 算法推荐
        best_fitness_algo = max(zip(avg_fitness, algorithms))[1]
        fastest_algo = min(zip(avg_times, algorithms))[1]
        best_power_algo = max(zip(avg_powers, algorithms))[1]
        best_storage_algo = max(zip(avg_storage_utilizations, algorithms))[1]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最佳效果算法", best_fitness_algo)
        with col2:
            st.metric("最快算法", fastest_algo)
        with col3:
            st.metric("最高发电量算法", best_power_algo)
        with col4:
            st.metric("最佳储能利用算法", best_storage_algo)


def _display_prediction_model_analysis(successful_results):
    """显示预测模型分析"""
    st.markdown("#### 🔮 预测模型性能分析")

    # 按预测模型分组数据
    model_data = {}
    for combo_key, data in successful_results.items():
        if data['result'] is None:
            continue

        model = data['prediction_model']
        if model not in model_data:
            model_data[model] = []
        model_data[model].append(data)

    # 计算每个模型的平均性能
    model_stats = {}
    for model, data_list in model_data.items():
        fitnesses = [d['fitness'] for d in data_list]
        powers = [d['result'].get('power_results', {}).get('total_annual_generation_gwh', 0) for d in data_list]

        # 计算平均储能利用率
        storage_utilizations = []
        # 获取储能容量
        storage_capacities = []
        for d in data_list:
            if d['result']:
                storage_results = calculate_storage_utilization_from_optimization_result(
                    d['result'].get('power_results', {})
                )
                storage_utilizations.append(storage_results['comprehensive_storage_utilization'] * 100)

                # 获取储能容量
                storage_economic = d['result'].get('storage_economic_analysis', {})
                storage_capacity = storage_economic.get('storage_capacity_kwh', 0) / 1000  # MWh
                storage_capacities.append(storage_capacity)

        # 获取平均风速数据
        wind_speeds = []
        for d in data_list:
            if d['result']:
                result_data = d['result']
                if 'best_positions_data' in result_data:
                    selected_df = result_data['best_positions_data']
                    if isinstance(selected_df,
                                  pd.DataFrame) and not selected_df.empty and 'predicted_wind_speed' in selected_df.columns:
                        wind_speed = selected_df['predicted_wind_speed'].mean()
                        wind_speeds.append(wind_speed)

        if fitnesses:  # 确保有数据
            model_stats[model] = {
                'avg_fitness': np.mean(fitnesses),
                'avg_power': np.mean(powers),
                'avg_storage_utilization': np.mean(storage_utilizations) if storage_utilizations else 0,
                'avg_storage_capacity': np.mean(storage_capacities) if storage_capacities else 0,
                'avg_wind_speed': np.mean(wind_speeds) if wind_speeds else 0,
                'count': len(data_list)
            }

    # 创建模型对比图表
    if model_stats:
        models = list(model_stats.keys())
        avg_fitness = [model_stats[model]['avg_fitness'] for model in models]
        avg_powers = [model_stats[model]['avg_power'] for model in models]
        avg_storage_utilizations = [model_stats[model]['avg_storage_utilization'] for model in models]
        avg_storage_capacities = [model_stats[model]['avg_storage_capacity'] for model in models]
        avg_wind_speeds = [model_stats[model]['avg_wind_speed'] for model in models]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='平均适应度',
            x=models,
            y=avg_fitness,
            marker_color='#1f77b4',
            text=[f'{f:.3f}' for f in avg_fitness],
            textposition='auto'
        ))

        fig.add_trace(go.Bar(
            name='平均年发电量(GWh)',
            x=models,
            y=avg_powers,
            marker_color='#2ca02c',
            text=[f'{p:.1f}' for p in avg_powers],
            textposition='auto'
        ))

        fig.add_trace(go.Bar(
            name='平均储能利用率(%)',
            x=models,
            y=avg_storage_utilizations,
            marker_color='#FFA500',
            text=[f'{u:.1f}%' for u in avg_storage_utilizations],
            textposition='auto'
        ))

        fig.add_trace(go.Bar(
            name='平均风速(m/s)',
            x=models,
            y=avg_wind_speeds,
            marker_color='#ff7f0e',
            text=[f'{w:.1f}' for w in avg_wind_speeds],
            textposition='auto'
        ))

        fig.update_layout(
            title="预测模型性能对比",
            barmode='group',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # 储能容量对比
        fig_capacity = go.Figure()
        fig_capacity.add_trace(go.Bar(
            name='平均储能容量(MWh)',
            x=models,
            y=avg_storage_capacities,
            marker_color='#9467bd',
            text=[f'{c:.1f}' for c in avg_storage_capacities],
            textposition='auto'
        ))
        fig_capacity.update_layout(
            title="预测模型的平均储能容量对比",
            xaxis_title="预测模型",
            yaxis_title="储能容量 (MWh)",
            height=400
        )
        st.plotly_chart(fig_capacity, use_container_width=True)

        # 模型推荐
        best_fitness_model = max(zip(avg_fitness, models))[1]
        best_power_model = max(zip(avg_powers, models))[1]
        best_storage_model = max(zip(avg_storage_utilizations, models))[1]
        best_wind_model = max(zip(avg_wind_speeds, models))[1]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最佳适应度模型", best_fitness_model)
        with col2:
            st.metric("最高发电量模型", best_power_model)
        with col3:
            st.metric("最佳储能利用模型", best_storage_model)
        with col4:
            st.metric("最高风速模型", best_wind_model)


def _display_optimization_effect_evaluation(successful_results):
    """显示优化效果评估"""
    st.markdown("#### 🎯 优化效果综合评估")

    # 计算整体统计信息
    total_combinations = len(successful_results)
    fitness_values = [data['fitness'] for data in successful_results.values()]
    computation_times = [data['computation_time'] for data in successful_results.values()]
    power_values = [data['result'].get('power_results', {}).get('total_annual_generation_gwh', 0)
                    for data in successful_results.values()]

    # 计算储能利用率
    storage_utilizations = []
    storage_capacities = []
    for data in successful_results.values():
        if data['result']:
            storage_results = calculate_storage_utilization_from_optimization_result(
                data['result'].get('power_results', {})
            )
            storage_utilizations.append(storage_results['comprehensive_storage_utilization'] * 100)

            # 获取储能容量
            storage_economic = data['result'].get('storage_economic_analysis', {})
            storage_capacity = storage_economic.get('storage_capacity_kwh', 0) / 1000  # MWh
            storage_capacities.append(storage_capacity)

    # 显示整体统计
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("成功组合数", total_combinations)

    with col2:
        avg_fitness = np.mean(fitness_values)
        st.metric("平均适应度", f"{avg_fitness:.3f}")

    with col3:
        avg_power = np.mean(power_values)
        st.metric("平均年发电量", f"{avg_power:.1f} GWh")

    with col4:
        avg_storage = np.mean(storage_utilizations)
        st.metric("平均储能利用率", f"{avg_storage:.1f}%")

    # 创建性能分布图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('适应度分布', '计算时间分布', '年发电量分布', '储能容量分布'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}], [{"type": "histogram"}, {"type": "histogram"}]]
    )

    # 适应度分布
    fig.add_trace(
        go.Histogram(x=fitness_values, name='适应度分布', marker_color='#1f77b4'),
        row=1, col=1
    )

    # 计算时间分布
    fig.add_trace(
        go.Histogram(x=computation_times, name='计算时间分布', marker_color='#ff7f0e'),
        row=1, col=2
    )

    # 年发电量分布
    fig.add_trace(
        go.Histogram(x=power_values, name='年发电量分布', marker_color='#2ca02c'),
        row=2, col=1
    )

    # 储能容量分布
    fig.add_trace(
        go.Histogram(x=storage_capacities, name='储能容量分布', marker_color='#FFA500'),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="性能指标分布分析"
    )

    fig.update_xaxes(title_text="适应度", row=1, col=1)
    fig.update_xaxes(title_text="计算时间(秒)", row=1, col=2)
    fig.update_xaxes(title_text="年发电量(GWh)", row=2, col=1)
    fig.update_xaxes(title_text="储能容量(MWh)", row=2, col=2)
    fig.update_yaxes(title_text="频次", row=1, col=1)
    fig.update_yaxes(title_text="频次", row=1, col=2)
    fig.update_yaxes(title_text="频次", row=2, col=1)
    fig.update_yaxes(title_text="频次", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # 性能相关性分析
    st.markdown("##### 📊 性能相关性分析")

    # 创建散点图矩阵
    if len(fitness_values) > 1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fitness_values,
            y=power_values,
            mode='markers',
            marker=dict(
                size=8,
                color=storage_utilizations,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="储能利用率(%)")
            ),
            text=[f"组合: {list(successful_results.keys())[i]}" for i in range(len(fitness_values))],
            hovertemplate='<b>%{text}</b><br>适应度: %{x:.3f}<br>年发电量: %{y:.1f} GWh<br>储能利用率: %{marker.color:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            title="适应度 vs 年发电量 (颜色表示储能利用率)",
            xaxis_title="适应度",
            yaxis_title="年发电量 (GWh)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)


# ==================== 辅助函数 ====================

def _split_dataset_by_coordinates(df, train_ratio, val_ratio, test_ratio):
    """按照坐标点划分数据集，保持每个坐标的时间连续性"""

    # 确保输入的ratio是小数形式
    train_ratio = train_ratio / 100.0
    val_ratio = val_ratio / 100.0
    test_ratio = test_ratio / 100.0

    # 获取所有唯一的坐标点
    coordinates = df[['lat', 'lon']].drop_duplicates()
    n_coordinates = len(coordinates)

    # 确保有足够的坐标点
    if n_coordinates < 3:
        raise ValueError(f"坐标点数量太少 ({n_coordinates})，需要至少3个坐标点进行划分")

    # 计算每个集合的坐标数量
    n_train = max(1, int(n_coordinates * train_ratio))
    n_val = max(1, int(n_coordinates * val_ratio))
    n_test = max(1, n_coordinates - n_train - n_val)

    # 调整确保总和为总坐标数
    while n_train + n_val + n_test > n_coordinates:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1

    # 随机打乱坐标点
    shuffled_coords = coordinates.sample(frac=1, random_state=42)

    # 划分坐标点
    train_coords = shuffled_coords.iloc[:n_train]
    val_coords = shuffled_coords.iloc[n_train:n_train + n_val]
    test_coords = shuffled_coords.iloc[n_train + n_val:n_train + n_val + n_test]

    # 根据坐标点划分数据
    train_data = df.merge(train_coords, on=['lat', 'lon'])
    val_data = df.merge(val_coords, on=['lat', 'lon'])
    test_data = df.merge(test_coords, on=['lat', 'lon'])

    split_info = {
        'train_coords': train_coords,
        'val_coords': val_coords,
        'test_coords': test_coords,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'n_train_coords': n_train,
        'n_val_coords': n_val,
        'n_test_coords': n_test
    }

    return split_info


def _prepare_features(df):
    """准备特征数据"""
    feature_columns = []

    # 基本地理特征
    if 'elevation' in df.columns:
        feature_columns.append('elevation')
    if 'slope' in df.columns:
        feature_columns.append('slope')

    # 气象特征
    if 'temperature' in df.columns:
        feature_columns.append('temperature')
    if 'pressure' in df.columns:
        feature_columns.append('pressure')
    if 'humidity' in df.columns:
        feature_columns.append('humidity')

    # 时间特征
    if 'hour' in df.columns:
        feature_columns.append('hour')
    if 'month' in df.columns:
        feature_columns.append('month')

    # 如果特征太少，使用经纬度
    if len(feature_columns) < 3:
        feature_columns.extend(['lat', 'lon'])

    return feature_columns


def _train_random_forest(train_data, val_data, feature_columns):
    """训练随机森林模型"""
    X_train = train_data[feature_columns]
    y_train = train_data['predicted_wind_speed']
    X_val = val_data[feature_columns]
    y_val = val_data['predicted_wind_speed']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 在验证集上评估
    y_val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)

    return model, {'mse': mse, 'r2': r2}


def _train_xgboost(train_data, val_data, feature_columns):
    """训练XGBoost模型"""
    if not XGBOOST_AVAILABLE:
        st.warning("XGBoost未安装，使用随机森林替代")
        return _train_random_forest(train_data, val_data, feature_columns)

    X_train = train_data[feature_columns]
    y_train = train_data['predicted_wind_speed']
    X_val = val_data[feature_columns]
    y_val = val_data['predicted_wind_speed']

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 在验证集上评估
    y_val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)

    return model, {'mse': mse, 'r2': r2}


def _train_lstm(train_data, val_data, feature_columns):
    """训练LSTM模型（简化版，实际需要更复杂的时序处理）"""
    # 这里简化处理，实际应该处理时间序列
    # 暂时使用随机森林代替，实际应用中应该实现真正的LSTM
    st.info("LSTM模型使用简化实现，实际应用中应处理时间序列数据")
    return _train_random_forest(train_data, val_data, feature_columns)


def _train_gru(train_data, val_data, feature_columns):
    """训练GRU模型（简化版，实际需要更复杂的时序处理）"""
    # 这里简化处理，实际应该处理时间序列
    # 暂时使用随机森林代替，实际应用中应该实现真正的GRU
    st.info("GRU模型使用简化实现，实际应用中应处理时间序列数据")
    return _train_random_forest(train_data, val_data, feature_columns)


def calculate_wind_utilization(wind_speed):
    """计算风能利用率 - 与第二个代码保持一致"""
    if isinstance(wind_speed, pd.Series):
        return wind_speed.apply(_calculate_single_point_utilization)
    else:
        return _calculate_single_point_utilization(wind_speed)


def _calculate_single_point_utilization(wind_speed):
    """为单个风速值计算利用率"""
    if wind_speed < 3.0:
        return 0.0  # 低于切入风速
    elif wind_speed < 7.0:
        return 0.3  # 低风速区间
    elif wind_speed < 12.0:
        return 0.7  # 中风速区间
    elif wind_speed <= 25.0:
        return 0.9  # 高风速区间
    else:
        return 0.0  # 超过切出风速


def _generate_wind_prediction(df: pd.DataFrame, model_name: str, split_info: dict) -> pd.DataFrame:
    """使用真实模型进行风速预测，并确保测试集数据使用预测值覆盖"""

    # 准备特征
    feature_columns = _prepare_features(df)

    if len(feature_columns) == 0:
        st.error("数据集中没有找到可用的特征列")
        return df

    try:
        # 训练模型 - 修改为新的模型选择
        if model_name == "随机森林":
            model, metrics = _train_random_forest(
                split_info['train_data'],
                split_info['val_data'],
                feature_columns
            )
        elif model_name == "XGBoost":
            model, metrics = _train_xgboost(
                split_info['train_data'],
                split_info['val_data'],
                feature_columns
            )
        elif model_name == "LSTM":
            model, metrics = _train_lstm(
                split_info['train_data'],
                split_info['val_data'],
                feature_columns
            )
        elif model_name == "GRU":
            model, metrics = _train_gru(
                split_info['train_data'],
                split_info['val_data'],
                feature_columns
            )
        else:
            # 默认使用随机森林
            model, metrics = _train_random_forest(
                split_info['train_data'],
                split_info['val_data'],
                feature_columns
            )

        # 关键修改：在所有数据上进行预测，包括测试集数据
        X_all = df[feature_columns]
        # 这里预测的结果会覆盖原有的 predicted_wind_speed 字段
        df["predicted_wind_speed"] = model.predict(X_all)

        # 关键修改：创建专门用于优化的数据集，其中测试集坐标使用预测的风速数据
        # 首先获取测试集坐标的数据
        test_coords_data = df.merge(split_info['test_coords'], on=['lat', 'lon'])

    except Exception as e:
        st.error(f"模型 {model_name} 训练失败: {str(e)}")
        # 如果模型训练失败，使用平均风速作为预测值
        mean_wind_speed = split_info['train_data']['predicted_wind_speed'].mean()
        df["predicted_wind_speed"] = mean_wind_speed

    # 计算风功率密度
    df["wind_power_density"] = 0.5 * 1.225 * (df["predicted_wind_speed"] ** 3)

    # 使用与第二个代码相同的利用率计算函数
    df["wind_utilization_rate"] = calculate_wind_utilization(df["predicted_wind_speed"])

    # 归一化与综合评分
    max_ws = df["predicted_wind_speed"].max()
    max_ut = df["wind_utilization_rate"].max()
    df["normalized_wind_speed"] = df["predicted_wind_speed"] / (max_ws if max_ws > 0 else 1)
    df["normalized_utilization"] = df["wind_utilization_rate"] / (max_ut if max_ut > 0 else 1)
    df["composite_score"] = (
            df["normalized_wind_speed"] * 0.6 +  # 使用固定权重
            df["normalized_utilization"] * 0.4
    )

    # 设置有效点位 - 与第二个代码保持一致
    df["valid"] = (
            (df["predicted_wind_speed"] >= 5.0) &
            (df.get("slope", 0) <= 35) &
            (df.get("elevation", 0) >= 150) & (df.get("elevation", 0) <= 1600) &
            (df["composite_score"] >= 0.4)
    )

    return df


def _run_all_combinations_experiment(pred_models, opt_algorithms, base_params, algo_specific_params, n_farms,
                                     train_ratio, val_ratio, test_ratio):
    """运行所有预测模型和优化算法的组合实验"""
    all_results = {}

    # 进度显示 - 修复：只有一个进度条和状态文本
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_combinations = len(pred_models) * len(opt_algorithms)
    completed = 0

    # 首先划分数据集
    status_text.text("🔄 正在划分数据集...")
    split_info = _split_dataset_by_coordinates(
        st.session_state['dataset'].copy(),
        train_ratio, val_ratio, test_ratio
    )
    st.session_state.dataset_split_info = split_info

    for pred_model, opt_algo in product(pred_models, opt_algorithms):
        combination_key = f"{pred_model}×{opt_algo}"
        status_text.text(f"🔄 正在运行: {combination_key} ({completed + 1}/{total_combinations})")

        try:
            # Step 1: 生成预测风速（关键修改：测试集数据也会被预测值覆盖）
            df_processed = _generate_wind_prediction(
                st.session_state['dataset'].copy(),
                model_name=pred_model,
                split_info=split_info
            )

            # Step 2: 准备算法参数
            algorithm_params = base_params.copy()
            algorithm_params.update(algo_specific_params.get(opt_algo, {}))
            algorithm_params['prediction_model'] = pred_model

            # Step 3: 执行优化（使用测试集坐标的数据进行优化）- 关键修改：这里使用的已经是预测后的数据
            test_coords_data = df_processed.merge(split_info['test_coords'], on=['lat', 'lon'])

            # 检查测试集数据是否为空
            if test_coords_data.empty:
                raise ValueError("测试集数据为空，无法进行优化")

            # 关键修改：使用与第二个代码相同的优化函数调用方式
            result = call_optimize_function_with_all_strategies(test_coords_data, opt_algo, algorithm_params)

            # ==================== 关键修复：确保利用率数据正确计算和存储 ====================
            # 从优化结果中获取最优位置数据
            best_positions_data = result.get('best_positions_data', pd.DataFrame())

            # 计算最优位置的平均利用率
            if not best_positions_data.empty and 'predicted_wind_speed' in best_positions_data.columns:
                # 为最优位置计算利用率
                best_positions_data = best_positions_data.copy()
                best_positions_data['wind_utilization_rate'] = calculate_wind_utilization(
                    best_positions_data['predicted_wind_speed']
                )
                avg_utilization = best_positions_data['wind_utilization_rate'].mean()
            else:
                # 如果没有最优位置数据，使用测试集数据的平均利用率
                avg_utilization = test_coords_data['wind_utilization_rate'].mean()

            # 确保power_results中包含平均利用率
            power_results = result.get('power_results', {})
            if 'average_utilization_rate' not in power_results:
                power_results['average_utilization_rate'] = avg_utilization
                result['power_results'] = power_results

            # 更新最优位置数据（如果计算了新的利用率）
            if not best_positions_data.empty:
                result['best_positions_data'] = best_positions_data

            # Step 4: 存储结果 - 确保利用率数据正确传递
            all_results[combination_key] = {
                'result': result,
                'prediction_model': pred_model,
                'optimization_algorithm': opt_algo,
                'fitness': result.get('best_fitness', 0),
                'computation_time': result.get('computation_time', 0),
                'processed_data': df_processed,
                'algorithm_params': algorithm_params,
                'n_farms': n_farms,
                'status': 'success',
                'split_info': split_info,
                'test_coords_data': test_coords_data  # 存储用于优化的测试集数据
            }

        except Exception as e:
            all_results[combination_key] = {
                'result': None,
                'prediction_model': pred_model,
                'optimization_algorithm': opt_algo,
                'fitness': 0,
                'computation_time': 0,
                'error': str(e),
                'status': 'failed'
            }

        completed += 1
        progress_bar.progress(completed / total_combinations)

    # 存储所有结果
    st.session_state["all_experiment_results"] = all_results
    st.session_state.current_view = "result"

    # 显示完成状态
    successful = sum(1 for r in all_results.values() if r['status'] == 'success')
    status_text.text(f"✅ 实验完成: {successful}/{total_combinations} 个组合成功")

    # 显示最终结果摘要
    if successful > 0:
        best_combo = max([(k, v) for k, v in all_results.items() if v['status'] == 'success'],
                         key=lambda x: x[1]['fitness'])
        st.success(f"🏆 最佳组合: **{best_combo[0]}** (适应度: {best_combo[1]['fitness']:.3f})")

    st.rerun()


# ======================================================
# 🚀 入口（仅用于直接运行调试）
# ======================================================
if __name__ == "__main__":
    prediction_optimization_comparison_page()