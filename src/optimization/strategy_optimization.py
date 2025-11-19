from pickletools import optimize

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.create_map import display_fengjie_standalone_map, display_environment, display_optimization_map
from src.visualization.opt_result_show import display_optimization_result


# ======================================================
# 🔋 新增函数：风机充放电策略
# ======================================================
def calculate_power_from_wind_speed(wind_speed, rated_power=2000, cut_in_speed=3.0, rated_speed=12.0,
                                    cut_out_speed=25.0):
    """
    根据风速计算风机功率
    """
    if wind_speed < cut_in_speed or wind_speed > cut_out_speed:
        return 0
    elif wind_speed >= rated_speed:
        return rated_power
    else:
        # 功率曲线：在切入风速和额定风速之间按立方关系计算
        return rated_power * ((wind_speed - cut_in_speed) / (rated_speed - cut_in_speed)) ** 3


def generate_wind_power_time_series(base_wind_speed, time_horizon=24):
    """
    基于基础风速生成功率时间序列，考虑昼夜变化和随机波动
    """
    time_series_power = []

    for hour in range(time_horizon):
        # 昼夜变化因子（白天风大，晚上风小）
        diurnal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour / 24 - np.pi / 2)
        # 随机波动
        random_factor = np.random.normal(1, 0.15)
        # 计算当前小时风速
        hour_wind_speed = base_wind_speed * diurnal_factor * random_factor
        # 计算功率
        hour_power = calculate_power_from_wind_speed(hour_wind_speed)
        time_series_power.append(max(hour_power, 0))

    return time_series_power


def turbine_charge_discharge_strategy(turbine_power, turbine_id, storage_capacity_kwh=1000,
                                      max_charge_rate_kw=200, time_horizon=24):
    """
    单个风机的充放电策略
    返回：可消纳电量、弃风比例、SOC曲线等
    """
    # 初始化变量
    storage_soc = storage_capacity_kwh * 0.5  # 初始SOC为50%
    storage_soc_history = [storage_soc]
    charge_power_history = []
    discharge_power_history = []
    net_power_history = []
    wind_curtailment_history = []

    strategy_log = []
    total_curtailment = 0
    total_original_power = sum(turbine_power)

    for t in range(time_horizon):
        current_power = turbine_power[t]

        # 充放电决策逻辑
        if current_power > max_charge_rate_kw and storage_soc < storage_capacity_kwh:
            # 高功率时充电
            charge_power = min(current_power - max_charge_rate_kw,
                               max_charge_rate_kw,
                               storage_capacity_kwh - storage_soc)
            discharge_power = 0
            net_power = max_charge_rate_kw
            storage_soc += charge_power
            curtailment = current_power - max_charge_rate_kw - charge_power
            action = "充电"

        elif current_power < 100 and storage_soc > 0:  # 低功率时放电
            # 可放电功率
            available_discharge = min(200, max_charge_rate_kw, storage_soc)
            discharge_power = available_discharge
            charge_power = 0
            net_power = current_power + discharge_power
            storage_soc -= discharge_power
            curtailment = 0
            action = "放电"

        else:
            # 正常发电
            charge_power = 0
            discharge_power = 0
            net_power = min(current_power, max_charge_rate_kw)
            curtailment = max(0, current_power - max_charge_rate_kw)
            action = "正常发电"

        total_curtailment += curtailment

        # 记录数据
        storage_soc_history.append(storage_soc)
        charge_power_history.append(charge_power)
        discharge_power_history.append(discharge_power)
        net_power_history.append(net_power)
        wind_curtailment_history.append(curtailment)

        strategy_log.append({
            '时间': t,
            '原始功率': current_power,
            '充电功率': charge_power,
            '放电功率': discharge_power,
            '净输出功率': net_power,
            '弃风功率': curtailment,
            'SOC': storage_soc,
            '动作': action
        })

    # 计算性能指标
    total_net_power = sum(net_power_history)
    utilization_rate = (
                               total_original_power - total_curtailment) / total_original_power if total_original_power > 0 else 0
    curtailment_rate = total_curtailment / total_original_power if total_original_power > 0 else 0

    performance_metrics = {
        '风机编号': turbine_id,
        '总发电量': total_original_power,
        '可消纳电量': total_net_power,
        '弃风电量': total_curtailment,
        '弃风比例': curtailment_rate,
        '风电利用率': utilization_rate,
        '充电次数': len([p for p in charge_power_history if p > 0]),
        '放电次数': len([p for p in discharge_power_history if p > 0]),
        '平均SOC': np.mean(storage_soc_history),
        'SOC波动': np.std(storage_soc_history)
    }

    return {
        'performance_metrics': performance_metrics,
        'time_series': {
            'storage_soc': storage_soc_history,
            'charge_power': charge_power_history,
            'discharge_power': discharge_power_history,
            'net_power': net_power_history,
            'original_power': turbine_power,
            'wind_curtailment': wind_curtailment_history
        },
        'strategy_log': strategy_log
    }


def power_smoothing_for_turbine(turbine_power, smoothing_window=4):
    """
    针对单个风机的功率平滑策略
    """
    smoothed_power = []
    for i in range(len(turbine_power)):
        start_idx = max(0, i - smoothing_window // 2)
        end_idx = min(len(turbine_power), i + smoothing_window // 2 + 1)
        window_power = turbine_power[start_idx:end_idx]
        smoothed_value = np.mean(window_power)
        smoothed_power.append(smoothed_value)

    # 计算平滑效果
    original_variance = np.var(turbine_power)
    smoothed_variance = np.var(smoothed_power)
    smoothing_effect = (original_variance - smoothed_variance) / original_variance if original_variance > 0 else 0

    return {
        'original_power': turbine_power,
        'smoothed_power': smoothed_power,
        'smoothing_effect': smoothing_effect
    }


def analyze_all_turbines_strategy(selected_locations, time_horizon=24):
    """
    分析所有风机的充放电策略
    """
    turbines_strategy = {}

    for i, (_, location) in enumerate(selected_locations.iterrows()):
        turbine_id = f"T{i + 1}"

        # 安全地获取数据，处理可能的列缺失
        base_wind_speed = location.get('predicted_wind_speed', 0)
        latitude = location.get('latitude', 0)
        longitude = location.get('longitude', 0)
        elevation = location.get('elevation', 0)

        # 生成功率时间序列
        turbine_power = generate_wind_power_time_series(base_wind_speed, time_horizon)

        # 执行充放电策略
        charge_discharge_result = turbine_charge_discharge_strategy(
            turbine_power, turbine_id, time_horizon=time_horizon
        )

        # 执行功率平滑策略
        smoothing_result = power_smoothing_for_turbine(turbine_power)

        # 存储结果
        turbines_strategy[turbine_id] = {
            'location_data': {
                'latitude': latitude,
                'longitude': longitude,
                'elevation': elevation,
                'base_wind_speed': base_wind_speed
            },
            'charge_discharge': charge_discharge_result,
            'smoothing': smoothing_result
        }

    return turbines_strategy


def display_turbines_strategy_analysis(turbines_strategy):
    """
    显示所有风机的充放电策略分析结果
    """
    st.markdown("## 🔋 各风机充放电策略分析")

    # 汇总表格
    st.markdown("### 📊 各风机性能汇总")
    summary_data = []
    for turbine_id, strategy in turbines_strategy.items():
        metrics = strategy['charge_discharge']['performance_metrics']
        summary_data.append({
            '风机编号': turbine_id,
            '基础风速(m/s)': f"{strategy['location_data']['base_wind_speed']:.1f}",
            '总发电量(kWh)': f"{metrics['总发电量']:.0f}",
            '可消纳电量(kWh)': f"{metrics['可消纳电量']:.0f}",
            '弃风比例': f"{metrics['弃风比例']:.2%}",
            '风电利用率': f"{metrics['风电利用率']:.2%}",
            '充放电次数': f"{metrics['充电次数']}/{metrics['放电次数']}",
            '平滑效果': f"{strategy['smoothing']['smoothing_effect']:.2%}"
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

    # 为每个风机创建详细分析
    tabs = st.tabs([f"风机 {turbine_id}" for turbine_id in turbines_strategy.keys()])

    for idx, (turbine_id, strategy) in enumerate(turbines_strategy.items()):
        with tabs[idx]:
            display_single_turbine_analysis(turbine_id, strategy)


def display_single_turbine_analysis(turbine_id, strategy):
    """
    显示单个风机的详细分析
    """
    st.markdown(f"### 🌀 风机 {turbine_id} 详细分析")

    location_data = strategy['location_data']
    charge_discharge = strategy['charge_discharge']
    smoothing = strategy['smoothing']
    metrics = charge_discharge['performance_metrics']

    # 基础信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("基础风速", f"{location_data['base_wind_speed']:.1f} m/s")
    with col2:
        st.metric("总发电量", f"{metrics['总发电量']:.0f} kWh")
    with col3:
        st.metric("可消纳电量", f"{metrics['可消纳电量']:.0f} kWh")
    with col4:
        st.metric("弃风比例", f"{metrics['弃风比例']:.2%}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("风电利用率", f"{metrics['风电利用率']:.2%}")
    with col6:
        st.metric("充放电次数", metrics['充放电次数'])
    with col7:
        st.metric("平均SOC", f"{metrics['平均SOC']:.0f} kWh")
    with col8:
        st.metric("平滑效果", f"{smoothing['smoothing_effect']:.2%}")

    # 功率曲线对比
    st.markdown("#### 📈 功率曲线分析")
    power_data = pd.DataFrame({
        '原始功率': charge_discharge['time_series']['original_power'],
        '净输出功率': charge_discharge['time_series']['net_power'],
        '平滑后功率': smoothing['smoothed_power']
    })
    st.line_chart(power_data, use_container_width=True)

    # SOC变化曲线
    st.markdown("#### 🔋 SOC变化曲线")
    soc_data = pd.DataFrame({
        'SOC': charge_discharge['time_series']['storage_soc']
    })
    st.area_chart(soc_data, use_container_width=True)

    # 充放电功率
    st.markdown("#### ⚡ 充放电功率")
    charge_discharge_data = pd.DataFrame({
        '充电功率': charge_discharge['time_series']['charge_power'],
        '放电功率': charge_discharge['time_series']['discharge_power'],
        '弃风功率': charge_discharge['time_series']['wind_curtailment']
    })
    st.bar_chart(charge_discharge_data, use_container_width=True)

    # 策略执行详情
    st.markdown("#### 📋 策略执行记录")
    strategy_df = pd.DataFrame(charge_discharge['strategy_log'])
    st.dataframe(strategy_df, use_container_width=True, height=300)

    # 策略建议
    display_turbine_recommendation(turbine_id, metrics, smoothing['smoothing_effect'])


def display_turbine_recommendation(turbine_id, metrics, smoothing_effect):
    """
    显示针对单个风机的策略建议
    """
    st.markdown("#### 💡 优化建议")

    if metrics['弃风比例'] > 0.2:
        st.warning(
            f"**⚠️ 风机 {turbine_id} 弃风严重**: 弃风比例{metrics['弃风比例']:.2%}，建议增加储能容量或优化充放电策略")
    elif metrics['弃风比例'] > 0.1:
        st.info(f"**🔶 风机 {turbine_id} 弃风较高**: 弃风比例{metrics['弃风比例']:.2%}，可考虑调整充放电阈值")
    else:
        st.success(f"**✅ 风机 {turbine_id} 运行良好**: 弃风比例{metrics['弃风比例']:.2%}，消纳效果优秀")

    if metrics['风电利用率'] < 0.7:
        st.warning("**风电利用率偏低**: 建议检查风机运行状态或优化控制策略")

    if smoothing_effect < 0.3:
        st.info("**功率波动较大**: 建议加强功率平滑控制")

    if metrics['充放电次数'].split('/')[0] == '0':
        st.info("**未执行充电操作**: 考虑优化充电策略以提高消纳能力")


# ======================================================
# 🔧 修改后的优化函数调用 - 添加错误处理
# ======================================================
def call_optimize_function(df, algo, algorithm_params):
    """调用优化函数，正确传递所有参数"""
    try:
        # 基础参数 - 包含所有约束条件
        base_params = {
            'df': df,
            'algo': algo,
            'n_turbines': algorithm_params['n_turbines'],
            'cost_weight': algorithm_params['cost_weight'],
            'max_slope': algorithm_params['max_slope'],
            'max_road_distance': algorithm_params['max_road_distance'],
            'min_residential_distance': algorithm_params['min_residential_distance'],
            'min_heritage_distance': algorithm_params['min_heritage_distance'],
            'min_geology_distance': algorithm_params['min_geology_distance'],
            'min_water_distance': algorithm_params['min_water_distance']
        }

        # 根据算法类型添加额外参数
        if algo == "遗传算法":
            extended_params = base_params.copy()
            extended_params.update({
                'pop_size': algorithm_params.get('pop_size', 50),
                'generations': algorithm_params.get('generations', 100),
                'mutation_rate': algorithm_params.get('mutation_rate', 0.1),
                'crossover_rate': algorithm_params.get('crossover_rate', 0.8)
            })
            result = optimize(**extended_params)

        elif algo == "模拟退火算法":
            extended_params = base_params.copy()
            extended_params.update({
                'initial_temp': algorithm_params.get('initial_temp', 1000),
                'cooling_rate': algorithm_params.get('cooling_rate', 0.95),
                'iterations_per_temp': algorithm_params.get('iterations_per_temp', 50)
            })
            result = optimize(**extended_params)

        elif algo == "粒子群优化算法":
            extended_params = base_params.copy()
            extended_params.update({
                'pop_size': algorithm_params.get('pop_size', 30),
                'generations': algorithm_params.get('generations', 100),
                'w': algorithm_params.get('w', 0.7),
                'c1': algorithm_params.get('c1', 1.5),
                'c2': algorithm_params.get('c2', 1.5)
            })
            result = optimize(**extended_params)

        elif algo == "PuLP优化求解器":
            extended_params = base_params.copy()
            extended_params.update({
                'solver_type': algorithm_params.get('solver_type', 'CBC'),
                'time_limit': algorithm_params.get('time_limit', 60)
            })
            result = optimize(**extended_params)

        else:  # 两者对比
            extended_params = base_params.copy()
            extended_params.update({
                'generations': algorithm_params.get('compare_generations', 100)
            })
            result = optimize(**extended_params)

        return result

    except Exception as e:
        st.error(f"优化函数调用失败: {str(e)}")
        # 返回一个默认结果结构
        return {
            'selected_locations': pd.DataFrame(),
            'best_fitness': 0,
            'convergence': []
        }


# ======================================================
# 🌬️ 主页面：风电场选址优化系统
# ======================================================
def strategy_optimization_page():
    # 页面标题 - 更紧凑
    st.markdown("### 🌬️ 风电场选址与充放电优化系统")
    st.caption("基于风机优化选址 + 个性化充放电策略 · 实现高稳定性电能输出")

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

        # 算法选择
        algo = st.selectbox("优化算法",
                            ["遗传算法", "模拟退火算法", "粒子群优化算法", "PuLP优化求解器", "两者对比"],
                            help="选择优化算法")

        # 算法参数
        st.markdown("**🔧 算法参数（可选）**")
        with st.expander("🔧 算法高级参数", expanded=False):
            algorithm_params = {
                'n_turbines': 10,
                'cost_weight': 0.5,
                'max_slope': 15,
                'max_road_distance': 1000,
                'min_residential_distance': 600,
                'min_heritage_distance': 700,
                'min_geology_distance': 800,
                'min_water_distance': 1000
            }

            if algo == "遗传算法":
                algorithm_params['pop_size'] = st.slider("种群大小", 20, 200, 50)
                algorithm_params['generations'] = st.slider("迭代代数", 50, 500, 200)
                algorithm_params['mutation_rate'] = st.slider("变异率", 0.01, 0.3, 0.1, 0.01)
                algorithm_params['crossover_rate'] = st.slider("交叉率", 0.5, 1.0, 0.8, 0.05)

            elif algo == "模拟退火算法":
                algorithm_params['initial_temp'] = st.slider("初始温度", 100, 5000, 1000, 100)
                algorithm_params['cooling_rate'] = st.slider("降温速率", 0.85, 0.99, 0.95, 0.01)
                algorithm_params['iterations_per_temp'] = st.slider("每温度迭代次数", 10, 200, 50)

            elif algo == "粒子群优化算法":
                algorithm_params['pop_size'] = st.slider("粒子数量", 20, 100, 30)
                algorithm_params['generations'] = st.slider("迭代次数", 50, 500, 100)
                algorithm_params['w'] = st.slider("惯性权重", 0.1, 1.0, 0.7, 0.1)
                algorithm_params['c1'] = st.slider("个体学习因子", 0.1, 2.0, 1.5, 0.1)
                algorithm_params['c2'] = st.slider("社会学习因子", 0.1, 2.0, 1.5, 0.1)

            elif algo == "PuLP优化求解器":
                algorithm_params['solver_type'] = st.selectbox("求解器类型", ["CBC", "GLPK", "CPLEX"])
                algorithm_params['time_limit'] = st.slider("时间限制(秒)", 10, 300, 60)

            elif algo == "两者对比":
                algorithm_params['compare_generations'] = st.slider("对比迭代次数", 50, 300, 100)

        # 文件上传
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📂 上传风速预测数据", type=["csv"])

        # 处理文件上传
        if uploaded_file is not None:
            if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                df = pd.read_csv(uploaded_file)

                # 检查必要的列是否存在
                required_columns = ['predicted_wind_speed', 'latitude', 'longitude', 'elevation', 'slope']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    st.error(f"❌ 数据文件缺少必要的列: {missing_columns}")
                    return

                if "predicted_wind_speed" in df.columns:
                    df["wind_power_density"] = 0.5 * 1.225 * (df["predicted_wind_speed"] ** 3)

                df["valid"] = (
                        (df["predicted_wind_speed"] >= 5.0) &
                        (df["slope"] <= 35) &
                        (df["elevation"] >= 150) & (df["elevation"] <= 1600)
                )

                st.session_state["windfarm_data"] = df
                st.session_state.last_uploaded_file = uploaded_file.name
                st.success("✅ 数据加载成功")
                st.session_state.current_page = "wind"
                st.rerun()
        else:
            # 清理所有相关状态
            keys_to_clear = ['last_uploaded_file', 'windfarm_data', 'optimization_result',
                             'turbines_strategy']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

        # 优化按钮
        st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
        if "windfarm_data" in st.session_state:
            df = st.session_state["windfarm_data"]
            if "predicted_wind_speed" in df.columns and df["predicted_wind_speed"].std() < 0.5:
                st.warning("⚠️ 风速数据变化较小，可能影响优化效果")

            if st.button("🚀 开始优化计算", use_container_width=True, type="primary"):
                with st.spinner("正在计算最优布局和各风机充放电策略..."):
                    try:
                        result = call_optimize_function(df, algo, algorithm_params)
                        st.session_state["optimization_result"] = result

                        # 获取选中的风机位置
                        valid_points = df[df["valid"]]
                        if 'selected_locations' in result and len(result['selected_locations']) > 0:
                            selected_locations = result['selected_locations']
                        else:
                            # 如果没有选中的位置，使用最佳点位
                            selected_locations = valid_points.nlargest(
                                min(algorithm_params['n_turbines'], len(valid_points)),
                                'predicted_wind_speed'
                            )

                        # 为每个风机制定充放电策略
                        if len(selected_locations) > 0:
                            turbines_strategy = analyze_all_turbines_strategy(selected_locations)
                            st.session_state["turbines_strategy"] = turbines_strategy
                            st.success("🎯 优化完成，各风机充放电策略分析已自动执行")
                        else:
                            st.warning("⚠️ 没有找到合适的风机位置")

                        st.session_state.current_page = "result"
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 优化计算失败: {str(e)}")
                        st.info("💡 建议：检查数据格式或尝试使用不同的参数")
        else:
            st.button("🚀 开始优化计算", use_container_width=True, disabled=True)

    # ========== 优化结果详情展示 ==========
    if st.session_state.current_page == "result" and "optimization_result" in st.session_state:
        st.markdown("---")
        st.markdown("#### 📊 优化结果分析")

        result = st.session_state["optimization_result"]
        df = st.session_state["windfarm_data"]
        display_optimization_result(result, df)

    # ========== 各风机充放电策略展示 ==========
    if "turbines_strategy" in st.session_state:
        st.markdown("---")
        display_turbines_strategy_analysis(st.session_state["turbines_strategy"])


# ======================================================
# 🚀 运行 Streamlit
# ======================================================
if __name__ == "__main__":
    strategy_optimization_page()