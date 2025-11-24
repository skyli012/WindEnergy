# src/visualization/storage_schedule_display.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    显示单风场储能调度分析
    """
    storage_results = result['storage_results']
    best_strategy = result.get('best_strategy', '未知')

    st.markdown("### ⚡ 储能调度详细分析")

    # 创建标签页显示不同的分析视图
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 功率平衡分析",
        "🔋 储能状态分析",
        "📈 策略效果对比",
        "🎯 调度性能指标"
    ])

    with tab1:
        display_power_balance_analysis(storage_results, best_strategy, farm_name="主风场")

    with tab2:
        display_storage_state_analysis(storage_results, best_strategy, farm_name="主风场")

    with tab3:
        display_strategy_effect_comparison(result)

    with tab4:
        display_scheduling_performance_metrics(storage_results, best_strategy, farm_name="主风场")


def display_multi_farm_storage_analysis(result, df):
    """
    显示多风场储能调度分析
    """

    # 获取风场信息
    n_farms = result.get('n_farms', 1)
    storage_results_list = result['storage_results']
    best_strategy = result.get('best_strategy', '未知')

    # 为每个风场创建标签页
    farm_tabs = st.tabs([f"🏭 风场 {i + 1}" for i in range(n_farms)])

    for i, tab in enumerate(farm_tabs):
        with tab:
            if i < len(storage_results_list):
                farm_storage_results = storage_results_list[i]
                st.markdown(f"#### 风场 {i + 1} - {best_strategy}策略")

                # 为每个风场显示完整的分析
                col1, col2 = st.columns(2)

                with col1:
                    display_power_balance_analysis(farm_storage_results, best_strategy, f"风场 {i + 1}")

                with col2:
                    display_storage_state_analysis(farm_storage_results, best_strategy, f"风场 {i + 1}")

                # 性能指标
                st.markdown("##### 性能指标")
                display_scheduling_performance_metrics(farm_storage_results, best_strategy, f"风场 {i + 1}")
            else:
                st.info(f"风场 {i + 1} 暂无储能调度数据")

    # 显示策略效果对比（所有风场）
    st.markdown("---")
    display_strategy_effect_comparison(result)

    # 显示多风场综合对比
    display_multi_farm_comparison(result)


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
    显示功率平衡分析
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
        showlegend=True
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