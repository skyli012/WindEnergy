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