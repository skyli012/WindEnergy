import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def create_optimization_comparison_table(baseline_metrics, optimized_metrics):
    """创建优化前后性能指标对比表格（只显示改善的指标）"""

    # 定义指标分类和显示格式 - 只保留发电性能、风资源、经济性指标
    metric_categories = {
        '发电性能指标': [
            ('年发电量', 'GWh', 3),
            ('总装机容量', 'MW', 1),
            ('平均容量因数', '%', 1),
            ('等效满发小时', 'h', 0),
            ('风能密度', 'W/m²', 1)
        ],
        '风资源指标': [
            ('平均风速', 'm/s', 1),
            ('最大风速', 'm/s', 1),
            ('最小风速', 'm/s', 1),
            ('风速标准差', 'm/s', 2)
        ],
        '经济性指标': [
            ('平均成本', '万元', 1),
            ('总成本', '万元', 1),
            ('总投资', '亿元', 2),
            ('年收益', '百万元', 1),
            ('投资回收期', '年', 1)
        ]
    }

    # 创建对比数据
    comparison_data = []
    improved_categories = set()  # 记录有改善指标的类别

    for category, metrics in metric_categories.items():
        category_has_improvement = False
        category_metrics_data = []

        for metric_info in metrics:
            metric_name = metric_info[0]
            unit = metric_info[1]
            decimals = metric_info[2]

            if metric_name in baseline_metrics and metric_name in optimized_metrics:
                baseline_value = baseline_metrics[metric_name]
                optimized_value = optimized_metrics[metric_name]

                # 格式化数值显示
                if isinstance(baseline_value, (int, float)) and isinstance(optimized_value, (int, float)):
                    # 数值型指标
                    if decimals == 0:
                        baseline_display = f"{baseline_value:.0f}"
                        optimized_display = f"{optimized_value:.0f}"
                    else:
                        baseline_display = f"{baseline_value:.{decimals}f}"
                        optimized_display = f"{optimized_value:.{decimals}f}"

                    # 计算提升率（百分比）
                    if baseline_value != 0:
                        improvement = ((optimized_value - baseline_value) / abs(baseline_value)) * 100
                    else:
                        improvement = 0

                    # 确定状态
                    if '成本' in metric_name or '投资回收期' in metric_name:
                        # 这些指标越小越好
                        is_improved = improvement < 0
                    else:
                        # 其他指标越大越好
                        is_improved = improvement > 0

                    if is_improved:
                        status = "✅ 改善"
                        improvement_display = f"{improvement:.1f}%" if improvement < 0 else f"+{improvement:.1f}%"

                        category_metrics_data.append({
                            '指标': metric_name,
                            '单位': unit,
                            '初始方案': baseline_display,
                            '优化后': optimized_display,
                            '提升率': improvement_display,
                            '状态': status
                        })
                        category_has_improvement = True

        # 如果这个类别有改善的指标，添加到数据中
        if category_has_improvement:
            # 添加分类标题行
            comparison_data.append({
                '指标': f'**{category}**',
                '单位': '',
                '初始方案': '',
                '优化后': '',
                '提升率': '',
                '状态': ''
            })
            # 添加这个类别的改善指标
            comparison_data.extend(category_metrics_data)
            improved_categories.add(category)

    # 如果没有改善的指标，显示提示信息
    if not comparison_data:
        st.info("📊 本次优化没有明显改善的指标")
        return

    # 创建DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # 显示表格
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "指标": st.column_config.TextColumn("性能指标", width="medium"),
            "单位": st.column_config.TextColumn("单位", width="small"),
            "初始方案": st.column_config.TextColumn("初始方案", width="medium"),
            "优化后": st.column_config.TextColumn("优化后", width="medium"),
            "提升率": st.column_config.TextColumn("提升率", width="small"),
            "状态": st.column_config.TextColumn("状态", width="small")
        }
    )

    # 添加总结统计
    total_metrics = len([item for sublist in metric_categories.values() for item in sublist])
    improved_metrics = len(comparison_data) - len(improved_categories)  # 减去分类标题行
    improvement_rate = (improved_metrics / total_metrics) * 100 if total_metrics > 0 else 0

    st.markdown("---")
    st.subheader("🎯 优化效果总结")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总指标数量", total_metrics)
    with col2:
        st.metric("改善指标数量", improved_metrics)
    with col3:
        st.metric("改善类别数量", len(improved_categories))
    with col4:
        st.metric("整体改善率", f"{improvement_rate:.1f}%")

    # 显示改善的类别
    if improved_categories:
        st.info(f"**改善的指标类别**: {', '.join(improved_categories)}")

    # 下载按钮
    if comparison_data:
        csv = comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载优化改善数据 (CSV)",
            data=csv,
            file_name=f"优化改善指标_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


def create_convergence_chart(fitness_history):
    """创建算法收敛过程图表"""
    if fitness_history:
        fitness_smooth = pd.Series(fitness_history).rolling(5, min_periods=1).mean()
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            y=fitness_history,
            mode="lines",
            name="原始适应度",
            line=dict(color='lightblue', width=1)
        ))
        fig_conv.add_trace(go.Scatter(
            y=fitness_smooth,
            mode="lines",
            name="平滑趋势",
            line=dict(color="crimson", width=3)
        ))
        fig_conv.update_layout(
            height=400,
            template="plotly_white",
            title="算法收敛曲线",
            xaxis_title="迭代次数",
            yaxis_title="适应度值"
        )
        st.plotly_chart(fig_conv, use_container_width=True, key="convergence_chart")
    else:
        st.info("📊 未找到收敛历史数据")


def create_wind_farm_tables(wind_farm_fengjie, n_farms, n_turbines_per_farm):
    """创建风场详细数据表格"""
    # 将风机数据分组到不同的风场
    farm_data_list = []

    for i in range(n_farms):
        start_idx = i * n_turbines_per_farm
        end_idx = start_idx + n_turbines_per_farm

        # 确保索引不超出范围
        if start_idx < len(wind_farm_fengjie):
            farm_turbines = wind_farm_fengjie.iloc[start_idx:end_idx].copy()

            # 计算风场的中心坐标（平均位置）
            center_lat = farm_turbines['lat'].mean() if 'lat' in farm_turbines.columns else 'N/A'
            center_lon = farm_turbines['lon'].mean() if 'lon' in farm_turbines.columns else 'N/A'

            # 计算风场的边界坐标
            min_lat = farm_turbines['lat'].min() if 'lat' in farm_turbines.columns else 'N/A'
            max_lat = farm_turbines['lat'].max() if 'lat' in farm_turbines.columns else 'N/A'
            min_lon = farm_turbines['lon'].min() if 'lon' in farm_turbines.columns else 'N/A'
            max_lon = farm_turbines['lon'].max() if 'lon' in farm_turbines.columns else 'N/A'

            # 计算风场的各项统计数据
            farm_stats = {
                '风场编号': f'风场{i + 1}',
                '风机数量': len(farm_turbines),
                '中心纬度': center_lat,
                '中心经度': center_lon,
                '纬度范围': f"{min_lat:.4f}~{max_lat:.4f}" if min_lat != 'N/A' and max_lat != 'N/A' else 'N/A',
                '经度范围': f"{min_lon:.4f}~{max_lon:.4f}" if min_lon != 'N/A' and max_lon != 'N/A' else 'N/A',
                '平均海拔(m)': farm_turbines['elevation'].mean() if 'elevation' in farm_turbines.columns else 'N/A',
                '平均坡度(°)': farm_turbines['slope'].mean() if 'slope' in farm_turbines.columns else 'N/A',
                '最大坡度(°)': farm_turbines['slope'].max() if 'slope' in farm_turbines.columns else 'N/A',
                '最小坡度(°)': farm_turbines['slope'].min() if 'slope' in farm_turbines.columns else 'N/A',
                '到道路平均距离(m)': farm_turbines[
                    'road_distance'].mean() if 'road_distance' in farm_turbines.columns else 'N/A',
                '到居民区平均距离(m)': farm_turbines[
                    'residential_distance'].mean() if 'residential_distance' in farm_turbines.columns else 'N/A',
                '到水体平均距离(m)': farm_turbines[
                    'water_distance'].mean() if 'water_distance' in farm_turbines.columns else 'N/A',
                '平均风速(m/s)': farm_turbines[
                    'predicted_wind_speed'].mean() if 'predicted_wind_speed' in farm_turbines.columns else 'N/A',
                '平均成本': farm_turbines['cost'].mean() if 'cost' in farm_turbines.columns else 'N/A'
            }

            # 格式化数值
            for key, value in farm_stats.items():
                if isinstance(value, (int, float)) and key != '风机数量':
                    if '中心纬度' in key or '中心经度' in key:
                        farm_stats[key] = f"{value:.4f}"
                    elif '距离' in key:
                        farm_stats[key] = f"{value:.0f}"
                    elif '海拔' in key:
                        farm_stats[key] = f"{value:.0f}"
                    elif '坡度' in key or '成本' in key:
                        farm_stats[key] = f"{value:.1f}"
                    elif '风速' in key:
                        farm_stats[key] = f"{value:.2f}"

            farm_data_list.append(farm_stats)

    # 创建DataFrame并显示表格
    if farm_data_list:
        farm_df = pd.DataFrame(farm_data_list)

        # 设置索引为风场编号
        farm_df.set_index('风场编号', inplace=True)

        # 重新排列列的顺序，让坐标信息在前面
        column_order = [
            '风机数量', '中心纬度', '中心经度', '纬度范围', '经度范围',
            '平均海拔(m)', '平均坡度(°)', '最大坡度(°)', '最小坡度(°)',
            '到道路平均距离(m)', '到居民区平均距离(m)', '到水体平均距离(m)',
            '平均风速(m/s)', '平均成本'
        ]

        # 只保留实际存在的列
        available_columns = [col for col in column_order if col in farm_df.columns]
        farm_df = farm_df[available_columns]

        # 显示表格
        st.dataframe(farm_df, use_container_width=True)

        # 可选：下载数据按钮
        csv = farm_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 下载风场数据表格 (CSV)",
            data=csv,
            file_name=f"风场详细数据_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("📊 暂无风场数据可显示")


def create_wind_resource_tables(wind_farm_fengjie, n_farms, n_turbines_per_farm):
    """创建风能资源性能表格"""
    # 计算每个风场的风能资源性能
    wind_resource_data_list = []

    for i in range(n_farms):
        start_idx = i * n_turbines_per_farm
        end_idx = start_idx + n_turbines_per_farm

        # 确保索引不超出范围
        if start_idx < len(wind_farm_fengjie):
            farm_turbines = wind_farm_fengjie.iloc[start_idx:end_idx].copy()

            # 计算风能资源性能指标
            if 'predicted_wind_speed' in farm_turbines.columns:
                # 平均风速
                avg_wind_speed = farm_turbines['predicted_wind_speed'].mean()

                # 风能密度 (W/m²) - 使用标准风能密度公式: P = 0.5 * ρ * v³
                air_density = 1.225  # kg/m³ (标准空气密度)
                wind_power_density = 0.5 * air_density * (avg_wind_speed ** 3)

                # 年利用小时数估算 - 基于风速分布和风机功率曲线
                # 假设风速在3-25m/s范围内有效运行
                effective_hours = 8760 * 0.85  # 假设85%的时间在有效风速范围内

                # 单台风机年发电量估算 (kWh/年)
                # 使用标准风机参数
                TURBINE_RATED_POWER = 2500  # kW
                CAPACITY_FACTOR = 0.25  # 典型容量因数25%
                annual_energy_per_turbine = TURBINE_RATED_POWER * 8760 * CAPACITY_FACTOR

                # 风场总年发电量估算 (kWh/年)
                total_annual_energy = annual_energy_per_turbine * len(farm_turbines)

                # 更精确的容量因数估算（基于风速）
                if avg_wind_speed <= 3.0:
                    capacity_factor_estimated = 0.05
                elif avg_wind_speed <= 5.0:
                    capacity_factor_estimated = 0.15
                elif avg_wind_speed <= 7.0:
                    capacity_factor_estimated = 0.25
                elif avg_wind_speed <= 9.0:
                    capacity_factor_estimated = 0.35
                elif avg_wind_speed <= 11.0:
                    capacity_factor_estimated = 0.45
                else:
                    capacity_factor_estimated = 0.50

                # 使用估算的容量因数重新计算发电量
                annual_energy_per_turbine_estimated = TURBINE_RATED_POWER * 8760 * capacity_factor_estimated
                total_annual_energy_estimated = annual_energy_per_turbine_estimated * len(farm_turbines)

                wind_resource_stats = {
                    '风场编号': f'风场{i + 1}',
                    '风机数量': len(farm_turbines),
                    '平均风速(m/s)': avg_wind_speed,
                    '风能密度(W/m²)': wind_power_density,
                    '估算容量因数(%)': capacity_factor_estimated * 100,
                    '年利用小时数(h)': effective_hours,
                    '单机年发电量(kWh)': annual_energy_per_turbine_estimated,
                    '风场年发电量(kWh)': total_annual_energy_estimated,
                    '风场年发电量(MWh)': total_annual_energy_estimated / 1000,
                    '风场年发电量(GWh)': total_annual_energy_estimated / 1e6
                }

                # 格式化数值
                wind_resource_stats['平均风速(m/s)'] = f"{avg_wind_speed:.2f}"
                wind_resource_stats['风能密度(W/m²)'] = f"{wind_power_density:.1f}"
                wind_resource_stats['估算容量因数(%)'] = f"{capacity_factor_estimated * 100:.1f}%"
                wind_resource_stats['年利用小时数(h)'] = f"{effective_hours:.0f}"
                wind_resource_stats['单机年发电量(kWh)'] = f"{annual_energy_per_turbine_estimated:,.0f}"
                wind_resource_stats['风场年发电量(kWh)'] = f"{total_annual_energy_estimated:,.0f}"
                wind_resource_stats['风场年发电量(MWh)'] = f"{total_annual_energy_estimated / 1000:,.1f}"
                wind_resource_stats['风场年发电量(GWh)'] = f"{total_annual_energy_estimated / 1e6:.3f}"

            else:
                # 如果没有风速数据，显示N/A
                wind_resource_stats = {
                    '风场编号': f'风场{i + 1}',
                    '风机数量': len(farm_turbines),
                    '平均风速(m/s)': 'N/A',
                    '风能密度(W/m²)': 'N/A',
                    '估算容量因数(%)': 'N/A',
                    '年利用小时数(h)': 'N/A',
                    '单机年发电量(kWh)': 'N/A',
                    '风场年发电量(kWh)': 'N/A',
                    '风场年发电量(MWh)': 'N/A',
                    '风场年发电量(GWh)': 'N/A'
                }

            wind_resource_data_list.append(wind_resource_stats)

    # 创建风能资源性能DataFrame并显示表格
    if wind_resource_data_list:
        wind_resource_df = pd.DataFrame(wind_resource_data_list)

        # 设置索引为风场编号
        wind_resource_df.set_index('风场编号', inplace=True)

        # 选择要显示的列（避免信息重复）
        display_columns = [
            '风机数量', '平均风速(m/s)', '风能密度(W/m²)', '估算容量因数(%)',
            '年利用小时数(h)', '单机年发电量(kWh)', '风场年发电量(MWh)'
        ]

        # 只显示存在的列
        available_columns = [col for col in display_columns if col in wind_resource_df.columns]
        display_df = wind_resource_df[available_columns]

        # 显示表格
        st.dataframe(display_df, use_container_width=True)

        # 下载风能资源数据按钮
        csv_wind = wind_resource_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 下载风能资源性能数据 (CSV)",
            data=csv_wind,
            file_name=f"风能资源性能_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("🌬️ 暂无风能资源性能数据可显示")