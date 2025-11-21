# import geopandas as gpd
# import streamlit as st
# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# from shapely.geometry import Point
# import plotly.graph_objects as go
# import altair as alt
#
# from src.optimization.algorithm_convergence_curve import call_optimize_function
# from src.utils.check_data import check_data_quality
# from src.visualization.opt_result_show import display_optimization_result
#
#
# # ======================================================
# # 🔋 改进的储能调度策略核心算法
# # ======================================================
#
# class EnhancedEnergyStorageScheduler:
#     """改进的储能充放电调度策略 - 适配山地风电特性"""
#
#     def __init__(self, capacity_kwh, max_power_kw, grid_capacity_kw, efficiency=0.92):
#         """
#         初始化储能系统
#         capacity_kwh: 储能容量 (kWh)
#         max_power_kw: 最大充放电功率 (kW)
#         grid_capacity_kw: 电网接收容量 (kW) - 新增关键参数
#         efficiency: 充放电效率
#         """
#         self.capacity = capacity_kwh
#         self.max_power = max_power_kw
#         self.grid_capacity = grid_capacity_kw  # 电网硬约束
#         self.efficiency = efficiency
#
#     def integrated_strategy(self, wind_power, max_ramp_rate=5):
#         """
#         综合调度策略 - 修复弃风量计算问题
#         """
#         n = len(wind_power)
#         delivered_power = np.zeros(n)  # 实际并网功率
#         curtailed_power = np.zeros(n)  # 弃风功率 - 修复这里
#         battery_soc = np.zeros(n)  # 电池SOC (0-1)
#         charge_discharge = np.zeros(n)  # 充放电功率 (+放电, -充电)
#         grid_status = np.zeros(n)  # 电网状态记录
#
#         # 初始SOC设为50%
#         soc = 0.5
#
#         for i in range(n):
#             current_wind = wind_power[i]
#
#             # ============================================
#             # 🚫 关键修复：正确处理弃风量计算
#             # ============================================
#             if current_wind > self.grid_capacity:
#                 # 情况: 风电功率超过电网容量
#                 excess_power = current_wind - self.grid_capacity
#
#                 # 计算最大可充电功率
#                 max_charge = min(
#                     self.max_power,  # 储能功率限制
#                     (0.9 - soc) * self.capacity / self.efficiency,  # SOC上限约束
#                     excess_power  # 超额功率
#                 )
#
#                 if max_charge > 0:
#                     # 执行充电 - 部分超额功率存入储能
#                     charge_power = max_charge
#                     soc += charge_power * self.efficiency / self.capacity
#                     charge_discharge[i] = -charge_power
#
#                     # 🚫 修复弃风量计算：剩余的超额功率就是弃风
#                     curtailed_power[i] = excess_power - charge_power
#                     delivered_power[i] = self.grid_capacity
#                     grid_status[i] = 1  # 标记为超额充电状态
#                 else:
#                     # 无法充电，全部超额功率都弃风
#                     curtailed_power[i] = excess_power
#                     delivered_power[i] = self.grid_capacity
#                     grid_status[i] = 2  # 标记为强制弃风状态
#
#             elif current_wind < self.grid_capacity:
#                 # ============================================
#                 # 主线2: 风电突降补偿 + 平滑输出
#                 # ============================================
#                 if i > 0:
#                     # 计算功率变化率
#                     power_ramp = current_wind - delivered_power[i - 1]
#
#                     # 如果变化率超过限制，进行平滑
#                     if abs(power_ramp) > max_ramp_rate:
#                         target_power = delivered_power[i - 1] + (
#                             max_ramp_rate if power_ramp > 0 else -max_ramp_rate
#                         )
#                     else:
#                         target_power = current_wind
#                 else:
#                     target_power = current_wind
#
#                 # 计算需要的调节功率
#                 power_diff = target_power - current_wind
#
#                 if power_diff > 0:  # 需要放电补偿
#                     max_discharge = min(
#                         self.max_power,  # 储能功率限制
#                         (soc - 0.2) * self.capacity,  # SOC下限约束(保留20%)
#                         power_diff  # 需要补偿的功率
#                     )
#
#                     if max_discharge > 0:
#                         actual_discharge = max_discharge
#                         soc -= actual_discharge / self.capacity
#                         charge_discharge[i] = actual_discharge
#                         delivered_power[i] = current_wind + actual_discharge
#                         curtailed_power[i] = 0  # 这种情况没有弃风
#                         grid_status[i] = 3  # 标记为放电补偿状态
#                     else:
#                         delivered_power[i] = current_wind
#                         curtailed_power[i] = 0
#                         grid_status[i] = 0  # 正常状态
#
#                 elif power_diff < 0:  # 需要充电平滑
#                     max_charge = min(
#                         self.max_power,
#                         (0.9 - soc) * self.capacity / self.efficiency,
#                         -power_diff
#                     )
#
#                     if max_charge > 0:
#                         actual_charge = max_charge
#                         soc += actual_charge * self.efficiency / self.capacity
#                         charge_discharge[i] = -actual_charge
#                         delivered_power[i] = current_wind - actual_charge
#                         curtailed_power[i] = 0  # 这种情况没有弃风
#                         grid_status[i] = 4  # 标记为平滑充电状态
#                     else:
#                         delivered_power[i] = current_wind
#                         curtailed_power[i] = 0
#                         grid_status[i] = 0
#
#                 else:
#                     delivered_power[i] = current_wind
#                     curtailed_power[i] = 0
#                     grid_status[i] = 0
#
#             else:
#                 # 风电功率正好等于电网容量
#                 delivered_power[i] = current_wind
#                 curtailed_power[i] = 0
#                 charge_discharge[i] = 0
#                 grid_status[i] = 0
#
#             # ============================================
#             # 主线3: SOC安全区管理 (20%-90%)
#             # ============================================
#             soc = max(0.2, min(0.9, soc))  # 严格控制在20%-90%
#             battery_soc[i] = soc
#
#         return {
#             'delivered_power': delivered_power,
#             'curtailed_power': curtailed_power,  # 🚫 现在这里会有正确的弃风数据
#             'battery_soc': battery_soc,
#             'charge_discharge': charge_discharge,
#             'grid_status': grid_status,
#             'wind_power': wind_power,
#             'grid_capacity': self.grid_capacity  # 确保返回电网容量用于显示
#         }
#
#     def smoothing_strategy(self, wind_power, max_ramp_rate=5):
#         """出力平滑策略 - 优先平滑功率波动"""
#         n = len(wind_power)
#         delivered_power = np.zeros(n)
#         curtailed_power = np.zeros(n)
#         battery_soc = np.zeros(n)
#         charge_discharge = np.zeros(n)
#         grid_status = np.zeros(n)
#
#         soc = 0.5
#
#         for i in range(n):
#             current_wind = wind_power[i]
#
#             # 平滑策略：优先考虑功率变化率限制
#             if i > 0:
#                 power_ramp = current_wind - delivered_power[i - 1]
#                 if abs(power_ramp) > max_ramp_rate:
#                     target_power = delivered_power[i - 1] + (
#                         max_ramp_rate if power_ramp > 0 else -max_ramp_rate
#                     )
#                 else:
#                     target_power = current_wind
#             else:
#                 target_power = current_wind
#
#             # 计算需要的调节功率
#             power_diff = target_power - current_wind
#
#             if power_diff > 0:  # 需要放电
#                 max_discharge = min(
#                     self.max_power,
#                     (soc - 0.2) * self.capacity,
#                     power_diff
#                 )
#                 if max_discharge > 0:
#                     soc -= max_discharge / self.capacity
#                     charge_discharge[i] = max_discharge
#                     delivered_power[i] = current_wind + max_discharge
#                 else:
#                     delivered_power[i] = current_wind
#
#             elif power_diff < 0:  # 需要充电
#                 max_charge = min(
#                     self.max_power,
#                     (0.9 - soc) * self.capacity / self.efficiency,
#                     -power_diff
#                 )
#                 if max_charge > 0:
#                     soc += max_charge * self.efficiency / self.capacity
#                     charge_discharge[i] = -max_charge
#                     delivered_power[i] = current_wind - max_charge
#                 else:
#                     delivered_power[i] = current_wind
#             else:
#                 delivered_power[i] = current_wind
#
#             # 电网约束检查
#             if delivered_power[i] > self.grid_capacity:
#                 excess = delivered_power[i] - self.grid_capacity
#                 curtailed_power[i] = excess
#                 delivered_power[i] = self.grid_capacity
#                 grid_status[i] = 2
#             else:
#                 curtailed_power[i] = 0
#
#             soc = max(0.2, min(0.9, soc))
#             battery_soc[i] = soc
#
#         return {
#             'delivered_power': delivered_power,
#             'curtailed_power': curtailed_power,
#             'battery_soc': battery_soc,
#             'charge_discharge': charge_discharge,
#             'grid_status': grid_status,
#             'wind_power': wind_power,
#             'strategy': '出力平滑'
#         }
#
#     def curtailment_reduction_strategy(self, wind_power, max_ramp_rate=5):
#         """弃风消减策略 - 最小化弃风"""
#         n = len(wind_power)
#         delivered_power = np.zeros(n)
#         curtailed_power = np.zeros(n)
#         battery_soc = np.zeros(n)
#         charge_discharge = np.zeros(n)
#         grid_status = np.zeros(n)
#
#         soc = 0.5
#
#         for i in range(n):
#             current_wind = wind_power[i]
#
#             # 弃风消减策略：优先充电减少弃风
#             if current_wind > self.grid_capacity:
#                 excess_power = current_wind - self.grid_capacity
#
#                 # 尽可能多地充电
#                 max_charge = min(
#                     self.max_power,
#                     (0.9 - soc) * self.capacity / self.efficiency,
#                     excess_power
#                 )
#
#                 if max_charge > 0:
#                     soc += max_charge * self.efficiency / self.capacity
#                     charge_discharge[i] = -max_charge
#                     curtailed_power[i] = excess_power - max_charge
#                     delivered_power[i] = self.grid_capacity
#                     grid_status[i] = 1
#                 else:
#                     curtailed_power[i] = excess_power
#                     delivered_power[i] = self.grid_capacity
#                     grid_status[i] = 2
#
#             elif current_wind < self.grid_capacity:
#                 # 正常情况，尽量保持SOC在中等水平以便后续充电
#                 delivered_power[i] = current_wind
#                 curtailed_power[i] = 0
#
#                 # 如果SOC较低且有空间，可以适当放电
#                 if soc > 0.6 and i > 0:
#                     # 适当放电以准备后续充电
#                     discharge_power = min(
#                         self.max_power * 0.3,
#                         (soc - 0.4) * self.capacity
#                     )
#                     if discharge_power > 0:
#                         soc -= discharge_power / self.capacity
#                         charge_discharge[i] = discharge_power
#                         delivered_power[i] += discharge_power
#
#             else:
#                 delivered_power[i] = current_wind
#                 curtailed_power[i] = 0
#
#             soc = max(0.2, min(0.9, soc))
#             battery_soc[i] = soc
#
#         return {
#             'delivered_power': delivered_power,
#             'curtailed_power': curtailed_power,
#             'battery_soc': battery_soc,
#             'charge_discharge': charge_discharge,
#             'grid_status': grid_status,
#             'wind_power': wind_power,
#             'strategy': '弃风消减'
#         }
#
#     def grid_priority_strategy(self, wind_power, max_ramp_rate=5):
#         """电网优先策略 - 优先保障电网稳定"""
#         n = len(wind_power)
#         delivered_power = np.zeros(n)
#         curtailed_power = np.zeros(n)
#         battery_soc = np.zeros(n)
#         charge_discharge = np.zeros(n)
#         grid_status = np.zeros(n)
#
#         soc = 0.5
#
#         for i in range(n):
#             current_wind = wind_power[i]
#
#             # 电网优先：严格限制并网功率在电网容量内
#             if current_wind > self.grid_capacity:
#                 # 立即弃风，不尝试充电（保障电网安全）
#                 curtailed_power[i] = current_wind - self.grid_capacity
#                 delivered_power[i] = self.grid_capacity
#                 grid_status[i] = 2
#
#             elif current_wind < self.grid_capacity:
#                 # 使用储能进行频率调节
#                 power_gap = self.grid_capacity - current_wind
#
#                 # 如果SOC允许，放电填补功率缺口
#                 if soc > 0.3 and power_gap > 0:
#                     discharge_power = min(
#                         self.max_power,
#                         (soc - 0.2) * self.capacity,
#                         power_gap
#                     )
#                     if discharge_power > 0:
#                         soc -= discharge_power / self.capacity
#                         charge_discharge[i] = discharge_power
#                         delivered_power[i] = current_wind + discharge_power
#                     else:
#                         delivered_power[i] = current_wind
#                 else:
#                     delivered_power[i] = current_wind
#
#                 curtailed_power[i] = 0
#
#             else:
#                 delivered_power[i] = current_wind
#                 curtailed_power[i] = 0
#
#             soc = max(0.2, min(0.9, soc))
#             battery_soc[i] = soc
#
#         return {
#             'delivered_power': delivered_power,
#             'curtailed_power': curtailed_power,
#             'battery_soc': battery_soc,
#             'charge_discharge': charge_discharge,
#             'grid_status': grid_status,
#             'wind_power': wind_power,
#             'strategy': '电网优先'
#         }
#     def calculate_optimal_storage_size(self, wind_power_analysis):
#         """
#         根据风电特性计算最优储能规模
#         基于: 风电突变幅度(150→80MW), 功率需求≥30-40MW, 容量需求≥60-80MWh
#         """
#         # 分析风电波动特性
#         max_power = np.max(wind_power_analysis)
#         min_power = np.min(wind_power_analysis)
#         power_variation = max_power - min_power
#
#         # 计算功率需求 (基于最大波动)
#         power_demand = min(power_variation * 0.3, self.grid_capacity * 0.3)  # 30%的波动幅度
#         power_demand = max(power_demand, 30000)  # 至少30MW
#
#         # 计算容量需求 (基于4小时备用)
#         capacity_demand = power_demand * 4  # 4小时放电时间
#         capacity_demand = max(capacity_demand, 60000)  # 至少60MWh
#
#         return {
#             'recommended_power_kw': power_demand,
#             'recommended_capacity_kwh': capacity_demand,
#             'max_wind_power': max_power,
#             'min_wind_power': min_power,
#             'power_variation': power_variation,
#             'analysis': f"基于风电波动{power_variation / 1000:.1f}MW, 推荐配置: {power_demand / 1000:.1f}MW/{capacity_demand / 1000:.1f}MWh"
#         }
#
#
# def calculate_wind_power_from_speed(wind_speed, turbine_capacity=2500):
#     """修复的功率曲线计算 - 确保有变化"""
#     cut_in = 3.0  # 切入风速
#     rated = 12.0  # 额定风速
#     cut_out = 25.0  # 切出风速
#
#     power = np.zeros_like(wind_speed)
#
#     for i, speed in enumerate(wind_speed):
#         if speed < cut_in:
#             power[i] = 0
#         elif speed < rated:
#             # 在切入和额定之间，功率按立方增长
#             power_ratio = ((speed - cut_in) / (rated - cut_in)) ** 3
#             power[i] = turbine_capacity * power_ratio
#         elif speed <= cut_out:
#             power[i] = turbine_capacity  # 额定功率
#         else:
#             power[i] = 0  # 切出
#
#     return power
#
#
# def calculate_enhanced_metrics(optimization_result):
#     """计算改进的性能指标"""
#     wind_power = optimization_result['wind_power']
#     delivered_power = optimization_result['delivered_power']
#     curtailed_power = optimization_result['curtailed_power']
#     grid_status = optimization_result['grid_status']
#
#     total_generation = np.sum(wind_power)
#     total_delivered = np.sum(delivered_power)
#     total_curtailed = np.sum(curtailed_power)
#
#     # 基础指标
#     curtailment_rate = total_curtailed / total_generation * 100 if total_generation > 0 else 0
#     utilization_improvement = ((total_delivered - total_generation + total_curtailed) /
#                                total_generation * 100) if total_generation > 0 else 0
#
#     # 波动性分析
#     original_fluctuation = np.std(np.diff(wind_power))
#     delivered_fluctuation = np.std(np.diff(delivered_power))
#     fluctuation_reduction = (original_fluctuation - delivered_fluctuation) / original_fluctuation * 100
#
#     # 电网约束遵守情况
#     grid_violations = np.sum(delivered_power > optimization_result.get('grid_capacity', 120000))
#     grid_compliance = (1 - grid_violations / len(delivered_power)) * 100
#
#     # SOC健康度
#     soc_values = optimization_result['battery_soc']
#     soc_health = np.mean((soc_values >= 0.2) & (soc_values <= 0.9)) * 100
#
#     return {
#         'total_generation_mwh': total_generation / 1000,
#         'total_delivered_mwh': total_delivered / 1000,
#         'total_curtailed_mwh': total_curtailed / 1000,
#         'curtailment_rate_percent': curtailment_rate,
#         'utilization_improvement_percent': utilization_improvement,
#         'fluctuation_reduction_percent': fluctuation_reduction,
#         'grid_compliance_percent': grid_compliance,
#         'soc_health_percent': soc_health,
#         'original_fluctuation': original_fluctuation,
#         'delivered_fluctuation': delivered_fluctuation,
#         'grid_violations': grid_violations
#     }
#
#
# def create_enhanced_single_turbine_assessment(optimization_result, hours):
#     """创建改进的单个风机评估图表"""
#
#     wind_power = optimization_result['wind_power']
#     delivered_power = optimization_result['delivered_power']
#     curtailed_power = optimization_result['curtailed_power']
#     battery_soc = optimization_result['battery_soc']
#     charge_discharge = optimization_result['charge_discharge']
#     grid_status = optimization_result['grid_status']
#
#     # 使用卡片式布局
#     st.markdown("### 📊 风机运行概况")
#
#     # 第一行指标 - 核心性能
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         total_gen = np.sum(wind_power) / 1000
#         st.metric(
#             "总发电量",
#             f"{total_gen:.1f} MWh",
#             help="风机总发电能力"
#         )
#     with col2:
#         total_delivered = np.sum(delivered_power) / 1000
#         st.metric(
#             "可消纳电量",
#             f"{total_delivered:.1f} MWh",
#             help="实际可并网的电量"
#         )
#     with col3:
#         curtailment_rate = (np.sum(curtailed_power) / np.sum(wind_power)) * 100 if np.sum(wind_power) > 0 else 0
#         st.metric(
#             "弃风率",
#             f"{curtailment_rate:.1f}%",
#             delta=f"-{curtailment_rate:.1f}%" if curtailment_rate > 0 else None,
#             delta_color="inverse",
#             help="因电网限制未能利用的电量比例"
#         )
#     with col4:
#         avg_soc = np.mean(battery_soc) * 100
#         st.metric(
#             "平均SOC",
#             f"{avg_soc:.1f}%",
#             help="电池平均荷电状态"
#         )
#
#     # 第二行指标 - 运行质量
#     col5, col6, col7, col8 = st.columns(4)
#     with col5:
#         utilization = (total_delivered / total_gen * 100) if total_gen > 0 else 0
#         st.metric(
#             "电能利用率",
#             f"{utilization:.1f}%",
#             help="发电量的有效利用比例"
#         )
#     with col6:
#         power_fluctuation = np.std(np.diff(wind_power))
#         st.metric(
#             "功率波动",
#             f"{power_fluctuation:.0f} kW",
#             help="功率变化的剧烈程度"
#         )
#     with col7:
#         max_charge = np.max(np.abs(charge_discharge))
#         st.metric(
#             "最大充放电",
#             f"{max_charge / 1000:.1f} MW",
#             help="储能系统最大调节能力"
#         )
#     with col8:
#         soc_range = (np.max(battery_soc) - np.min(battery_soc)) * 100
#         st.metric(
#             "SOC变化范围",
#             f"{soc_range:.1f}%",
#             help="电池SOC的波动范围"
#         )
#
#     # 使用选项卡组织图表
#     tab1, tab2, tab3 = st.tabs(["📈 功率曲线", "🔋 电池状态", "🎯 运行分析"])
#
#     with tab1:
#         # 功率曲线图 - 显示电网容量线
#         st.markdown("#### 功率曲线与电网约束")
#         power_data = pd.DataFrame({
#             '小时': hours,
#             '原始功率': wind_power / 1000,
#             '并网功率': delivered_power / 1000,
#             '弃风功率': curtailed_power / 1000,
#             '电网容量': [5] * len(hours)
#         })
#
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=power_data['小时'], y=power_data['原始功率'],
#             mode='lines', name='🌬️ 原始功率',
#             line=dict(dash='dot', color='#1f77b4', width=2)
#         ))
#         fig.add_trace(go.Scatter(
#             x=power_data['小时'], y=power_data['并网功率'],
#             mode='lines', name='🔌 并网功率',
#             line=dict(color='#2ca02c', width=3)
#         ))
#         fig.add_trace(go.Scatter(
#             x=power_data['小时'], y=power_data['弃风功率'],
#             mode='lines', name='🚫 弃风功率',
#             line=dict(color='#d62728', width=2)
#         ))
#         fig.add_trace(go.Scatter(
#             x=power_data['小时'], y=power_data['电网容量'],
#             mode='lines', name='⚡ 电网容量',
#             line=dict(dash='dash', color='#000000', width=2)
#         ))
#
#         fig.update_layout(
#             title='风电功率与电网约束分析',
#             xaxis_title='时间 (小时)',
#             yaxis_title='功率 (MW)',
#             height=400,
#             template='plotly_white',
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#         )
#         st.plotly_chart(fig, use_container_width=True)
#
#     with tab2:
#         # SOC曲线图
#         col_left, col_right = st.columns([2, 1])
#
#         with col_left:
#             st.markdown("#### 电池SOC曲线")
#             soc_data = pd.DataFrame({
#                 '小时': hours,
#                 'SOC': battery_soc * 100
#             })
#
#             soc_chart = alt.Chart(soc_data).mark_area(
#                 line={'color': '#ff7f0e', 'width': 2},
#                 color=alt.Gradient(
#                     gradient='linear',
#                     stops=[alt.GradientStop(color='white', offset=0),
#                            alt.GradientStop(color='#ff7f0e', offset=1)],
#                     x1=0, x2=0, y1=1, y2=0
#                 )
#             ).encode(
#                 x=alt.X('小时:Q', title='时间 (小时)'),
#                 y=alt.Y('SOC:Q', title='SOC (%)', scale=alt.Scale(domain=[0, 100]))
#             ).properties(height=300)
#
#             # 添加安全区间参考线
#             safe_zone = alt.Chart(pd.DataFrame({'y': [20, 90]})).mark_rule(
#                 strokeDash=[5, 5], color='red', strokeWidth=2
#             ).encode(y='y:Q')
#
#             st.altair_chart(soc_chart + safe_zone, use_container_width=True)
#
#         with col_right:
#             st.markdown("#### SOC统计")
#             soc_stats = {
#                 '平均SOC': f"{np.mean(battery_soc) * 100:.1f}%",
#                 '最大SOC': f"{np.max(battery_soc) * 100:.1f}%",
#                 '最小SOC': f"{np.min(battery_soc) * 100:.1f}%",
#                 '安全运行率': f"{np.mean((battery_soc >= 0.2) & (battery_soc <= 0.9)) * 100:.1f}%"
#             }
#
#             for key, value in soc_stats.items():
#                 st.metric(key, value)
#
#     with tab3:
#         # 运行状态分析
#         col1, col2 = st.columns(2)
#
#         with col1:
#             st.markdown("#### 运行状态分布")
#             status_counts = pd.Series(grid_status).value_counts().sort_index()
#             status_labels = {
#                 0: '✅ 正常运行',
#                 1: '🔋 超额充电',
#                 2: '🚫 强制弃风',
#                 3: '⚡ 放电补偿',
#                 4: '🔄 平滑充电'
#             }
#
#             status_data = pd.DataFrame({
#                 '状态': [status_labels.get(i, f'状态{i}') for i in status_counts.index],
#                 '次数': status_counts.values,
#                 '占比': (status_counts.values / len(grid_status) * 100).round(1)
#             })
#
#             status_chart = alt.Chart(status_data).mark_arc(innerRadius=50).encode(
#                 theta='次数:Q',
#                 color=alt.Color('状态:N', scale=alt.Scale(
#                     domain=list(status_labels.values()),
#                     range=['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e', '#9467bd']
#                 )),
#                 tooltip=['状态', '次数', '占比']
#             ).properties(height=300, title="运行状态分布")
#
#             st.altair_chart(status_chart, use_container_width=True)
#
#         with col2:
#             st.markdown("#### 充放电分析")
#             charge_data = pd.DataFrame({
#                 '类型': ['充电总量', '放电总量', '净调节量'],
#                 '数值': [
#                     np.sum(np.abs(charge_discharge[charge_discharge < 0])) / 1000,
#                     np.sum(charge_discharge[charge_discharge > 0]) / 1000,
#                     np.sum(charge_discharge) / 1000
#                 ]
#             })
#
#             charge_chart = alt.Chart(charge_data).mark_bar().encode(
#                 x='类型:N',
#                 y='数值:Q',
#                 color=alt.Color('类型:N', scale=alt.Scale(
#                     domain=['充电总量', '放电总量', '净调节量'],
#                     range=['#1f77b4', '#ff7f0e', '#2ca02c']
#                 )),
#                 tooltip=['类型', '数值']
#             ).properties(height=300, title="充放电能量统计 (MWh)")
#
#             st.altair_chart(charge_chart, use_container_width=True)
#
#
# def create_enhanced_wind_farm_assessment(metrics, storage_capacity, max_power, n_turbines, storage_recommendation):
#     """创建改进的整体风场评估"""
#
#     st.markdown("## 🏭 整体风场评估")
#
#     # 使用选项卡组织内容
#     tab1, tab2, tab3, tab4 = st.tabs(["📈 性能指标", "🔋 储能配置", "📊 改善对比", "💡 优化建议"])
#
#     with tab1:
#         # 性能指标展示
#         st.markdown("### 关键性能指标")
#
#         # 第一行 - 核心指标
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric(
#                 "电能利用率提升",
#                 f"{metrics['utilization_improvement_percent']:.1f}%",
#                 delta=f"+{metrics['utilization_improvement_percent']:.1f}%",
#                 help="储能系统带来的电能利用率提升"
#             )
#         with col2:
#             st.metric(
#                 "电网约束遵守率",
#                 f"{metrics['grid_compliance_percent']:.1f}%",
#                 help="并网功率符合电网限制的比例"
#             )
#         with col3:
#             st.metric(
#                 "功率波动降低",
#                 f"{metrics['fluctuation_reduction_percent']:.1f}%",
#                 delta=f"+{metrics['fluctuation_reduction_percent']:.1f}%",
#                 help="储能系统平滑功率波动的效果"
#             )
#         with col4:
#             st.metric(
#                 "SOC健康度",
#                 f"{metrics['soc_health_percent']:.1f}%",
#                 help="电池在安全区间内运行的时间比例"
#             )
#
#         # 第二行 - 发电指标
#         col5, col6, col7, col8 = st.columns(4)
#         with col5:
#             st.metric(
#                 "总发电量",
#                 f"{metrics['total_generation_mwh']:.1f} MWh",
#                 help="风电场总发电量"
#             )
#         with col6:
#             st.metric(
#                 "可消纳电量",
#                 f"{metrics['total_delivered_mwh']:.1f} MWh",
#                 help="实际并网电量"
#             )
#         with col7:
#             st.metric(
#                 "弃风电量",
#                 f"{metrics['total_curtailed_mwh']:.1f} MWh",
#                 delta=f"-{metrics['curtailment_rate_percent']:.1f}%",
#                 delta_color="inverse",
#                 help="因电网限制损失的电量"
#             )
#         with col8:
#             st.metric(
#                 "电网违规次数",
#                 f"{metrics['grid_violations']}",
#                 help="超过电网容量的次数"
#             )
#
#     with tab2:
#         # 储能配置分析
#         st.markdown("### 储能配置分析")
#
#         col1, col2 = st.columns(2)
#
#         with col1:
#             st.markdown("#### 当前配置")
#             # 使用卡片形式展示当前配置
#             st.info("""
#             **🔧 系统配置详情**
#             - **储能容量**: {:.1f} MWh
#             - **最大功率**: {:.1f} MW
#             - **风机数量**: {} 台
#             - **电网容量**: 120 MW
#             """.format(
#                 storage_capacity / 1000,
#                 max_power / 1000,
#                 n_turbines
#             ))
#
#             # 配置合理性评估
#             capacity_utilization = metrics['total_curtailed_mwh'] / (storage_capacity / 1000) * 100
#             st.metric(
#                 "容量利用率",
#                 f"{capacity_utilization:.1f}%",
#                 help="储能容量对弃风电量的消纳比例"
#             )
#
#         with col2:
#             st.markdown("#### 推荐配置")
#             # 使用成功样式展示推荐配置
#             st.success("""
#             **🎯 智能推荐配置**
#             - **建议功率**: {:.1f} MW
#             - **建议容量**: {:.1f} MWh
#             - **分析依据**: {}
#             """.format(
#                 storage_recommendation['recommended_power_kw'] / 1000,
#                 storage_recommendation['recommended_capacity_kwh'] / 1000,
#                 storage_recommendation['analysis']
#             ))
#
#             # 配置对比
#             power_ratio = (storage_recommendation['recommended_power_kw'] / 1000) / (max_power / 1000)
#             st.metric(
#                 "功率配置比",
#                 f"{power_ratio:.1f}",
#                 help="推荐功率与当前功率的比值"
#             )
#
#     with tab3:
#         # 性能改善对比
#         st.markdown("### 性能改善对比分析")
#
#         col1, col2 = st.columns(2)
#
#         with col1:
#             # 改善对比雷达图数据
#             categories = ['电能利用率', '电网遵守率', '波动抑制', 'SOC健康度']
#
#             before_values = [60, 70, 40, 50]  # 假设的改善前值
#             after_values = [
#                 min(100, 60 + metrics['utilization_improvement_percent']),
#                 metrics['grid_compliance_percent'],
#                 min(100, 40 + metrics['fluctuation_reduction_percent']),
#                 metrics['soc_health_percent']
#             ]
#
#             fig = go.Figure()
#
#             fig.add_trace(go.Scatterpolar(
#                 r=before_values,
#                 theta=categories,
#                 fill='toself',
#                 name='改善前',
#                 line=dict(color='red'),
#                 opacity=0.5
#             ))
#
#             fig.add_trace(go.Scatterpolar(
#                 r=after_values,
#                 theta=categories,
#                 fill='toself',
#                 name='改善后',
#                 line=dict(color='green'),
#                 opacity=0.5
#             ))
#
#             fig.update_layout(
#                 polar=dict(
#                     radialaxis=dict(
#                         visible=True,
#                         range=[0, 100]
#                     )),
#                 showlegend=True,
#                 title="性能改善雷达图",
#                 height=400
#             )
#
#             st.plotly_chart(fig, use_container_width=True)
#
#         with col2:
#             # 关键改善指标
#             st.markdown("#### 关键改善指标")
#
#             improvements = [
#                 ("弃风率降低", f"{max(0, 15 - metrics['curtailment_rate_percent']):.1f}%"),
#                 ("波动抑制", f"{metrics['fluctuation_reduction_percent']:.1f}%"),
#                 ("电网稳定性", f"{metrics['grid_compliance_percent'] - 70:.1f}%"),
#                 ("电池健康度", f"{metrics['soc_health_percent'] - 50:.1f}%")
#             ]
#
#             for name, value in improvements:
#                 st.metric(name, value)
#
#     with tab4:
#         # 优化建议
#         st.markdown("### 优化建议")
#
#         col1, col2 = st.columns(2)
#
#         with col1:
#             st.markdown("#### 🎯 立即优化建议")
#             st.info("""
#             **1. 储能功率调整**
#             - 当前: {:.1f} MW
#             - 建议: {:.1f} MW
#             - 效果: 更好的波动抑制
#
#             **2. 运行策略优化**
#             - 加强平滑控制
#             - 优化SOC管理
#             - 提高响应速度
#             """.format(max_power / 1000, storage_recommendation['recommended_power_kw'] / 1000))
#
#         with col2:
#             st.markdown("#### 📈 长期发展建议")
#             st.success("""
#             **1. 容量扩展规划**
#             - 当前: {:.1f} MWh
#             - 建议: {:.1f} MWh
#             - 收益: 减少弃风{}
#
#             **2. 智能调度升级**
#             - 引入AI预测
#             - 实时优化控制
#             - 多目标协调
#             """.format(
#                 storage_capacity / 1000,
#                 storage_recommendation['recommended_capacity_kwh'] / 1000,
#                 f"{metrics['curtailment_rate_percent']:.1f}% → <5%"
#             ))
#
#         # 总体评估
#         st.markdown("#### 📊 总体评估")
#         overall_score = (
#                                 metrics['utilization_improvement_percent'] +
#                                 metrics['grid_compliance_percent'] +
#                                 metrics['fluctuation_reduction_percent'] +
#                                 metrics['soc_health_percent']
#                         ) / 4
#
#         st.metric(
#             "综合性能评分",
#             f"{overall_score:.1f}",
#             delta=f"+{(overall_score - 50):.1f}" if overall_score > 50 else f"{(overall_score - 50):.1f}",
#             delta_color="normal" if overall_score > 60 else "off"
#         )