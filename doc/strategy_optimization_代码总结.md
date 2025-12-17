# strategy_optimization.py 代码总结

## 📋 概述

`strategy_optimization.py` 是风电场选址-储能协同决策系统的核心模块，实现了**两个层次的决策**：

1. **风场选址决策**：确定风电场的位置和布局
2. **风机运行调度决策**：优化风机的运行策略和储能调度

---

## 🎯 核心功能

### 一、风场选址决策（Site Selection Decision）

#### 1.1 多风场配置

**功能描述：**
- 支持同时规划 **1-5个风电场**
- 每个风电场可配置 **1-10台风机**
- 自动计算总风机数量：`total_turbines = n_farms × n_turbines_per_farm`

**关键代码：**
```python
n_farms = st.slider("风场数量", 1, 5, st.session_state.n_farms)
n_turbines = st.slider("单场风机数", 1, 10, st.session_state.n_turbines_per_farm)
total_turbines = n_farms * n_turbines
```

#### 1.2 选址约束条件

**地理约束：**
- **坡度约束**：`max_slope = 35°`
- **海拔约束**：`elevation ∈ [150, 1600] m`
- **边界约束**：必须在指定地理边界内

**距离约束：**
- **道路距离**：`max_road_distance = 100 m`
- **居民区距离**：`min_residential_distance = 60 m`
- **文化遗产距离**：`min_heritage_distance = 70 m`
- **地质距离**：`min_geology_distance = 80 m`
- **水体距离**：`min_water_distance = 100 m`

**间距约束：**
- **风场间距**：根据风场数量动态调整
  - 1个风场：无间距要求
  - 2个风场：≥ 3.0 km
  - 3个风场：≥ 2.5 km
  - 4个风场：≥ 2.0 km
  - 5个风场：≥ 1.5 km

- **风机间距**：
  - 主风向间距：`8D = 1120 m`（D=140m为风机直径）
  - 侧向间距：`4D = 560 m`

#### 1.3 优化目标函数

**综合适应度函数：**
\[
F(\mathbf{x}) = w_1 \cdot S_{\text{wind}}(\mathbf{x}) + w_2 \cdot S_{\text{util}}(\mathbf{x}) - w_c \cdot C_{\text{penalty}}(\mathbf{x})
\]

**参数配置：**
- **风速权重**：`wind_speed_weight = 0.6`（默认）
- **利用率权重**：`utilization_weight = 0.4`（默认）
- **成本惩罚权重**：`cost_weight = 0.5`

**评分计算：**
1. **风速得分**：归一化风速值
2. **利用率得分**：归一化风能利用率
3. **成本惩罚**：违反约束条件的惩罚项

#### 1.4 优化算法选择

**支持的算法：**
1. **遗传算法（GA）**
   - 种群大小：20-300（根据风场数量自适应）
   - 迭代代数：50-500
   - 变异率：0.01-0.3
   - 交叉率：0.5-1.0

2. **模拟退火算法（SA）**
   - 初始温度：100-5000
   - 降温速率：0.85-0.99
   - 每温度迭代次数：10-200

3. **粒子群优化算法（PSO）**
   - 粒子数量：20-150
   - 迭代次数：50-500
   - 惯性权重：0.1-1.0
   - 学习因子：c1, c2 = 0.1-2.0

4. **PuLP优化求解器（线性规划）**
   - 求解器类型：CBC / GLPK / CPLEX
   - 时间限制：10-600秒

#### 1.5 数据预处理

**`process_imported_data()` 函数功能：**

1. **计算风功率密度：**
   ```python
   wind_power_density = 0.5 * 1.225 * (predicted_wind_speed ** 3)
   ```

2. **计算风能利用率：**
   ```python
   wind_utilization_rate = calculate_wind_utilization(predicted_wind_speed)
   ```

3. **归一化处理：**
   ```python
   normalized_wind_speed = predicted_wind_speed / max_wind_speed
   normalized_utilization = wind_utilization_rate / max_utilization
   ```

4. **综合评分：**
   ```python
   composite_score = 0.6 * normalized_wind_speed + 0.4 * normalized_utilization
   ```

5. **有效点位筛选：**
   ```python
   valid = (predicted_wind_speed >= 5.0) & 
           (slope <= 35) & 
           (elevation >= 150) & (elevation <= 1600) & 
           (composite_score >= 0.4)
   ```

---

### 二、风机运行调度决策（Turbine Operation Scheduling Decision）

#### 2.1 储能策略选择

**通过 `call_optimize_function_with_all_strategies()` 实现：**

系统会**自动测试三种储能调度策略**，并选择最优策略：

1. **平滑输出策略（Smoothing Output）**
   - 目标：平滑风电功率波动
   - 平滑因子：`smoothing_factor = 0.8`
   - 峰值阈值：`peak_threshold = 0.9`

2. **削峰填谷策略（Peak Shaving & Valley Filling）**
   - 目标：在高峰时段充电，低谷时段放电
   - 平滑因子：`smoothing_factor = 0.6`
   - 峰值阈值：`peak_threshold = 0.8`
   - 低谷阈值：`valley_threshold = 0.6`

3. **混合模式策略（Hybrid Mode）**
   - 目标：结合平滑输出和削峰填谷的优点
   - 平滑因子：`smoothing_factor = 0.7`
   - 峰值阈值：`peak_threshold = 0.85`
   - 根据SOC动态调整充放电策略

#### 2.2 多风场独立调度

**关键特性：**
- 每个风场**独立生成储能调度数据**
- 支持不同风场使用不同调度策略
- 为每个风场计算独立的性能指标

**实现逻辑：**
```python
# 分割风场数据
for farm_idx in range(n_farms):
    start_idx = farm_idx * n_turbines_per_farm
    end_idx = start_idx + n_turbines_per_farm
    farm_positions = result['best_positions'][start_idx:end_idx]
    
    # 为当前风场生成储能调度数据
    farm_storage = generate_storage_schedule_data(df, farm_positions, storage_params)
    farm_storage_results.append(farm_storage)
```

#### 2.3 储能调度参数

**储能系统配置：**
- **储能容量**：`storage_capacity = 60000 kWh`（默认）
- **最大功率**：`max_power = 30000 kW`（默认）
- **电网容量**：`grid_capacity = 20000 kW`（默认）

**SOC约束：**
- **最小SOC**：`SOC_min = 0.1`（10%）
- **最大SOC**：`SOC_max = 0.9`（90%）
- **初始SOC**：`SOC_0 = 0.5`（50%）

#### 2.4 调度决策逻辑

**功率平衡方程：**
\[
P_{\text{grid}}(t) = P_{\text{wind}}(t) + P_b(t) - P_{\text{curtail}}(t)
\]

**SOC更新方程：**
\[
\text{SOC}(t+1) = \text{SOC}(t) + \frac{-P_b(t) \cdot \Delta t}{E_{\text{capacity}}}
\]

**充放电决策：**
- \(P_b(t) > 0\)：放电（向电网供电）
- \(P_b(t) < 0\)：充电（从风电吸收功率）
- \(P_b(t) = 0\)：储能系统不动作

---

## 🔄 工作流程

### 阶段1：数据准备
1. 导入风速预测数据
2. 计算风功率密度和利用率
3. 筛选有效点位
4. 地理边界过滤

### 阶段2：选址优化
1. 配置风场数量和风机数量
2. 设置优化目标权重
3. 选择优化算法
4. 运行优化计算
5. 获取最优位置方案

### 阶段3：调度优化
1. 为每个风场生成功率时间序列
2. 测试三种储能策略
3. 计算每种策略的性能指标
4. 选择最优策略
5. 生成调度方案

### 阶段4：结果分析
1. 显示优化结果地图
2. 展示发电量分析
3. 显示风能利用率分析
4. 展示储能调度分析
5. 算法对比分析（如果启用）

---

## 📊 输出结果

### 选址决策结果

**`optimization_result` 字典包含：**
- `best_positions`：最优风机位置索引列表
- `best_positions_data`：最优位置的详细数据（DataFrame）
- `best_fitness`：最优适应度值
- `fitness_history`：收敛历史
- `power_results`：发电量计算结果
- `n_farms`：风场数量
- `n_turbines_per_farm`：单场风机数

### 调度决策结果

**`storage_results` 列表包含（每个风场一个）：**
- `schedule_data`：调度时间序列数据（DataFrame）
  - `wind_power`：风电功率
  - `battery_power`：储能功率
  - `grid_power`：并网功率
  - `storage_soc`：SOC状态
  - `wind_curtailment`：弃风功率
  - `net_power`：净功率

- `performance_metrics`：性能指标
  - `smoothing_effect`：平滑效果（%）
  - `storage_utilization`：储能利用率（%）
  - `curtailment_rate`：弃风率（%）
  - `system_efficiency`：系统效率（%）

- `storage_params`：储能参数
- `strategy`：使用的策略名称

---

## 🎨 用户界面

### 页面布局
- **左侧（2/3宽度）**：地图显示区域
  - 地图视图：显示地理边界
  - 风能分布视图：显示风速分布
  - 优化结果视图：显示最优位置

- **右侧（1/3宽度）**：控制面板
  - 基础参数设置
  - 优化目标权重
  - 算法选择与参数
  - 数据状态显示
  - 优化执行按钮

### 页面状态管理
- `current_page`：当前页面状态（"map" / "wind" / "result"）
- `windfarm_data`：处理后的风场数据
- `optimization_result`：优化结果
- `algorithm_comparison_results`：算法对比结果

---

## 🔍 关键函数说明

### `strategy_optimization_page()`
**主函数**，负责整个页面的UI布局和交互逻辑。

### `process_imported_data()`
**数据预处理函数**，将原始数据转换为优化可用的格式。

### `run_algorithm_comparison()`
**多算法对比函数**，同时运行多个优化算法并对比结果。

### `display_algorithm_comparison()`
**算法对比展示函数**，生成对比表格和可视化图表。

---

## 💡 设计特点

### 1. 双层决策架构
- **第一层**：选址决策（确定在哪里建风场）
- **第二层**：调度决策（确定如何运行风机）

### 2. 多策略自动选择
- 自动测试多种储能策略
- 根据性能指标选择最优策略
- 支持不同风场使用不同策略

### 3. 多算法支持
- 提供4种优化算法选择
- 支持算法参数自定义
- 支持多算法对比分析

### 4. 约束条件全面
- 地理约束（坡度、海拔、边界）
- 距离约束（道路、居民区、文化遗产等）
- 间距约束（风场间距、风机间距）

### 5. 可视化丰富
- 地图可视化
- 性能指标对比
- 雷达图多维度分析
- 收敛曲线展示

---

## 📝 使用示例

### 基本使用流程

1. **导入数据**
   ```python
   # 在数据导入页面上传CSV文件
   # 系统自动调用 process_imported_data()
   ```

2. **配置参数**
   ```python
   n_farms = 2  # 选择2个风场
   n_turbines_per_farm = 4  # 每个风场4台风机
   wind_speed_weight = 0.6  # 风速权重
   utilization_weight = 0.4  # 利用率权重
   ```

3. **选择算法**
   ```python
   algo = "遗传算法"  # 或 "模拟退火算法" / "粒子群优化算法" / "PuLP优化求解器"
   ```

4. **执行优化**
   ```python
   result = call_optimize_function_with_all_strategies(df, algo, algorithm_params)
   # 自动完成选址和调度优化
   ```

5. **查看结果**
   - 地图上显示最优位置
   - 查看发电量分析
   - 查看储能调度分析
   - 对比不同算法性能

---

## 🔧 扩展建议

### 可能的改进方向

1. **实时调度优化**
   - 基于实时风速预测
   - 动态调整储能策略

2. **多目标优化**
   - 同时优化发电量、成本、环境影响
   - 使用Pareto前沿分析

3. **不确定性处理**
   - 考虑风速预测不确定性
   - 鲁棒优化方法

4. **协同优化**
   - 多风场协同调度
   - 考虑电网约束

---

## 📚 相关文件

- `src/optimization/algorithm_convergence_curve.py`：优化算法实现
- `src/visualization/opt_result_show.py`：结果展示
- `src/visualization/storage_schedule_display.py`：储能调度展示
- `src/utils/create_map.py`：地图可视化

---

**文档版本：** 1.0  
**最后更新：** 2024年


