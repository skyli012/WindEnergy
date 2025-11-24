import pandas as pd
import numpy as np


def create_and_save_sample_data(file_path="sample_wind_storage_data.csv"):
    """
    创建增强的模拟数据并保存为CSV文件
    （已修改为每10分钟一个数据点）
    """

    np.random.seed(42)  # 保证可重现

    periods = 144  # 24小时 * 6（10分钟间隔）
    index = pd.date_range('2024-01-01 00:00', periods=periods, freq='10T')

    # 电网和储能参数
    grid_limit = 20  # MW
    battery_capacity = 30  # MWh
    max_charge_power = 8   # MW
    max_discharge_power = 8  # MW

    # 创建人为设计的风电模式
    t = np.linspace(0, 4 * np.pi, periods)

    daily_pattern = 0.6 + 0.3 * np.sin(t - np.pi / 2)
    morning_peak = 0.5 * np.exp(-((t - 1.5 * np.pi) ** 2) / 0.3)
    afternoon_peak = 0.6 * np.exp(-((t - 2.5 * np.pi) ** 2) / 0.25)
    ramp_event_1 = 0.4 * np.exp(-((t - 0.8 * np.pi) ** 2) / 0.1)
    ramp_event_2 = -0.35 * np.exp(-((t - 3.2 * np.pi) ** 2) / 0.08)
    night_fluctuation = 0.25 * np.sin(8 * t) * np.exp(-((t - 3.8 * np.pi) ** 2) / 0.5)

    random_component = 0.15 * np.sin(12 * t) + 0.1 * np.sin(20 * t)
    noise = 0.1 * np.random.normal(size=periods)

    # 组合风电出力
    wind_power = (
        25 * daily_pattern +
        18 * morning_peak +
        20 * afternoon_peak +
        15 * ramp_event_1 +
        12 * ramp_event_2 +
        10 * night_fluctuation +
        8 * random_component +
        6 * noise
    )

    wind_power = np.clip(wind_power, 3, 45)

    # 初始化数组
    grid_power = np.zeros(periods)
    battery_power = np.zeros(periods)
    soc = np.zeros(periods)
    soc[0] = 50  # 初始SOC%

    Δt = 1/6   # 小时 = 10分钟

    # 储能调度模拟
    for i in range(periods):
        current_wind = wind_power[i]
        current_soc = soc[i]

        power_diff = current_wind - grid_limit

        # ---- 削峰 ----
        if power_diff > 0:
            available_charge = min(
                (100 - current_soc) * battery_capacity / 100 / Δt,  # SOC 空间能接纳的功率
                max_charge_power,
                power_diff
            )

            battery_power[i] = -available_charge
            grid_power[i] = current_wind - available_charge

            # 再次避免超过电网限制
            if grid_power[i] > grid_limit:
                grid_power[i] = grid_limit

        # ---- 填谷 ----
        elif current_wind < 10:
            needed_power = 10 - current_wind

            available_discharge = min(
                (current_soc - 20) * battery_capacity / 100 / Δt,  # SOC可放出的功率
                max_discharge_power,
                needed_power
            )

            if available_discharge > 0.5:
                battery_power[i] = available_discharge
                grid_power[i] = current_wind + available_discharge
            else:
                battery_power[i] = 0
                grid_power[i] = current_wind

        else:
            battery_power[i] = 0
            grid_power[i] = current_wind

        # ---- SOC 更新 ----
        if i < periods - 1:
            soc_change = -battery_power[i] * Δt / battery_capacity * 100
            soc[i + 1] = max(20, min(90, current_soc + soc_change))

    # 最终限制并网功率
    grid_power = np.clip(grid_power, 0, grid_limit)

    # 创建 DataFrame
    data = pd.DataFrame({
        'timestamp': index,
        'wind_power': wind_power,
        'grid_power': grid_power,
        'battery_power': battery_power,
        'storage_soc': soc
    })

    data.to_csv(file_path, index=False)

    # ---- 统计 ----
    wind_energy = wind_power.sum() * Δt
    grid_energy = grid_power.sum() * Δt
    curtailment_energy = wind_energy - grid_energy
    curtailment_rate = curtailment_energy / wind_energy * 100

    print("增强模拟数据统计：")
    print(f"风电总能量: {wind_energy:.1f} MWh")
    print(f"并网总能量: {grid_energy:.1f} MWh")
    print(f"弃风能量: {curtailment_energy:.1f} MWh")
    print(f"弃风率: {curtailment_rate:.2f}%")
    print(f"最大风电功率: {wind_power.max():.1f} MW")
    print(f"超限时段: {np.sum(wind_power > grid_limit)}/{periods}")
    print(f"SOC范围: {soc.min():.1f}%-{soc.max():.1f}%")
    print(f"数据已保存到: {file_path}")

    return file_path


# 生成 CSV 文件
if __name__ == "__main__":
    csv_file_path = create_and_save_sample_data("wind_storage_simulation_data.csv")
    print(f"CSV 文件生成完成: {csv_file_path}")





