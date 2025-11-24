import numpy as np
import pandas as pd


def compute_power_and_storage(df, storage_capacity_kwh=60000,
                              max_power_kw=30000, grid_capacity_kw=20000,
                              timestep_minutes=15,
                              output_filename="energy_storage_results.csv"):
    """
    修正版本：确保返回数据格式与显示函数兼容
    """

    # 原有的功率计算逻辑保持不变
    def wind_turbine_power(v):
        rated_power = 2500
        if v < 3:
            return 0
        elif v < 12:
            return rated_power * ((v - 3) / (12 - 3)) ** 3
        elif v < 25:
            return rated_power
        else:
            return 0

    df["wind_power"] = df["predicted_wind_speed"].apply(wind_turbine_power)

    # 储能计算逻辑
    battery_power_list = []
    soc_list = []
    soc = 0.5  # 初始 SOC = 50%
    delta_h = timestep_minutes / 60

    for _, row in df.iterrows():
        wind = row["wind_power"]

        if wind > grid_capacity_kw:
            charge = min(max_power_kw, wind - grid_capacity_kw)
            battery_power = -charge
        elif wind < grid_capacity_kw:
            discharge = min(max_power_kw, grid_capacity_kw - wind)
            battery_power = discharge
        else:
            battery_power = 0

        energy_change = -battery_power * delta_h
        soc = soc + energy_change / storage_capacity_kwh
        soc = np.clip(soc, 0, 1)

        battery_power_list.append(battery_power)
        soc_list.append(soc)

    # 关键修正：确保数据格式兼容
    df["battery_power"] = battery_power_list
    df["storage_soc"] = [s * 100 for s in soc_list]  # 转换为百分比

    # 并网功率计算
    df["grid_power"] = df["wind_power"] + df["battery_power"]
    df["grid_power"] = df["grid_power"].clip(0, grid_capacity_kw)

    # 时间戳处理 - 确保有正确的时间索引
    if 'hour' in df.columns:
        start_date = '2024-01-01'
        df["timestamp"] = pd.to_datetime(start_date) + pd.to_timedelta(df["hour"], unit='h')
    else:
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        time_deltas = pd.timedelta_range(start='0 minutes', periods=len(df), freq=f'{timestep_minutes}T')
        df["timestamp"] = start_time + time_deltas

    # 设置时间索引 - 重要！
    result_df = df.set_index('timestamp')

    # 选择需要的列
    result_columns = ["wind_power", "grid_power", "battery_power", "storage_soc"]
    available_columns = [col for col in result_columns if col in result_df.columns]
    result_df = result_df[available_columns]

    # 保存结果
    result_df.to_csv(output_filename, index=True)  # 保存索引

    print(f"✅ 储能计算完成，数据格式已适配可视化需求")
    print(f"📊 数据形状: {result_df.shape}")
    print(f"📅 时间范围: {result_df.index.min()} 到 {result_df.index.max()}")

    return result_df
