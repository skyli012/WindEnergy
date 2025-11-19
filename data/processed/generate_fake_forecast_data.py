# generate_full_fengjie_wind_data.py
import pandas as pd
import numpy as np
import os


def generate_full_fengjie_wind_data(n_points=5000):
    """
    生成覆盖整个奉节县的风速预测 + GIS约束数据
    包含所有优化约束参数
    """

    # ------------------------------------------
    # 🌍 奉节县官方经纬度范围（十进制）
    # ------------------------------------------
    # 东经109°1′17″—109°45′58″，北纬30°29′19″—31°22′33″
    lat_min = 30 + 29 / 60 + 19 / 3600  # 30.4886°
    lat_max = 31 + 22 / 60 + 33 / 3600  # 31.3758°
    lon_min = 109 + 1 / 60 + 17 / 3600  # 109.0214°
    lon_max = 109 + 45 / 60 + 58 / 3600  # 109.7661°

    latitudes = np.random.uniform(lat_min, lat_max, n_points)
    longitudes = np.random.uniform(lon_min, lon_max, n_points)

    # ------------------------------------------
    # 🏔️ 模拟奉节县完整地形特征
    # ------------------------------------------
    elevation = (
        400  # 基础海拔
        + 800 * np.exp(-((latitudes - 30.9) ** 2 + (longitudes - 109.2) ** 2) / 0.04)  # 西部山区
        + 700 * np.exp(-((latitudes - 31.2) ** 2 + (longitudes - 109.7) ** 2) / 0.03)  # 东部山区
        - 200 * np.exp(-((latitudes - 30.95) ** 2 + (longitudes - 109.5) ** 2) / 0.02)  # 长江河谷
    )

    # ------------------------------------------
    # 🌬️ 模拟完整风速分布
    # ------------------------------------------
    base_wind = 5.0  # 基础风速

    # 地形影响：山地风速增强
    elevation_effect = 0.001 * (elevation - 400)

    # 主要山脉风能带
    mountain_effect = (
        1.2 * np.exp(-((latitudes - 30.9) ** 2 + (longitudes - 109.2) ** 2) / 0.03)
        + 1.0 * np.exp(-((latitudes - 31.2) ** 2 + (longitudes - 109.7) ** 2) / 0.025)
    )

    wind_speed = (
        base_wind
        + elevation_effect
        + mountain_effect
        + np.random.normal(0, 0.5, n_points)  # 随机波动
    )

    # 限制风速在合理范围内
    wind_speed = np.clip(wind_speed, 3.5, 9.5)

    # ------------------------------------------
    # 🗺️ 基于完整地理的GIS约束
    # ------------------------------------------

    # 坡度计算（度）
    slope = np.random.normal(12, 6, n_points)  # 坡度(°)
    slope = np.clip(slope, 0, 35)  # 限制在0-35度之间

    # 道路接近度（米）
    road_proximity = (
        # 主要公路
        0.7 * np.exp(-((latitudes - 30.95) ** 2 + (longitudes - 109.5) ** 2) / 0.008) * 1500  # 县城周边
        + 0.5 * np.exp(-((latitudes - 30.85) ** 2 + (longitudes - 109.55) ** 2) / 0.012) * 1500  # 长江南岸
        + 0.5 * np.exp(-((latitudes - 31.05) ** 2 + (longitudes - 109.45) ** 2) / 0.012) * 1500  # 长江北岸
        + 0.3 * np.exp(-((latitudes - 31.2) ** 2 + (longitudes - 109.6) ** 2) / 0.015) * 1500  # 东北部
        + 0.3 * np.exp(-((latitudes - 30.75) ** 2 + (longitudes - 109.25) ** 2) / 0.014) * 1500  # 西南部
        + np.random.uniform(100, 500, n_points)  # 基础道路距离
    )
    road_distance = np.clip(road_proximity, 100, 1500)

    # 居民区距离（米）
    residential_distance = (
        0.8 * np.exp(-((latitudes - 30.95) ** 2 + (longitudes - 109.5) ** 2) / 0.005) * 1500  # 县城
        + 0.6 * np.exp(-((latitudes - 31.1) ** 2 + (longitudes - 109.4) ** 2) / 0.01) * 1500  # 主要乡镇
        + 0.4 * np.exp(-((latitudes - 30.8) ** 2 + (longitudes - 109.3) ** 2) / 0.008) * 1500  # 乡村聚居点
        + np.random.uniform(300, 1200, n_points)  # 随机基础距离
    )
    residential_distance = np.clip(residential_distance, 300, 1500)

    # 文化遗产距离（米）
    heritage_distance = (
        0.9 * np.exp(-((latitudes - 30.92) ** 2 + (longitudes - 109.52) ** 2) / 0.003) * 1500  # 白帝城
        + 0.7 * np.exp(-((latitudes - 30.88) ** 2 + (longitudes - 109.48) ** 2) / 0.004) * 1500  # 瞿塘峡
        + 0.5 * np.exp(-((latitudes - 30.75) ** 2 + (longitudes - 109.25) ** 2) / 0.006) * 1500  # 天坑地缝
        + np.random.uniform(400, 1300, n_points)  # 随机基础距离
    )
    heritage_distance = np.clip(heritage_distance, 400, 1500)

    # 地质不稳定区距离（米）
    geology_distance = (
        0.8 * np.exp(-((latitudes - 30.9) ** 2 + (longitudes - 109.3) ** 2) / 0.007) * 1500  # 西部山区
        + 0.6 * np.exp(-((latitudes - 31.15) ** 2 + (longitudes - 109.65) ** 2) / 0.009) * 1500  # 东部山区
        + np.random.uniform(500, 1300, n_points)  # 随机基础距离
    )
    geology_distance = np.clip(geology_distance, 500, 1500)

    # 水源保护距离（米）
    water_distance = (
        0.9 * np.exp(-((latitudes - 30.95) ** 2 + (longitudes - 109.5) ** 2) / 0.004) * 1500  # 长江主干
        + 0.7 * np.exp(-((latitudes - 31.05) ** 2 + (longitudes - 109.42) ** 2) / 0.005) * 1500  # 梅溪河
        + 0.6 * np.exp(-((latitudes - 30.85) ** 2 + (longitudes - 109.55) ** 2) / 0.006) * 1500  # 大溪河
        + np.random.uniform(600, 1300, n_points)  # 随机基础距离
    )
    water_distance = np.clip(water_distance, 600, 1500)

    # 电网接近度（连续值，0-1之间）
    grid_proximity = (
        # 主要城镇周边
        0.6 * np.exp(-((latitudes - 30.95) ** 2 + (longitudes - 109.5) ** 2) / 0.01)  # 奉节县城
        + 0.4 * np.exp(-((latitudes - 31.15) ** 2 + (longitudes - 109.35) ** 2) / 0.015)  # 公平镇
        + 0.4 * np.exp(-((latitudes - 30.8) ** 2 + (longitudes - 109.2) ** 2) / 0.012)  # 兴隆镇
        + 0.3 * np.exp(-((latitudes - 31.25) ** 2 + (longitudes - 109.7) ** 2) / 0.018)  # 竹园镇
        # 基础电网覆盖
        + 0.2  # 基础覆盖度
    )
    grid_proximity = np.clip(grid_proximity, 0, 1)

    # ------------------------------------------
    # 💰 经济成本估算
    # ------------------------------------------
    base_cost = 900  # 万元

    cost = (
        base_cost
        + (road_distance > 800) * 60  # 远离道路
        + (1 - grid_proximity) * 100  # 远离电网（使用连续值）
        + (slope > 25) * 50  # 陡坡
        + (elevation > 1000) * 40  # 高海拔
        + (residential_distance < 800) * (-30)  # 靠近居民区成本较低（基础设施好）
        + np.random.normal(0, 40, n_points)  # 随机波动
    )

    # 限制成本在合理范围内
    cost = np.clip(cost, 800, 1200)

    # ------------------------------------------
    # 📋 构建DataFrame - 只保留连续字段
    # ------------------------------------------
    df = pd.DataFrame({
        "lat": latitudes,
        "lon": longitudes,
        "elevation": np.round(elevation, 1),
        "slope": np.round(slope, 1),
        "predicted_wind_speed": np.round(wind_speed, 2),
        "grid_proximity": np.round(grid_proximity, 3),  # 连续值替代 grid_near
        "road_distance": np.round(road_distance, 0),
        "residential_distance": np.round(residential_distance, 0),
        "heritage_distance": np.round(heritage_distance, 0),
        "geology_distance": np.round(geology_distance, 0),
        "water_distance": np.round(water_distance, 0),
        "cost": np.round(cost, 1)
    })

    # 过滤掉极端条件的点位（保持数据质量）
    df_valid = df[
        (df['slope'] < 35) &
        (df['elevation'] > 150) &
        (df['elevation'] < 1600)
    ].copy()

    # 如果有效点太少，使用宽松条件
    if len(df_valid) < 1000:
        print("⚠️ 有效点位较少，使用宽松条件...")
        df_valid = df[
            (df['slope'] < 40) &
            (df['elevation'] > 100) &
            (df['elevation'] < 1800)
        ].copy()

    # ------------------------------------------
    # 💾 保存数据
    # ------------------------------------------
    os.makedirs("data", exist_ok=True)
    output_path = "data/full_fengjie_wind_map.csv"
    df_valid.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 完整奉节县风能数据已生成：{output_path}")
    print(f"📊 数据统计：")
    print(f"   覆盖范围：{lat_min:.4f}°N-{lat_max:.4f}°N, {lon_min:.4f}°E-{lon_max:.4f}°E")
    print(f"   总点数：{len(df_valid)}")
    print(f"   平均风速：{df_valid['predicted_wind_speed'].mean():.2f} m/s")
    print(f"   平均成本：{df_valid['cost'].mean():.1f} 万元")
    print(f"   平均坡度：{df_valid['slope'].mean():.1f}°")
    print(f"   平均道路距离：{df_valid['road_distance'].mean():.0f} m")
    print(f"   平均电网接近度：{df_valid['grid_proximity'].mean():.3f}")
    print(f"   有效建设比例：{(len(df_valid) / n_points * 100):.1f}%")

    return df_valid


def analyze_coverage(df):
    """
    分析数据覆盖情况（使用官方范围）
    """
    print("\n🗺️ 数据覆盖分析：")

    # 官方范围
    lat_min, lat_max = 30.4886, 31.3758
    lon_min, lon_max = 109.0214, 109.7661

    # 按经纬度网格分析覆盖密度
    lat_bins = np.arange(lat_min, lat_max, 0.1)
    lon_bins = np.arange(lon_min, lon_max, 0.1)

    coverage_map = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))

    for i in range(len(lat_bins) - 1):
        for j in range(len(lon_bins) - 1):
            count = len(df[
                (df['lat'] >= lat_bins[i]) &
                (df['lat'] < lat_bins[i + 1]) &
                (df['lon'] >= lon_bins[j]) &
                (df['lon'] < lon_bins[j + 1])
            ])
            coverage_map[i, j] = count

    print(f"   网格覆盖统计：")
    print(f"   - 有数据的网格：{np.sum(coverage_map > 0)} / {coverage_map.size}")
    print(f"   - 平均每网格点数：{np.mean(coverage_map[coverage_map > 0]):.1f}")
    print(f"   - 最大网格密度：{np.max(coverage_map):.0f} 点")

    # 识别覆盖不足的区域
    low_coverage_areas = []
    for i in range(len(lat_bins) - 1):
        for j in range(len(lon_bins) - 1):
            if coverage_map[i, j] == 0:
                low_coverage_areas.append((
                    f"({lat_bins[i]:.1f}-{lat_bins[i + 1]:.1f}°N, "
                    f"{lon_bins[j]:.1f}-{lon_bins[j + 1]:.1f}°E)"
                ))

    if low_coverage_areas:
        print(f"   覆盖较弱区域：{', '.join(low_coverage_areas[:3])}...")
    else:
        print("   ✅ 整个奉节县区域都有良好覆盖")


if __name__ == "__main__":
    # 生成覆盖整个奉节县的数据（增加点数）
    df = generate_full_fengjie_wind_data(8000)

    # 分析覆盖情况
    analyze_coverage(df)

    print(f"\n📋 数据样例：")
    print(df.head(10))