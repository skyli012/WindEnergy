import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
import json


def create_comprehensive_dataset():
    """
    创建综合数据集 - 包含地理空间数据和储能系统模拟
    """
    np.random.seed(42)  # 保证结果可重现

    # -----------------------------
    # 1️⃣ 读取 GeoJSON 并提取 Polygon
    # -----------------------------
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "coordinates": [
                        [
                            [
                                35.34055168964275,
                                32.51848247177543
                            ],
                            [
                                35.3413041716567,
                                32.51885289140617
                            ],
                            [
                                35.34223550740023,
                                32.51914457969605
                            ],
                            [
                                35.343977580190256,
                                32.519271416826754
                            ],
                            [
                                35.34768240559822,
                                32.51857209667003
                            ],
                            [
                                35.34981081027138,
                                32.518105531420176
                            ],
                            [
                                35.35175637171179,
                                32.51865536622297
                            ],
                            [
                                35.356562023616505,
                                32.518737265297375
                            ],
                            [
                                35.359978749578175,
                                32.51772467485671
                            ],
                            [
                                35.36514433789122,
                                32.516952665423716
                            ],
                            [
                                35.37030904661711,
                                32.51516511669995
                            ],
                            [
                                35.37337974791859,
                                32.513210256436
                            ],
                            [
                                35.38066671240654,
                                32.50939914687423
                            ],
                            [
                                35.382876749338436,
                                32.50683807311724
                            ],
                            [
                                35.38563717496618,
                                32.50528999030571
                            ],
                            [
                                35.38950514620524,
                                32.50529177372839
                            ],
                            [
                                35.39644690828629,
                                32.5027897640032
                            ],
                            [
                                35.400408782470095,
                                32.50216814868216
                            ],
                            [
                                35.40280369122911,
                                32.50154796210461
                            ],
                            [
                                35.404459751525536,
                                32.49922106890111
                            ],
                            [
                                35.40418048591965,
                                32.495891707110076
                            ],
                            [
                                35.404179649110546,
                                32.4940328587449
                            ],
                            [
                                35.40592954625893,
                                32.489069006404534
                            ],
                            [
                                35.407028335633754,
                                32.48495589967321
                            ],
                            [
                                35.40684476286219,
                                32.4835596169288
                            ],
                            [
                                35.40795000791968,
                                32.48038106237301
                            ],
                            [
                                35.41043220674251,
                                32.47580133706745
                            ],
                            [
                                35.412547102226426,
                                32.472997566402455
                            ],
                            [
                                35.41438848156423,
                                32.470031602964454
                            ],
                            [
                                35.41972039243785,
                                32.47068381097111
                            ],
                            [
                                35.42504948706838,
                                32.47131157620973
                            ],
                            [
                                35.43202247127664,
                                32.472399614149325
                            ],
                            [
                                35.43594480346471,
                                32.47415869729733
                            ],
                            [
                                35.43961980844307,
                                32.480063004525576
                            ],
                            [
                                35.44082821526138,
                                32.48403122102623
                            ],
                            [
                                35.441420206531774,
                                32.488346422332384
                            ],
                            [
                                35.441419673864914,
                                32.49345461913073
                            ],
                            [
                                35.441325788394096,
                                32.49628347425609
                            ],
                            [
                                35.438594357159445,
                                32.50020067962775
                            ],
                            [
                                35.435883330527616,
                                32.50545067181959
                            ],
                            [
                                35.43390822610576,
                                32.50981432519802
                            ],
                            [
                                35.43111147684027,
                                32.51299591967377
                            ],
                            [
                                35.42476759328454,
                                32.51611093116341
                            ],
                            [
                                35.417934440187565,
                                32.52081619365708
                            ],
                            [
                                35.41287488901645,
                                32.52374424338055
                            ],
                            [
                                35.40705892759863,
                                32.52571589921516
                            ],
                            [
                                35.401700193249155,
                                32.52635202526562
                            ],
                            [
                                35.395732853856316,
                                32.52830018321896
                            ],
                            [
                                35.39029406031872,
                                32.53014534605437
                            ],
                            [
                                35.386742329762455,
                                32.53084401305112
                            ],
                            [
                                35.38455355045366,
                                32.53224277781629
                            ],
                            [
                                35.38387699846476,
                                32.53497935275304
                            ],
                            [
                                35.38120995420479,
                                32.53952475407482
                            ],
                            [
                                35.37918220745857,
                                32.54138503954434
                            ],
                            [
                                35.374419382378846,
                                32.542682215671235
                            ],
                            [
                                35.37178150937058,
                                32.54428820908815
                            ],
                            [
                                35.370242750117114,
                                32.54530738231894
                            ],
                            [
                                35.36753160285946,
                                32.54586329011332
                            ],
                            [
                                35.36566311129002,
                                32.545832411649485
                            ],
                            [
                                35.36467390891258,
                                32.54521473589702
                            ],
                            [
                                35.364490723287275,
                                32.54465882408623
                            ],
                            [
                                35.364930368788634,
                                32.54397937164339
                            ],
                            [
                                35.36504028016324,
                                32.54252779692561
                            ],
                            [
                                35.36482045741258,
                                32.54119973992489
                            ],
                            [
                                35.36357480105937,
                                32.53956280651013
                            ],
                            [
                                35.3626955100566,
                                32.538111160393186
                            ],
                            [
                                35.36210931605521,
                                32.53764786414018
                            ],
                            [
                                35.360424008300384,
                                32.536937471909525
                            ],
                            [
                                35.358811974796225,
                                32.53625796103965
                            ],
                            [
                                35.35844560354556,
                                32.53564021942367
                            ],
                            [
                                35.3580792343312,
                                32.53400318333955
                            ],
                            [
                                35.35760295170445,
                                32.53174834928819
                            ],
                            [
                                35.35654048112522,
                                32.530080353571165
                            ],
                            [
                                35.353975882368985,
                                32.52804164990833
                            ],
                            [
                                35.350971638110025,
                                32.5265280369608
                            ],
                            [
                                35.34822385372843,
                                32.525415978545865
                            ],
                            [
                                35.34381196824691,
                                32.52508339251776
                            ],
                            [
                                35.343043217909496,
                                32.525057650166076
                            ],
                            [
                                35.34220472310918,
                                32.52481525590551
                            ],
                            [
                                35.341737561752154,
                                32.5244718629705
                            ],
                            [
                                35.3414979918245,
                                32.524290066179816
                            ],
                            [
                                35.34059960459871,
                                32.52413856857274
                            ],
                            [
                                35.339737152862256,
                                32.52411836887319
                            ],
                            [
                                35.338886680698266,
                                32.52418906767636
                            ],
                            [
                                35.337808616027075,
                                32.524209267360774
                            ],
                            [
                                35.337053970757495,
                                32.524148668293805
                            ],
                            [
                                35.33629932528703,
                                32.523987070670856
                            ],
                            [
                                35.336119647841514,
                                32.52379517301226
                            ],
                            [
                                35.335999861120456,
                                32.52331037816103
                            ],
                            [
                                35.335939968638314,
                                32.52293667952753
                            ],
                            [
                                35.33585611916419,
                                32.522744779625626
                            ],
                            [
                                35.33556863525175,
                                32.522381178687056
                            ],
                            [
                                35.33536500081337,
                                32.52196707582577
                            ],
                            [
                                35.335376979309814,
                                32.52155297105695
                            ],
                            [
                                35.33549676427319,
                                32.521219665832405
                            ],
                            [
                                35.33602381811244,
                                32.52075505648685
                            ],
                            [
                                35.33641910849272,
                                32.520219742741205
                            ],
                            [
                                35.336922205071545,
                                32.519623823207496
                            ],
                            [
                                35.33754508688128,
                                32.51916930601945
                            ],
                            [
                                35.33816796869192,
                                32.51883599195402
                            ],
                            [
                                35.338886678472164,
                                32.51865418376116
                            ],
                            [
                                35.34055168964275,
                                32.51848247177543
                            ]
                        ]
                    ],
                    "type": "Polygon"
                }
            }
        ]
    }

    polygon = shape(geojson_data['features'][0]['geometry'])

    # -----------------------------
    # 2️⃣ 生成储能系统模拟数据
    # -----------------------------
    print("正在生成储能系统模拟数据...")
    storage_data = create_enhanced_sample_data()

    # -----------------------------
    # 3️⃣ 在 Polygon 内生成地理空间数据
    # -----------------------------
    print("正在生成地理空间数据...")
    n_points = 500
    rows = []

    for point_id in range(n_points):
        # 生成点
        while True:
            lon = np.random.uniform(polygon.bounds[0], polygon.bounds[2])
            lat = np.random.uniform(polygon.bounds[1], polygon.bounds[3])
            pt = Point(lon, lat)
            if polygon.contains(pt):
                break

        # 固定点的地理属性
        elevation = round(np.random.uniform(200, 500), 1)
        slope = round(np.random.uniform(0, 20), 1)
        grid_proximity = round(np.random.uniform(0.1, 1), 2)
        road_distance = round(np.random.uniform(50, 800), 1)
        residential_distance = round(np.random.uniform(100, 1200), 1)
        heritage_distance = round(np.random.uniform(100, 1500), 1)
        geology_distance = round(np.random.uniform(100, 1500), 1)
        water_distance = round(np.random.uniform(50, 1000), 1)
        cost = round(np.random.uniform(800, 1500), 1)

        # 为每个点生成24小时风速数据
        for hour in range(24):
            # 生成风速波动
            base = np.random.normal(7.5, 1.5)
            daily_variation = 2 * np.sin(2 * np.pi * hour / 24)
            wind_speed = base + daily_variation
            wind_speed = np.clip(wind_speed, 4, 12)

            # 计算风功率密度
            air_density = 1.225
            wind_power_density = 0.5 * air_density * (wind_speed ** 3)

            # 计算风能利用率
            operational_hours = 1.0 if 3 <= wind_speed <= 25 else 0.5
            wind_std = 0.5
            stability = 1 / (1 + wind_std)
            high_wind_hours = 1.0 if wind_speed >= 7 else 0.3
            utilization_rate = 0.5 * operational_hours + 0.3 * stability + 0.2 * high_wind_hours

            rows.append({
                "point_id": point_id,
                "hour": hour,
                "lat": lat,
                "lon": lon,
                "elevation": elevation,
                "slope": slope,
                "predicted_wind_speed": round(wind_speed, 2),
                "wind_power_density": round(wind_power_density, 2),
                "wind_utilization_rate": round(utilization_rate, 3),
                "grid_proximity": grid_proximity,
                "road_distance": road_distance,
                "residential_distance": residential_distance,
                "heritage_distance": heritage_distance,
                "geology_distance": geology_distance,
                "water_distance": water_distance,
                "cost": cost
            })

    # -----------------------------
    # 4️⃣ 创建综合 DataFrame
    # -----------------------------
    geospatial_df = pd.DataFrame(rows)

    # 计算综合评分
    max_wind_speed = geospatial_df["predicted_wind_speed"].max()
    max_utilization = geospatial_df["wind_utilization_rate"].max()

    geospatial_df["normalized_wind_speed"] = geospatial_df["predicted_wind_speed"] / max_wind_speed
    geospatial_df["normalized_utilization"] = geospatial_df["wind_utilization_rate"] / max_utilization

    wind_speed_weight = 0.6
    utilization_weight = 0.4
    geospatial_df["composite_score"] = (
            wind_speed_weight * geospatial_df["normalized_wind_speed"] +
            utilization_weight * geospatial_df["normalized_utilization"]
    )

    # 设置有效点位标志
    geospatial_df["valid"] = (
            (geospatial_df["predicted_wind_speed"] >= 5.0) &
            (geospatial_df["slope"] <= 35) &
            (geospatial_df["elevation"] >= 150) & (geospatial_df["elevation"] <= 1600) &
            (geospatial_df["composite_score"] >= 0.4)
    )

    # -----------------------------
    # 5️⃣ 保存数据
    # -----------------------------
    # 保存地理空间数据
    geospatial_filename = "comprehensive_maale_gilboa_dataset.csv"
    geospatial_df.to_csv(geospatial_filename, index=False)

    # 保存储能系统数据
    storage_filename = "energy_storage_simulation_data.csv"
    storage_data.to_csv(storage_filename, index=True)  # 保留时间索引

    # -----------------------------
    # 6️⃣ 打印统计信息
    # -----------------------------
    print(f"\n📊 地理空间数据集统计:")
    print(f"总数据点: {len(geospatial_df)} 行")
    print(f"地理点位: {n_points} 个")
    print(f"时间跨度: 24 小时")
    print(f"平均风速: {geospatial_df['predicted_wind_speed'].mean():.2f} m/s")
    print(
        f"风速范围: {geospatial_df['predicted_wind_speed'].min():.2f} - {geospatial_df['predicted_wind_speed'].max():.2f} m/s")
    print(f"平均利用率: {geospatial_df['wind_utilization_rate'].mean():.1%}")
    print(f"有效点位比例: {geospatial_df['valid'].mean():.1%}")

    print(f"\n🔋 储能系统模拟统计:")
    print(f"风电总能量: {storage_data['wind_power'].sum() * 0.25:.1f} MWh")
    print(f"并网总能量: {storage_data['grid_power'].sum() * 0.25:.1f} MWh")
    print(f"弃风能量: {(storage_data['wind_power'].sum() - storage_data['grid_power'].sum()) * 0.25:.1f} MWh")
    print(
        f"弃风率: {((storage_data['wind_power'].sum() - storage_data['grid_power'].sum()) / storage_data['wind_power'].sum() * 100):.2f}%")
    print(f"最大风电功率: {storage_data['wind_power'].max():.1f} MW")
    print(f"超限时段: {np.sum(storage_data['wind_power'] > 20)}/{len(storage_data)}")
    print(f"SOC范围: {storage_data['storage_soc'].min():.1f}%-{storage_data['storage_soc'].max():.1f}%")

    print(f"\n💾 数据已保存到:")
    print(f"- 地理空间数据: {geospatial_filename}")
    print(f"- 储能系统数据: {storage_filename}")

    return geospatial_df, storage_data


def create_enhanced_sample_data():
    """
    创建增强的模拟数据 - 专门为每个图表设计有代表性的场景
    """
    np.random.seed(42)
    periods = 96  # 24小时 * 4（15分钟间隔）
    index = pd.date_range('2024-01-01 00:00', periods=periods, freq='15T')

    # 电网限制和储能参数
    grid_limit = 20  # MW
    battery_capacity = 30  # MWh
    max_charge_power = 8  # MW
    max_discharge_power = 8  # MW

    # 创建专门设计的山地风电出力模式
    t = np.linspace(0, 4 * np.pi, periods)

    # 基础日变化模式
    daily_pattern = 0.6 + 0.3 * np.sin(t - np.pi / 2)

    # 专门设计几个关键场景来展示不同图表
    morning_peak = 0.5 * np.exp(-((t - 1.5 * np.pi) ** 2) / 0.3)
    afternoon_peak = 0.6 * np.exp(-((t - 2.5 * np.pi) ** 2) / 0.25)
    ramp_event_1 = 0.4 * np.exp(-((t - 0.8 * np.pi) ** 2) / 0.1)
    ramp_event_2 = -0.35 * np.exp(-((t - 3.2 * np.pi) ** 2) / 0.08)
    night_fluctuation = 0.25 * np.sin(8 * t) * np.exp(-((t - 3.8 * np.pi) ** 2) / 0.5)

    # 随机波动
    random_component = 0.15 * np.sin(12 * t) + 0.1 * np.sin(20 * t)
    noise = 0.1 * np.random.normal(size=periods)

    # 组合生成风电出力
    wind_power = (25 * daily_pattern +
                  18 * morning_peak +
                  20 * afternoon_peak +
                  15 * ramp_event_1 +
                  12 * ramp_event_2 +
                  10 * night_fluctuation +
                  8 * random_component +
                  6 * noise)

    wind_power = np.clip(wind_power, 3, 45)

    # 初始化数组
    grid_power = np.zeros(periods)
    battery_power = np.zeros(periods)
    soc = np.zeros(periods)
    soc[0] = 50  # 初始SOC

    # 模拟更智能的储能调度策略
    for i in range(periods):
        current_wind = wind_power[i]
        current_soc = soc[i]

        # 计算风电功率与电网限制的差值
        power_diff = current_wind - grid_limit

        if power_diff > 0:  # 需要削峰
            available_charge = min(
                (100 - current_soc) * battery_capacity / 100 * 4,
                max_charge_power,
                power_diff
            )

            battery_power[i] = -available_charge
            grid_power[i] = current_wind - available_charge

            if grid_power[i] > grid_limit:
                additional_curtailment = grid_power[i] - grid_limit
                grid_power[i] = grid_limit

        elif current_wind < 10:  # 需要填谷
            needed_power = 10 - current_wind

            available_discharge = min(
                (current_soc - 20) * battery_capacity / 100 * 4,
                max_discharge_power,
                needed_power
            )

            if available_discharge > 0.5:
                battery_power[i] = available_discharge
                grid_power[i] = current_wind + available_discharge
            else:
                battery_power[i] = 0
                grid_power[i] = current_wind
        else:  # 正常范围
            battery_power[i] = 0
            grid_power[i] = current_wind

        # 更新SOC (15分钟间隔)
        if i < periods - 1:
            soc_change = -battery_power[i] * 0.25 / battery_capacity * 100
            soc[i + 1] = max(20, min(90, current_soc + soc_change))

    # 最终确保并网功率不超过限制
    grid_power = np.clip(grid_power, 0, grid_limit)

    data = pd.DataFrame({
        'wind_power': wind_power,
        'grid_power': grid_power,
        'battery_power': battery_power,
        'storage_soc': soc
    }, index=index)

    return data


# 运行生成函数
if __name__ == "__main__":
    geospatial_data, storage_data = create_comprehensive_dataset()