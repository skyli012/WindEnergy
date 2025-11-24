import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
import json

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
# 2️⃣ 在 Polygon 内生成随机点
# -----------------------------
n_points = 1000
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

    # --------------------------
    # 3️⃣ 为每个点生成 24 小时（每10分钟）气象数据
    # --------------------------
    total_intervals = 144  # 24小时 * 6 = 144个10分钟间隔

    # 为每个点生成基础气象特征
    base_temperature = np.random.normal(18, 3)  # 基础温度
    base_humidity = np.random.normal(65, 10)  # 基础湿度
    base_wind_direction = np.random.uniform(0, 360)  # 基础风向

    for interval in range(total_intervals):
        # 计算当前时间（小时和分钟）
        total_minutes = interval * 10
        hour = total_minutes // 60
        minute = total_minutes % 60

        # --------------------------
        # 气象数据生成（与风速相关联）
        # --------------------------

        # 1. 温度 - 日变化模式
        temp_variation = 8 * np.sin(2 * np.pi * (interval / 144 - 0.25))  # 日变化
        temperature = base_temperature + temp_variation + np.random.normal(0, 1)
        temperature = np.clip(temperature, 5, 35)  # 合理温度范围

        # 2. 相对湿度 - 与温度负相关
        humidity_variation = -15 * np.sin(2 * np.pi * (interval / 144 - 0.25))  # 与温度反相
        relative_humidity = base_humidity + humidity_variation + np.random.normal(0, 5)
        relative_humidity = np.clip(relative_humidity, 20, 95)  # 合理湿度范围

        # 3. 风向 - 有日变化但相对稳定
        wind_direction_variation = np.random.normal(0, 15)  # 小范围波动
        wind_direction = (base_wind_direction + wind_direction_variation) % 360

        # 4. 阵风方向 - 与主风向相关但有偏差
        gust_direction_variation = np.random.normal(0, 25)
        gust_direction = (wind_direction + gust_direction_variation) % 360

        # 5. 风速预测（核心变量）
        # 日周期变化（早晚高，中午低）
        daily_variation = 2 * np.sin(2 * np.pi * (interval / 144 - 0.25))
        # 随机波动
        random_fluctuation = np.random.normal(0, 0.8)
        # 短期趋势
        trend_component = 0.5 * np.sin(2 * np.pi * interval / 36)

        # 考虑气象因素对风速的影响
        temp_effect = (temperature - 18) * 0.1  # 温度影响
        humidity_effect = (relative_humidity - 65) * 0.02  # 湿度影响

        wind_speed = (7.5 + daily_variation + random_fluctuation +
                      trend_component + temp_effect + humidity_effect)
        wind_speed = np.clip(wind_speed, 4, 12)

        # 6. 阵风速度 - 通常比平均风速高
        gust_factor = np.random.uniform(1.1, 1.4)  # 阵风系数
        gust_speed = wind_speed * gust_factor + np.random.normal(0, 0.5)
        gust_speed = np.clip(gust_speed, wind_speed, 15)  # 阵风不能小于平均风速

        # 7. 风向标准差 - 表示风向稳定性
        wind_direction_std = np.random.gamma(2, 3)  # 偏右分布，多数时候稳定
        wind_direction_std = np.clip(wind_direction_std, 2, 30)

        # 8. 降雨量 - 与湿度和风速相关
        rainfall_prob = max(0, (relative_humidity - 70) / 30)  # 湿度高时降雨概率大
        if np.random.random() < rainfall_prob * 0.1:  # 控制降雨频率
            rainfall = np.random.exponential(0.5)  # 小到中雨
            # 降雨时风速通常会增加
            wind_speed = wind_speed * np.random.uniform(1.05, 1.2)
        else:
            rainfall = 0.0

        # 创建时间戳
        timestamp = f"2024-01-01 {hour:02d}:{minute:02d}:00"

        rows.append({
            "point_id": point_id,
            "timestamp": timestamp,
            "hour": hour,
            "minute": minute,
            "time_interval": interval,
            "lat": lat,
            "lon": lon,
            "elevation": elevation,
            "slope": slope,

            # 核心气象字段
            "predicted_wind_speed": round(wind_speed, 2),
            "relative_humidity": round(relative_humidity, 1),
            "temperature_c": round(temperature, 1),
            "wind_direction": round(wind_direction, 1),
            "gust_direction": round(gust_direction, 1),
            "gust_speed": round(gust_speed, 2),
            "wind_direction_std": round(wind_direction_std, 1),
            "rainfall_mm": round(rainfall, 2),

            # 基础设施字段
            "grid_proximity": grid_proximity,
            "road_distance": road_distance,
            "residential_distance": residential_distance,
            "heritage_distance": heritage_distance,
            "geology_distance": geology_distance,
            "water_distance": water_distance,
            "cost": cost
        })

# -----------------------------
# 4️⃣ 保存为 CSV
# -----------------------------
df = pd.DataFrame(rows)
df.to_csv("maale_gilboa_24h_simulated_points.csv", index=False)

print(f"生成完成！{len(df)} rows 已保存到 maale_gilboa_24h_simulated_points.csv")
print(f"时间范围：从 2024-01-01 00:00:00 到 2024-01-01 23:50:00")
print(f"时间间隔：每10分钟一个数据点")
print(f"\n气象数据统计：")
print(f"风速范围: {df['predicted_wind_speed'].min():.1f} - {df['predicted_wind_speed'].max():.1f} m/s")
print(f"温度范围: {df['temperature_c'].min():.1f} - {df['temperature_c'].max():.1f} °C")
print(f"湿度范围: {df['relative_humidity'].min():.1f} - {df['relative_humidity'].max():.1f} %")
print(f"降雨总量: {df['rainfall_mm'].sum():.2f} mm")
print(f"有降雨的记录: {(df['rainfall_mm'] > 0).sum()} 条")

print(f"\n数据预览（前3个时间点）：")
print(df.head(3))





# maale_gilboa_24h_simulated_points
