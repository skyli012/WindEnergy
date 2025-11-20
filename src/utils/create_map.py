import numpy as np
import streamlit as st
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import Point
import os


def load_fengjie_boundary():
    """加载奉节县边界数据"""
    shp_path = r"D:\Study\thesis\ChinaAdminDivisonSHP-master\4. District\district.shp"
    if not os.path.exists(shp_path):
        return None

    try:
        gdf = gpd.read_file(shp_path)
        name_columns = [col for col in gdf.columns if
                        any(keyword in col.lower() for keyword in ['name', 'dt', 'county', 'xian'])]
        search_terms = ['奉节县', '奉节', 'Fengjie', 'fengjie']

        for col in name_columns:
            for term in search_terms:
                matches = gdf[gdf[col].astype(str).str.contains(term, na=False)]
                if not matches.empty:
                    return matches
        return gdf.iloc[[0]]
    except Exception:
        return None


def create_fengjie_base_map():
    """创建奉节县基础地图"""
    fengjie = load_fengjie_boundary()
    if fengjie is None:
        return None

    geometry = fengjie.geometry.iloc[0]

    if geometry.geom_type == 'Polygon':
        polygons = [geometry]
    elif geometry.geom_type == 'MultiPolygon':
        polygons = list(geometry.geoms)
    else:
        return None

    centroid = geometry.centroid
    center_lat, center_lon = centroid.y, centroid.x

    # 计算边界框以确定合适的缩放级别
    bounds = geometry.bounds
    min_lon, min_lat, max_lon, max_lat = bounds

    return {
        'polygons': polygons,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'geometry': geometry,
        'bounds': bounds
    }


def display_fengjie_standalone_map(height=600):
    """显示奉节县基础地图
    Args:
        height: 地图高度，默认600px
    """
    base_map = create_fengjie_base_map()
    if base_map is None:
        st.error("无法加载地图数据")
        return

    # 地图样式选择
    map_style = st.selectbox(
        "选择地图样式",
        ["open-street-map", "white-bg", "carto-positron", "carto-darkmatter",
         "stamen-terrain", "stamen-toner", "stamen-watercolor"],
        format_func=lambda x: {
            "open-street-map": "OpenStreetMap",
            "white-bg": "白色背景",
            "carto-positron": "浅色主题",
            "carto-darkmatter": "深色主题",
            "stamen-terrain": "地形图",
            "stamen-toner": "黑白地图",
            "stamen-watercolor": "水彩风格"
        }[x],
        key="standalone_map_style"
    )

    fig = go.Figure()

    # 添加边界线
    for polygon in base_map['polygons']:
        lats, lons = [], []
        for point in polygon.exterior.coords:
            lons.append(point[0])
            lats.append(point[1])

        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode='lines',
            line=dict(width=3, color='red'),
            name="奉节县边界",
            showlegend=True,
            hoverinfo='text',
            hovertext='奉节县行政边界'
        ))

    # 地图布局
    fig.update_layout(
        mapbox=dict(
            style=map_style,
            center=dict(lat=base_map['center_lat'], lon=base_map['center_lon']),
            zoom=8.6,
        ),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # 区域信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("行政区划", "奉节县")
    with col2:
        st.metric("所属地区", "重庆市")
    with col3:
        area_km2 = base_map['geometry'].area * 10000
        st.metric("区域面积", f"{area_km2:.0f} km²")


def display_environment(df, height=600):
    """显示风能资源分布
    Args:
        height: 地图高度，默认600px
    """
    base_map = create_fengjie_base_map()
    if base_map is None:
        st.error("无法加载地图数据")
        return

    # 地图样式选择
    map_style = st.selectbox(
        "选择地图样式",
        ["open-street-map", "white-bg", "carto-positron", "stamen-terrain", "stamen-toner"],
        format_func=lambda x: {
            "open-street-map": "OpenStreetMap",
            "white-bg": "白色背景",
            "carto-positron": "浅色主题",
            "stamen-terrain": "地形图",
            "stamen-toner": "黑白地图"
        }[x],
        key="environment_map_style"
    )

    # 数据预处理 - 确保数据格式正确
    try:
        # 创建数据副本以避免修改原始数据
        df_processed = df.copy()

        # 确保必要的列存在
        required_columns = ['lon', 'lat']
        if not all(col in df_processed.columns for col in required_columns):
            st.error(f"数据缺少必要的列: {required_columns}")
            return

        # 检查数据值列（支持多种数据）
        data_column = None
        possible_columns = ['predicted_wind_speed', 'temperature', 'wind_speed', 'value']
        for col in possible_columns:
            if col in df_processed.columns:
                data_column = col
                break

        if data_column is None:
            st.error("未找到可显示的数据列，请确保数据包含风速、温度等数值列")
            return

        # 处理有效点位（如果有valid列）
        if 'valid' in df_processed.columns:
            valid_points = df_processed[df_processed["valid"] == 1]
        else:
            valid_points = df_processed

        # 空间数据处理
        gdf = gpd.GeoDataFrame(
            df_processed,
            geometry=gpd.points_from_xy(df_processed["lon"], df_processed["lat"]),
            crs="EPSG:4326"
        )

        gdf_fengjie = gdf[gdf.within(base_map['geometry'])]
        if gdf_fengjie.empty:
            st.warning("所选数据在奉节县范围内无有效点位")
            return

        fig = go.Figure()

        # 添加边界
        for polygon in base_map['polygons']:
            lats, lons = [], []
            for point in polygon.exterior.coords:
                lons.append(point[0])
                lats.append(point[1])

            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode='lines',
                line=dict(width=3, color='red'),
                name="区域边界",
                showlegend=True
            ))

        # 添加热力图 - 修复colorbar配置
        display_points = gdf_fengjie
        if 'valid' in df_processed.columns:
            display_points = gdf_fengjie[gdf_fengjie["valid"] == 1]

        if not display_points.empty:
            # 根据数据类型设置颜色和标题
            if data_column == 'temperature':
                colorscale = 'Hot'
                colorbar_title = "温度 (°C)"
                hover_template = '<b>温度</b>: %{z:.1f} °C<br>经纬度: (%{lat:.3f}, %{lon:.3f})'
            elif data_column == 'predicted_wind_speed':
                colorscale = 'Viridis'
                colorbar_title = "风速 (m/s)"
                hover_template = '<b>风速</b>: %{z:.1f} m/s<br>经纬度: (%{lat:.3f}, %{lon:.3f})'
            else:
                colorscale = 'Plasma'
                colorbar_title = "数值"
                hover_template = '<b>数值</b>: %{z:.1f}<br>经纬度: (%{lat:.3f}, %{lon:.3f})'

            fig.add_trace(go.Densitymapbox(
                lat=display_points["lat"],
                lon=display_points["lon"],
                z=display_points[data_column],
                radius=25,
                colorscale=colorscale,
                opacity=0.7,
                name=f"{data_column}分布",
                showscale=True,
                hovertemplate=hover_template,
                colorbar=dict(
                    title=colorbar_title  # 修复：移除 titleside
                )
            ))

        # 地图布局
        fig.update_layout(
            mapbox=dict(
                style=map_style,
                center=dict(lat=base_map['center_lat'], lon=base_map['center_lon']),
                zoom=8,
            ),
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            title=f"奉节县{data_column}分布图"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 数据统计
        if not display_points.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_value = display_points[data_column].mean()
                st.metric("平均值", f"{avg_value:.1f}")
            with col2:
                valid_count = len(display_points)
                st.metric("有效点位", f"{valid_count} 个")
            with col3:
                max_value = display_points[data_column].max()
                st.metric("最大值", f"{max_value:.1f}")
            with col4:
                min_value = display_points[data_column].min()
                st.metric("最小值", f"{min_value:.1f}")

    except Exception as e:
        st.error(f"数据处理错误: {str(e)}")
        st.info("请检查数据格式，确保包含经纬度坐标和数值数据")


def display_optimization_map(result, df, height=600):
    """在左侧地图上显示优化结果（风机位置）
    Args:
        height: 地图高度，默认600px
    """
    base_map = create_fengjie_base_map()
    if base_map is None:
        st.error("无法加载地图数据")
        return

    # 地图样式选择
    map_style = st.selectbox(
        "选择地图样式",
        ["open-street-map", "white-bg", "carto-positron", "stamen-terrain"],
        format_func=lambda x: {
            "open-street-map": "OpenStreetMap",
            "white-bg": "白色背景",
            "carto-positron": "浅色主题",
            "stamen-terrain": "地形图"
        }[x],
        key="optimization_map_style"
    )

    # 🔧 修复：兼容不同的结果格式
    try:
        # 尝试不同的键名来获取解决方案
        if "solution" in result:
            sol = result["solution"]
        elif "best_positions" in result:
            sol = result["best_positions"]
        elif "positions" in result:
            sol = result["positions"]
        elif "selected_indices" in result:
            sol = result["selected_indices"]
        else:
            # 如果没有明确的解决方案键，尝试使用第一个可迭代的值
            for key, value in result.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    sol = value
                    break
            else:
                st.error("❌ 无法找到有效的解决方案数据")
                return

        if not sol:
            st.error("❌ 没有找到有效的解决方案")
            return

        # 数据预处理
        df_processed = df.copy()

        # 确保 sol 是有效的索引列表
        if isinstance(sol, (list, np.ndarray)):
            # 过滤掉超出数据范围的索引
            valid_indices = [idx for idx in sol if idx in df_processed.index]
            if not valid_indices:
                st.error("❌ 没有找到有效的风机位置索引")
                return

            turbines = df_processed.loc[valid_indices].copy().reset_index(drop=True)
        else:
            st.error(f"❌ 解决方案格式不正确: {type(sol)}")
            return

        turbines["turbine_id"] = [f"T{i + 1}" for i in range(len(turbines))]

        # 保留奉节县内风机
        turbines_fengjie = turbines[
            turbines.apply(lambda row: Point(row["lon"], row["lat"]).within(base_map['geometry']), axis=1)
        ]

        if turbines_fengjie.empty:
            st.warning("⚠️ 优化结果中没有在奉节县范围内的风机位置")
            return

        fig = go.Figure()

        # 添加奉节县边界线
        for polygon in base_map['polygons']:
            lats, lons = [], []
            for point in polygon.exterior.coords:
                lons.append(point[0])
                lats.append(point[1])

            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode='lines',
                line=dict(width=3, color='red'),
                name="奉节县边界",
                showlegend=True
            ))

        # 添加风能热力图背景
        gdf = gpd.GeoDataFrame(
            df_processed.copy(),
            geometry=gpd.points_from_xy(df_processed["lon"], df_processed["lat"]),
            crs="EPSG:4326"
        )
        gdf_fengjie = gdf[gdf.within(base_map['geometry'])]

        # 确定数据列
        data_column = 'predicted_wind_speed'
        if data_column not in gdf_fengjie.columns:
            # 尝试其他可能的列
            for col in ['wind_speed', 'temperature', 'value']:
                if col in gdf_fengjie.columns:
                    data_column = col
                    break

        valid_points = gdf_fengjie
        if 'valid' in gdf_fengjie.columns:
            valid_points = gdf_fengjie[gdf_fengjie["valid"] == 1]

        if not valid_points.empty and data_column in valid_points.columns:
            fig.add_trace(go.Densitymapbox(
                lat=valid_points["lat"],
                lon=valid_points["lon"],
                z=valid_points[data_column],
                radius=20,
                colorscale='Viridis',
                opacity=0.5,
                name="数据背景",
                showscale=True,
                hovertemplate=f'背景{data_column}: %{{z:.1f}}',
                colorbar=dict(title=data_column)
            ))

        # 添加优化后的风机位置
        if not turbines_fengjie.empty:
            fig.add_trace(go.Scattermapbox(
                lat=turbines_fengjie["lat"],
                lon=turbines_fengjie["lon"],
                mode="markers+text",
                marker=dict(
                    color='red',
                    size=14,
                    symbol="circle",
                    opacity=0.9
                ),
                text=turbines_fengjie["turbine_id"],
                textposition="top center",
                hovertext=[
                    f"<b>{row['turbine_id']}</b><br>"
                    f"经度: {row['lon']:.3f}<br>"
                    f"纬度: {row['lat']:.3f}<br>"
                    + (f"风速: {row['predicted_wind_speed']:.1f} m/s<br>" if 'predicted_wind_speed' in row else "")
                    + (f"功率密度: {row['wind_power_density']:.0f} W/m²" if 'wind_power_density' in row else "")
                    for _, row in turbines_fengjie.iterrows()
                ],
                hoverinfo="text",
                name="最优风机位置",
                textfont=dict(size=11, color='black', weight='bold')
            ))

        # 地图布局
        fig.update_layout(
            mapbox=dict(
                style=map_style,
                center=dict(lat=base_map['center_lat'], lon=base_map['center_lon']),
                zoom=8,
            ),
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            title="奉节县风机优化布局图"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 显示基本信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("优化风机数量", len(turbines_fengjie))
        with col2:
            if 'predicted_wind_speed' in turbines_fengjie.columns:
                avg_speed = turbines_fengjie["predicted_wind_speed"].mean()
                st.metric("平均风速", f"{avg_speed:.1f} m/s")
            else:
                st.metric("平均数值", "N/A")
        with col3:
            if 'wind_power_density' in turbines_fengjie.columns:
                total_power = turbines_fengjie["wind_power_density"].sum()
                st.metric("总功率密度", f"{total_power:.0f} W/m²")
            else:
                st.metric("总数值", "N/A")
        with col4:
            # 计算风机间距
            if len(turbines_fengjie) > 1:
                from geopy.distance import geodesic
                min_distance = float('inf')
                for i in range(len(turbines_fengjie)):
                    for j in range(i + 1, len(turbines_fengjie)):
                        coord1 = (turbines_fengjie.iloc[i]['lat'], turbines_fengjie.iloc[i]['lon'])
                        coord2 = (turbines_fengjie.iloc[j]['lat'], turbines_fengjie.iloc[j]['lon'])
                        dist = geodesic(coord1, coord2).km
                        if dist < min_distance:
                            min_distance = dist
                st.metric("最小间距", f"{min_distance:.1f} km")
            else:
                st.metric("最小间距", "N/A")

    except Exception as e:
        st.error(f"优化结果显示错误: {str(e)}")
        # 显示调试信息
        with st.expander("🔍 调试信息"):
            st.write("结果字典的键:", list(result.keys()))
            st.write("结果类型:", type(result))
            st.write("错误详情:", str(e))