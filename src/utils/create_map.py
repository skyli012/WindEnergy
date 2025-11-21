import numpy as np
import streamlit as st
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import Point
import os
import json
import pandas as pd


def load_maale_gilboa_boundary():
    """加载Maale Gilboa区域边界数据"""
    geojson_path = r"C:\Users\lhl\Downloads\map (10).geojson"
    if not os.path.exists(geojson_path):
        return None

    try:
        gdf = gpd.read_file(geojson_path)
        return gdf
    except Exception as e:
        st.error(f"加载地图数据错误: {str(e)}")
        return None


def create_maale_gilboa_base_map():
    """创建Maale Gilboa基础地图"""
    maale_gilboa = load_maale_gilboa_boundary()
    if maale_gilboa is None:
        return None

    geometry = maale_gilboa.geometry.iloc[0]

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
        'bounds': bounds,
        'gdf': maale_gilboa  # 保留原始GeoDataFrame
    }


def preprocess_wind_data(df):
    """
    预处理风速数据，计算每个坐标点的24小时平均风速

    Parameters:
    - df: 原始数据框，包含24小时记录

    Returns:
    - df_avg: 包含每个坐标点平均风速的数据框
    """
    # 检查必要列是否存在
    required_columns = ['lat', 'lon', 'predicted_wind_speed', 'hour']
    if not all(col in df.columns for col in required_columns):
        st.error(f"数据缺少必要的列: {required_columns}")
        return None

    try:
        # 计算每个坐标点的平均风速
        df_avg = df.groupby(['lat', 'lon']).agg({
            'predicted_wind_speed': 'mean',
            'elevation': 'first',
            'slope': 'first',
            'grid_proximity': 'first',
            'road_distance': 'first',
            'residential_distance': 'first',
            'heritage_distance': 'first',
            'geology_distance': 'first',
            'water_distance': 'first',
            'cost': 'first'
        }).reset_index()

        # 重命名风速列为平均风速
        df_avg = df_avg.rename(columns={'predicted_wind_speed': 'avg_wind_speed'})

        return df_avg

    except Exception as e:
        st.error(f"数据预处理错误: {str(e)}")
        return None


def display_maale_gilboa_standalone_map(height=600):
    """显示Maale Gilboa基础地图"""
    base_map = create_maale_gilboa_base_map()
    if base_map is None:
        st.error("无法加载地图数据")
        return

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
            name="Maale Gilboa边界",
            showlegend=True,
            hoverinfo='text',
            hovertext='Maale Gilboa区域边界'
        ))

    # 地图布局 - 默认使用OpenStreetMap
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",  # 固定使用OpenStreetMap
            center=dict(lat=base_map['center_lat'], lon=base_map['center_lon']),
            zoom=12,  # 调整缩放级别以适应Maale Gilboa区域
        ),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # 区域信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("区域名称", "Maale Gilboa")
    with col2:
        st.metric("所属地区", "以色列")
    with col3:
        area_km2 = base_map['geometry'].area * 10000
        st.metric("区域面积", f"{area_km2:.0f} km²")


def display_environment(df, height=600):
    """显示风能资源分布 - 使用平均风速"""
    base_map = create_maale_gilboa_base_map()
    if base_map is None:
        st.error("无法加载地图数据")
        return

    # 预处理数据，计算平均风速
    with st.spinner('正在计算平均风速...'):
        df_processed = preprocess_wind_data(df)

    if df_processed is None:
        return

    # 数据预处理 - 确保数据格式正确
    try:
        # 确保必要的列存在
        required_columns = ['lon', 'lat', 'avg_wind_speed']
        if not all(col in df_processed.columns for col in required_columns):
            st.error(f"处理后的数据缺少必要的列: {required_columns}")
            return

        # 空间数据处理
        gdf = gpd.GeoDataFrame(
            df_processed,
            geometry=gpd.points_from_xy(df_processed["lon"], df_processed["lat"]),
            crs="EPSG:4326"
        )

        gdf_maale_gilboa = gdf[gdf.within(base_map['geometry'])]
        if gdf_maale_gilboa.empty:
            st.warning("所选数据在Maale Gilboa区域内无有效点位")
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

        # 添加热力图 - 使用平均风速
        if not gdf_maale_gilboa.empty:
            fig.add_trace(go.Densitymapbox(
                lat=gdf_maale_gilboa["lat"],
                lon=gdf_maale_gilboa["lon"],
                z=gdf_maale_gilboa["avg_wind_speed"],
                radius=25,
                colorscale='Viridis',
                opacity=0.7,
                name="平均风速分布",
                showscale=True,
                hovertemplate=(
                    '<b>24小时平均风速</b>: %{z:.2f} m/s<br>'
                    '经纬度: (%{lat:.3f}, %{lon:.3f})<br>'
                    '<extra></extra>'
                ),
                colorbar=dict(
                    title="平均风速 (m/s)"
                )
            ))

        # 地图布局 - 默认使用OpenStreetMap
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",  # 固定使用OpenStreetMap
                center=dict(lat=base_map['center_lat'], lon=base_map['center_lon']),
                zoom=12,  # 调整缩放级别
            ),
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            title="Maale Gilboa区域 24小时平均风速分布图"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 数据统计
        if not gdf_maale_gilboa.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_value = gdf_maale_gilboa["avg_wind_speed"].mean()
                st.metric("平均风速", f"{avg_value:.2f} m/s")
            with col2:
                valid_count = len(gdf_maale_gilboa)
                st.metric("有效点位", f"{valid_count} 个")
            with col3:
                max_value = gdf_maale_gilboa["avg_wind_speed"].max()
                st.metric("最大平均风速", f"{max_value:.2f} m/s")
            with col4:
                min_value = gdf_maale_gilboa["avg_wind_speed"].min()
                st.metric("最小平均风速", f"{min_value:.2f} m/s")

            # 显示风速分布信息
            st.subheader("风速分布统计")
            col5, col6, col7 = st.columns(3)
            with col5:
                wind_std = gdf_maale_gilboa["avg_wind_speed"].std()
                st.metric("风速标准差", f"{wind_std:.2f} m/s")
            with col6:
                wind_median = gdf_maale_gilboa["avg_wind_speed"].median()
                st.metric("风速中位数", f"{wind_median:.2f} m/s")
            with col7:
                # 计算优质风能点位（假设平均风速 > 6 m/s 为优质点位）
                good_wind_points = len(gdf_maale_gilboa[gdf_maale_gilboa["avg_wind_speed"] > 6])
                st.metric("优质风能点位", f"{good_wind_points} 个")

    except Exception as e:
        st.error(f"数据处理错误: {str(e)}")
        st.info("请检查数据格式，确保包含经纬度坐标和风速数据")


def display_optimization_map(result, df, height=600):
    """在左侧地图上显示优化结果（风场位置） - 使用平均风速"""
    base_map = create_maale_gilboa_base_map()
    if base_map is None:
        st.error("无法加载地图数据")
        return

    # 预处理数据，计算平均风速
    with st.spinner('正在计算平均风速...'):
        df_processed = preprocess_wind_data(df)

    if df_processed is None:
        return

    # 兼容不同的结果格式
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

        # 关键修改：处理索引映射问题
        if isinstance(sol, (list, np.ndarray)):
            # 方法1：如果sol是坐标索引
            if max(sol) < len(df_processed):
                # 直接使用预处理数据的索引
                valid_indices = [idx for idx in sol if idx < len(df_processed)]
                turbines = df_processed.iloc[valid_indices].copy().reset_index(drop=True)
            else:
                # 获取原始数据中的唯一坐标点
                unique_coords = df[['lat', 'lon']].drop_duplicates().reset_index(drop=True)

                # 找出被选中的坐标点在唯一坐标列表中的索引
                selected_coord_indices = []
                for idx in sol:
                    if idx < len(df):
                        # 获取原始数据中该索引的坐标
                        original_coord = (df.iloc[idx]['lat'], df.iloc[idx]['lon'])
                        # 在唯一坐标列表中查找这个坐标
                        for i, coord in enumerate(unique_coords.itertuples()):
                            if abs(coord.lat - original_coord[0]) < 0.0001 and abs(
                                    coord.lon - original_coord[1]) < 0.0001:
                                selected_coord_indices.append(i)
                                break

                # 去重
                selected_coord_indices = list(set(selected_coord_indices))

                if not selected_coord_indices:
                    st.error("❌ 无法映射索引到预处理数据")
                    return

                # 从预处理数据中获取对应的点
                turbines = df_processed.iloc[selected_coord_indices].copy().reset_index(drop=True)

        else:
            st.error(f"❌ 解决方案格式不正确: {type(sol)}")
            return

        # 修改：将选中的点位分组为风场
        # 假设每个风场包含固定数量的风机（根据界面设置）
        n_farms = st.session_state.get('n_farms', 2)  # 从session_state获取风场数量
        n_turbines_per_farm = st.session_state.get('n_turbines_per_farm', 4)  # 从session_state获取单场风机数

        # 将选中的点位分组到不同的风场
        farms = []
        for i in range(n_farms):
            start_idx = i * n_turbines_per_farm
            end_idx = start_idx + n_turbines_per_farm
            farm_turbines = turbines.iloc[start_idx:end_idx].copy().reset_index(drop=True)

            if len(farm_turbines) > 0:
                # 计算风场的中心位置
                center_lat = farm_turbines['lat'].mean()
                center_lon = farm_turbines['lon'].mean()

                # 计算风场的平均风速
                avg_wind_speed = farm_turbines[
                    'avg_wind_speed'].mean() if 'avg_wind_speed' in farm_turbines.columns else 0

                farms.append({
                    'farm_id': f"风场{i + 1}",
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'avg_wind_speed': avg_wind_speed,
                    'turbine_count': len(farm_turbines),
                    'turbines': farm_turbines  # 保留该风场的所有风机信息
                })

        # 保留Maale Gilboa区域内的风场
        farms_maale_gilboa = []
        for farm in farms:
            if Point(farm['center_lon'], farm['center_lat']).within(base_map['geometry']):
                farms_maale_gilboa.append(farm)

        if not farms_maale_gilboa:
            st.warning("⚠️ 优化结果中没有在Maale Gilboa区域内的风场位置")
            return

        fig = go.Figure()

        # 添加区域边界线
        for polygon in base_map['polygons']:
            lats, lons = [], []
            for point in polygon.exterior.coords:
                lons.append(point[0])
                lats.append(point[1])

            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode='lines',
                line=dict(width=3, color='red'),
                name="Maale Gilboa边界",
                showlegend=True
            ))

        # 添加风能热力图背景 - 使用平均风速
        gdf = gpd.GeoDataFrame(
            df_processed.copy(),
            geometry=gpd.points_from_xy(df_processed["lon"], df_processed["lat"]),
            crs="EPSG:4326"
        )
        gdf_maale_gilboa = gdf[gdf.within(base_map['geometry'])]

        if not gdf_maale_gilboa.empty and 'avg_wind_speed' in gdf_maale_gilboa.columns:
            fig.add_trace(go.Densitymapbox(
                lat=gdf_maale_gilboa["lat"],
                lon=gdf_maale_gilboa["lon"],
                z=gdf_maale_gilboa["avg_wind_speed"],
                radius=20,
                colorscale='Viridis',
                opacity=0.5,
                name="平均风速背景",
                showscale=True,
                hovertemplate='24小时平均风速: %{z:.2f} m/s',
                colorbar=dict(title="平均风速 (m/s)")
            ))

        # 修改：添加风场位置而不是单个风机
        if farms_maale_gilboa:
            # 为不同的风场使用不同的颜色
            colors = ['red', 'blue', 'green', 'orange', 'purple']

            for i, farm in enumerate(farms_maale_gilboa):
                color = colors[i % len(colors)]

                # 添加风场中心位置
                fig.add_trace(go.Scattermapbox(
                    lat=[farm['center_lat']],
                    lon=[farm['center_lon']],
                    mode="markers+text",
                    marker=dict(
                        color=color,
                        size=20,  # 风场标记比风机大
                        symbol="circle",
                        opacity=0.9
                    ),
                    text=[farm['farm_id']],
                    textposition="top center",
                    hovertext=[
                        f"<b>{farm['farm_id']}</b><br>"
                        f"中心经度: {farm['center_lon']:.3f}<br>"
                        f"中心纬度: {farm['center_lat']:.3f}<br>"
                        f"风机数量: {farm['turbine_count']} 台<br>"
                        + (f"平均风速: {farm['avg_wind_speed']:.2f} m/s<br>" if farm['avg_wind_speed'] > 0 else "")
                    ],
                    hoverinfo="text",
                    name=farm['farm_id'],
                    textfont=dict(size=12, color='black', weight='bold')
                ))

        # 地图布局 - 默认使用OpenStreetMap
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",  # 固定使用OpenStreetMap
                center=dict(lat=base_map['center_lat'], lon=base_map['center_lon']),
                zoom=12,  # 调整缩放级别
            ),
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            title=f"Maale Gilboa区域风场优化布局图 - 共{len(farms_maale_gilboa)}个风场"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 显示基本信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("优化风场数量", len(farms_maale_gilboa))
        with col2:
            total_turbines = sum(farm['turbine_count'] for farm in farms_maale_gilboa)
            st.metric("总风机数量", total_turbines)
        with col3:
            if farms_maale_gilboa:
                avg_speed = np.mean([farm['avg_wind_speed'] for farm in farms_maale_gilboa])
                st.metric("平均风速", f"{avg_speed:.2f} m/s")
            else:
                st.metric("平均风速", "N/A")
        with col4:
            # 计算风场间距
            if len(farms_maale_gilboa) > 1:
                from geopy.distance import geodesic
                min_distance = float('inf')
                for i in range(len(farms_maale_gilboa)):
                    for j in range(i + 1, len(farms_maale_gilboa)):
                        coord1 = (farms_maale_gilboa[i]['center_lat'], farms_maale_gilboa[i]['center_lon'])
                        coord2 = (farms_maale_gilboa[j]['center_lat'], farms_maale_gilboa[j]['center_lon'])
                        dist = geodesic(coord1, coord2).km
                        if dist < min_distance:
                            min_distance = dist
                st.metric("最小风场间距", f"{min_distance:.1f} km")
            else:
                st.metric("最小风场间距", "N/A")

    except Exception as e:
        st.error(f"优化结果显示错误: {str(e)}")
        # 显示调试信息
        with st.expander("🔍 调试信息"):
            st.write("结果字典的键:", list(result.keys()))
            st.write("结果类型:", type(result))
            st.write("错误详情:", str(e))
            import traceback
            st.write("完整错误跟踪:")
            st.code(traceback.format_exc())