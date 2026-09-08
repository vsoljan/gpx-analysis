# -*- coding: UTF-8 -*-
"""
MIT License

Copyright (c) 2026 Vinko Soljan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import datetime
import math
import branca.colormap as cm
import folium
import gpxpy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import streamlit as st
from streamlit_folium import st_folium
import tzlocal  # Added for auto local timezone conversion

# Unit Conversion Constants
MS_TO_KMH = 3.6
MS_TO_KT = 1.94384
METERS_TO_KM = 0.001
METERS_TO_NM = 0.000539957
MOVING_SPEED_THRESHOLD_MS = 0.25  # ~0.5 knots threshold for active movement


# =====================================================================
# 1. VECTORIZED GEOMETRY & BEARING CALCULATIONS
# =====================================================================
def calculate_haversine_and_bearing_vectorized(lats, lons):
    """Vectorized calculation of segment distances (Haversine) and initial bearings
    using pure NumPy for massive performance gains over iterative geodesic loops.
    """
    R = 6371000.0  # Earth's mean radius in meters

    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    lat1 = lat_rad[:-1]
    lat2 = lat_rad[1:]
    lon1 = lon_rad[:-1]
    lon2 = lon_rad[1:]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine distance formula
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = np.concatenate(([0.0], R * c))

    # Forward bearing calculation
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (
        np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )
    initial_bearings = np.degrees(np.arctan2(x, y))
    bearings = np.concatenate(([0.0], (initial_bearings + 360) % 360))

    return distances, bearings


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================
def format_seconds_to_str(total_seconds):
    """Converts seconds to a readable 'Xh Ym Zs' string."""
    total_seconds = int(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


# =====================================================================
# 3. DATA LOADING & CLEANING (CACHED)
# =====================================================================
@st.cache_data
def load_gpx_to_dataframe(file_bytes):
    """Parses raw GPX file bytes into a structured Pandas DataFrame and extracts activity metadata."""
    gpx = gpxpy.parse(file_bytes)

    # 1. Extract Activity Name
    activity_name = gpx.name if gpx.name else "Unnamed Activity"
    if (
        activity_name == "Unnamed Activity"
        and gpx.tracks
        and gpx.tracks[0].name
    ):
        activity_name = gpx.tracks[0].name

    # 2. Parse Points
    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append(
                    {
                        "time": point.time,
                        "latitude": point.latitude,
                        "longitude": point.longitude,
                        "elevation": point.elevation,
                    }
                )

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Convert times to local timezone
    local_tz = tzlocal.get_localzone()
    start_time_local = (
        df["time"].min().tz_convert(local_tz) if not df.empty else None
    )
    end_time_local = (
        df["time"].max().tz_convert(local_tz) if not df.empty else None
    )

    activity_date = (
        start_time_local.strftime("%d.%m.%Y")
        if pd.notnull(start_time_local)
        else "N/A"
    )
    start_time_str = (
        start_time_local.strftime("%H:%M:%S")
        if pd.notnull(start_time_local)
        else "N/A"
    )
    end_time_str = (
        end_time_local.strftime("%H:%M:%S")
        if pd.notnull(end_time_local)
        else "N/A"
    )

    if pd.notnull(start_time_local) and pd.notnull(end_time_local):
        elapsed_seconds = (
            df["time"].max() - df["time"].min()
        ).total_seconds()
        elapsed_str = format_seconds_to_str(elapsed_seconds)
    else:
        elapsed_str = "N/A"

    metadata = {
        "name": activity_name,
        "date": activity_date,
        "start_time": start_time_str,
        "end_time": end_time_str,
        "elapsed_duration": elapsed_str,
    }

    return df, metadata


@st.cache_data
def clean_track_points(df, max_speed_ms=20.0, max_accel_ms2=6.0):
    """Filters out GPS position spikes and impossible accelerations using vectorized ops."""
    df = df.copy()

    distances, bearings = calculate_haversine_and_bearing_vectorized(
        df["latitude"].values, df["longitude"].values
    )

    df["segment_distance_m"] = distances
    df["heading_deg"] = bearings
    df["time_delta_s"] = df["time"].diff().dt.total_seconds().fillna(0)

    df["instant_speed_ms"] = np.where(
        df["time_delta_s"] > 0,
        df["segment_distance_m"] / df["time_delta_s"],
        0.0,
    )

    valid_mask = df["instant_speed_ms"] <= max_speed_ms
    speed_diff = df["instant_speed_ms"].diff().abs()
    accel = speed_diff / df["time_delta_s"]
    valid_mask = valid_mask & (accel.fillna(0) <= max_accel_ms2)
    valid_mask.iloc[0] = True

    clean_df = df[valid_mask].reset_index(drop=True)

    clean_distances, clean_bearings = calculate_haversine_and_bearing_vectorized(
        clean_df["latitude"].values, clean_df["longitude"].values
    )

    clean_df["segment_distance_m"] = clean_distances
    clean_df["heading_deg"] = clean_bearings
    clean_df["cum_distance_m"] = clean_df["segment_distance_m"].cumsum()
    clean_df["time_delta_s"] = (
        clean_df["time"].diff().dt.total_seconds().fillna(0)
    )

    clean_df["instant_speed_ms"] = np.where(
        clean_df["time_delta_s"] > 0,
        clean_df["segment_distance_m"] / clean_df["time_delta_s"],
        0.0,
    )
    clean_df["speed_kt"] = clean_df["instant_speed_ms"] * MS_TO_KT

    return clean_df


@st.cache_data
def smooth_track_savgol(df, smooth_window=15, smooth_polyorder=3):
    """Smooths latitude and longitude using a Savitzky-Golay filter."""
    clean_df = df.copy()

    n_points = len(clean_df)
    w = min(smooth_window, n_points - (1 - n_points % 2))
    if w % 2 == 0:
        w -= 1

    if w > smooth_polyorder and w >= 5:
        clean_df["latitude"] = savgol_filter(
            clean_df["latitude"], w, smooth_polyorder
        )
        clean_df["longitude"] = savgol_filter(
            clean_df["longitude"], w, smooth_polyorder
        )

    clean_distances, clean_bearings = calculate_haversine_and_bearing_vectorized(
        clean_df["latitude"].values, clean_df["longitude"].values
    )

    clean_df["segment_distance_m"] = clean_distances
    clean_df["heading_deg"] = clean_bearings
    clean_df["cum_distance_m"] = clean_df["segment_distance_m"].cumsum()
    clean_df["time_delta_s"] = (
        clean_df["time"].diff().dt.total_seconds().fillna(0)
    )
    clean_df["instant_speed_ms"] = np.where(
        clean_df["time_delta_s"] > 0,
        clean_df["segment_distance_m"] / clean_df["time_delta_s"],
        0.0,
    )
    clean_df["speed_kt"] = clean_df["instant_speed_ms"] * MS_TO_KT

    return clean_df


# =====================================================================
# 4. SPEED METRICS CALCULATIONS (CACHED)
# =====================================================================
def calculate_max_speed_over_distance(df, target_distance_m=100):
    """Finds peak speed over continuous distance using linear boundary interpolation."""
    cum_dist = df["cum_distance_m"].values
    times = (
        df["time"] - pd.Timestamp("1970-01-01", tz="UTC")
    ).dt.total_seconds().values

    max_speed = 0.0
    n = len(df)
    right = 0

    for left in range(n):
        while (
            right < n
            and (cum_dist[right] - cum_dist[left]) < target_distance_m
        ):
            right += 1

        if right < n:
            total_dist_window = cum_dist[right] - cum_dist[left]
            total_time_window = times[right] - times[left]

            if (
                total_dist_window > target_distance_m * 1.5
                or total_time_window > 300
                or total_time_window <= 0
            ):
                continue

            prev_dist_covered = cum_dist[right - 1] - cum_dist[left]
            prev_time_covered = times[right - 1] - times[left]

            segment_dist = cum_dist[right] - cum_dist[right - 1]
            segment_time = times[right] - times[right - 1]

            if segment_dist > 0 and segment_time > 0:
                needed_dist = target_distance_m - prev_dist_covered
                interpolated_extra_time = (
                    needed_dist / segment_dist
                ) * segment_time
                exact_time = prev_time_covered + interpolated_extra_time

                if exact_time > 0:
                    speed = target_distance_m / exact_time
                    if speed > max_speed:
                        max_speed = speed

    return max_speed


def format_results(speed_ms):
    """Format speed results"""
    return {
        "kmh": round(speed_ms * MS_TO_KMH, 2),
        "kt": round(speed_ms * MS_TO_KT, 2),
    }


@st.cache_data
def calculate_metrics(df):
    """Calculates total distance, moving time, and time/distance-based peak speeds."""
    df_filtered = smooth_track_savgol(df)
    total_dist_m = df_filtered["segment_distance_m"].sum()
    total_dist_km = round(total_dist_m * METERS_TO_KM, 2)
    total_dist_nm = round(total_dist_m * METERS_TO_NM, 2)

    # Calculate Moving Time (speed > MOVING_SPEED_THRESHOLD_MS)
    moving_mask = df["instant_speed_ms"] > MOVING_SPEED_THRESHOLD_MS
    moving_seconds = df.loc[moving_mask, "time_delta_s"].sum()
    moving_duration_str = format_seconds_to_str(moving_seconds)

    time_df = df.set_index("time")

    roll_2s_dist = time_df["segment_distance_m"].rolling("2s").sum()
    roll_2s_time = time_df["time_delta_s"].rolling("2s").sum()
    max_2s_ms = (roll_2s_dist / roll_2s_time.replace(0, np.nan)).max()

    roll_10s_dist = time_df["segment_distance_m"].rolling("10s").sum()
    roll_10s_time = time_df["time_delta_s"].rolling("10s").sum()
    max_10s_ms = (roll_10s_dist / roll_10s_time.replace(0, np.nan)).max()

    roll_1min_dist = time_df["segment_distance_m"].rolling("1min").sum()
    roll_1min_time = time_df["time_delta_s"].rolling("1min").sum()
    max_1min_ms = (roll_1min_dist / roll_1min_time.replace(0, np.nan)).max()

    roll_5min_dist = time_df["segment_distance_m"].rolling("5min").sum()
    roll_5min_time = time_df["time_delta_s"].rolling("5min").sum()
    max_5min_ms = (roll_5min_dist / roll_5min_time.replace(0, np.nan)).max()

    roll_30min_dist = time_df["segment_distance_m"].rolling("30min").sum()
    roll_30min_time = time_df["time_delta_s"].rolling("30min").sum()
    max_30min_ms = (roll_30min_dist / roll_30min_time.replace(0, np.nan)).max()

    max_100m_ms = calculate_max_speed_over_distance(
        df, target_distance_m=100
    )
    max_250m_ms = calculate_max_speed_over_distance(
        df, target_distance_m=250
    )
    max_500m_ms = calculate_max_speed_over_distance(
        df, target_distance_m=500
    )
    max_1852m_ms = calculate_max_speed_over_distance(
        df, target_distance_m=1852
    )

    speed_metrics = {
        "2s max": format_results(max_2s_ms),
        "10s max": format_results(max_10s_ms),
        "1min max": format_results(max_1min_ms),
        "5min max": format_results(max_5min_ms),
        "30min max": format_results(max_30min_ms),
        "100m max": format_results(max_100m_ms),
        "250m max": format_results(max_250m_ms),
        "500m max": format_results(max_500m_ms),
        "1852m max": format_results(max_1852m_ms),
    }

    return speed_metrics, total_dist_km, total_dist_nm, moving_duration_str


# =====================================================================
# 5. POLAR DIAGRAM PLOTTING
# =====================================================================
def infer_wind_direction_two_step(df, bin_size_deg=5):
    """Infers wind direction by averaging upwind and downwind minimums."""
    active_df = df[df["speed_kt"] >= 2.0].copy()
    if active_df.empty:
        return 280.0

    bins = np.arange(0, 360 + bin_size_deg, bin_size_deg)
    labels = bins[:-1] + bin_size_deg / 2

    active_df["heading_bin"] = pd.cut(
        active_df["heading_deg"],
        bins=bins,
        include_lowest=True,
        labels=labels,
    )

    binned_speeds = (
        active_df.groupby("heading_bin", observed=False)["speed_kt"]
        .max()
        .fillna(0)
    )

    smoothed_speeds = binned_speeds.rolling(
        window=3, center=True, min_periods=1
    ).mean()

    upwind_dir_deg = float(smoothed_speeds.idxmin())

    opposite_deg = (upwind_dir_deg + 180) % 360
    downwind_candidates = []
    for bin_deg in smoothed_speeds.index:
        diff = abs((bin_deg - opposite_deg + 180) % 360 - 180)
        if diff <= 60:
            downwind_candidates.append(bin_deg)

    if downwind_candidates:
        downwind_sub_series = smoothed_speeds.loc[downwind_candidates]
        downwind_min_deg = float(downwind_sub_series.idxmin())
    else:
        downwind_min_deg = opposite_deg

    u_x = math.sin(math.radians(upwind_dir_deg))
    u_y = math.cos(math.radians(upwind_dir_deg))

    d_flipped = (downwind_min_deg + 180) % 360
    d_x = math.sin(math.radians(d_flipped))
    d_y = math.cos(math.radians(d_flipped))

    avg_x = u_x + d_x
    avg_y = u_y + d_y

    avg_wind_rad = math.atan2(avg_x, avg_y)
    return (math.degrees(avg_wind_rad) + 360) % 360


def generate_speed_polar_figure(df, bin_size_deg=2):
    """Generates and returns the Matplotlib figure for Streamlit rendering."""
    bins = np.arange(0, 360 + bin_size_deg, bin_size_deg)
    labels = np.radians(bins[:-1] + bin_size_deg / 2)

    df_binned = df.copy()
    df_binned["heading_bin"] = pd.cut(
        df_binned["heading_deg"],
        bins=bins,
        include_lowest=True,
        labels=labels,
    )

    polar_data = (
        df_binned.groupby("heading_bin", observed=False)["speed_kt"]
        .max()
        .fillna(0)
        .reset_index()
    )

    angles = polar_data["heading_bin"].astype(float).values
    speeds = polar_data["speed_kt"].values

    angles = np.append(angles, angles[0])
    speeds = np.append(speeds, speeds[0])

    wind_deg = infer_wind_direction_two_step(df, bin_size_deg=5)

    max_speed_idx = np.argmax(speeds[:-1])
    max_speed_deg = np.degrees(angles[max_speed_idx])
    max_speed_value = speeds[max_speed_idx]

    fig = plt.figure(figsize=(8, 8), facecolor="#f0f2f5")
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_facecolor("#e6e8fa")
    ax.grid(True, color="#888888", linestyle="--", linewidth=0.7)

    ax.plot(
        angles, speeds, color="#00aa00", linewidth=1.5, label="Max Speed (kt)"
    )

    wind_rad = math.radians(wind_deg)
    downwind_rad = math.radians((wind_deg + 180) % 360)
    max_plot_radius = max(speeds) * 1.15

    ax.plot(
        [wind_rad, downwind_rad],
        [max_plot_radius, max_plot_radius],
        color="red",
        linestyle="--",
        linewidth=2.0,
        label=f"Inferred Wind Dir ({wind_deg:.0f}°)",
    )

    max_speed_rad = math.radians(max_speed_deg)
    ax.plot(
        [0, max_speed_rad],
        [0, max_plot_radius],
        color="red",
        linestyle="-",
        linewidth=0.8,
        label=f"Max Speed Dir ({max_speed_deg:.0f}° @ {max_speed_value:.1f}kt)",
    )

    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=12, fontweight="bold")

    plt.title(
        f"Inferred Wind Direction: {wind_deg:.0f}°",
        pad=20,
        fontsize=14,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    return fig


# =====================================================================
# 6. SPEED-COLORED TRACK MAP PLOTTING (OPTIMIZED)
# =====================================================================
def generate_speed_colored_map(df, colormap_name="jet"):
    """Creates a Folium map using ColorLine for fast client-side rendering."""
    coords = list(zip(df["latitude"], df["longitude"]))
    speeds = df["speed_kt"].values

    min_speed = float(np.min(speeds))
    max_speed = float(np.max(speeds))

    if max_speed == min_speed:
        max_speed += 0.1

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
    )

    # Convert Matplotlib colors to hex strings for branca
    cmap_mpl = plt.get_cmap(colormap_name)
    colors_hex = [mcolors.to_hex(cmap_mpl(i)) for i in np.linspace(0, 1, 10)]

    # Create a branca LinearColormap object that folium.ColorLine expects
    branca_cmap = cm.LinearColormap(
        colors=colors_hex, vmin=min_speed, vmax=max_speed
    )

    # Pass the branca colormap object to ColorLine
    folium.ColorLine(
        positions=coords,
        colors=speeds,
        colormap=branca_cmap,
        weight=4,
        opacity=0.9,
    ).add_to(m)

    folium.Marker(
        location=coords[0],
        popup="Start Point",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    folium.Marker(
        location=coords[-1],
        popup="End Point",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    m.fit_bounds(
        [
            [df["latitude"].min(), df["longitude"].min()],
            [df["latitude"].max(), df["longitude"].max()],
        ]
    )

    return m, min_speed, max_speed


def generate_colormap_legend_fig(min_speed, max_speed, colormap_name="jet"):
    """Generates a horizontal colorbar legend for the map."""
    fig, ax = plt.subplots(figsize=(8, 0.8), facecolor="#f0f2f5")
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.get_cmap(colormap_name)
    norm = mcolors.Normalize(vmin=min_speed, vmax=max_speed)

    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
    )
    cb.set_label("Sailing Speed (kt)", fontsize=10, fontweight="bold")
    return fig


# =====================================================================
# 7. STREAMLIT USER INTERFACE
# =====================================================================
def main():
    st.set_page_config(
        page_title="GPX Sailing Speed Analyzer",
        page_icon="⛵",
        layout="wide",
    )

    st.title("⛵ GPX Sailing Track Analyzer")
    st.markdown(
        "Upload a `.gpx` log file to view peak speed metrics, polar speed plot, and speed-colored track map."
    )

    uploaded_file = st.sidebar.file_uploader("Choose a GPX file", type=["gpx"])

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()

            with st.spinner("Processing GPX file and calculating metrics..."):
                raw_df, metadata = load_gpx_to_dataframe(file_bytes)
                clean_df = clean_track_points(raw_df)
                metrics, dist_km, dist_nm, moving_duration = calculate_metrics(
                    clean_df
                )

            # Display Activity Name
            st.markdown(f"### 🏷️ {metadata['name']}")

            # Row 1: Date, Start Time, End Time
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            row1_col1.metric("Date", metadata["date"])
            row1_col2.metric("Start Time", metadata["start_time"])
            row1_col3.metric("End Time", metadata["end_time"])

            # Row 2: Moving Time, Elapsed Time
            row2_col1, row2_col2 = st.columns(2)
            row2_col1.metric("Moving Time", moving_duration)
            row2_col2.metric("Elapsed Time", metadata["elapsed_duration"])

            # Row 3: Total Distance
            row3_col1, row3_col2 = st.columns(2)
            row3_col1.metric("Total Distance", f"{dist_km} km")
            row3_col2.metric("", f"{dist_nm} NM")

            st.markdown("---")

            col1, col2 = st.columns([1, 1.2])

            with col1:
                st.subheader("📊 Speed Metrics")

                metrics_data = [
                    {
                        "Metric": metric,
                        "Speed (km/h)": values["kmh"],
                        "Speed (kt)": values["kt"],
                    }
                    for metric, values in metrics.items()
                ]
                metrics_df = pd.DataFrame(metrics_data)

                st.dataframe(
                    metrics_df,
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:
                st.subheader("🧭 Polar Speed Plot")
                fig = generate_speed_polar_figure(clean_df, bin_size_deg=2)
                st.pyplot(fig)

            st.markdown("---")
            st.subheader("🗺️ Track Map")

            colormap_choice = "jet"
            track_map, min_s, max_s = generate_speed_colored_map(
                clean_df, colormap_name=colormap_choice
            )

            st_folium(track_map, width="100%", height=500, returned_objects=[])

            legend_fig = generate_colormap_legend_fig(
                min_s, max_s, colormap_name=colormap_choice
            )
            st.pyplot(legend_fig)

        except Exception as e:
            st.error(f"Error parsing GPX file: {e}")
    else:
        st.info("👈 Please upload a `.gpx` file from the sidebar to begin.")


if __name__ == "__main__":
    main()
