import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# -----------------------------
# 1. Load and label datasets
# -----------------------------
df_jam = pd.read_csv("./Dataset/OBD-II-Dataset/2018-03-29_Seat_Leon_KA_RT_Stau.csv")
df_normal = pd.read_csv("./Dataset/OBD-II-Dataset/2018-04-23_Seat_Leon_RT_KA_Normal.csv")
df_free = pd.read_csv("./Dataset/OBD-II-Dataset/2018-04-23_Seat_Leon_KA_KA_Frei.csv")

df_jam['traffic_state'] = 'traffic_jam'
df_normal['traffic_state'] = 'normal_traffic'
df_free['traffic_state'] = 'traffic_free'

df_all = pd.concat([df_jam, df_normal, df_free], ignore_index=True)

# -----------------------------
# 2. Convert time to seconds & fill missing values
# -----------------------------
def time_to_seconds(t):
    # handle possible formats safely
    if isinstance(t, str) and ':' in t:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        return float(t)

df_all['Time_sec'] = df_all['Time'].apply(time_to_seconds)
df_all.ffill(inplace=True)  # forward fill missing values

# -----------------------------
# 3. Compute adaptive sudden thresholds (time-aware)
# -----------------------------
# compute per-row dt in seconds (avoid zeros)
dt = df_all['Time_sec'].diff().replace(0, np.nan).fillna(method='bfill').fillna(1.0)

# accelerator pedal raw diff and rate (pedal % per second)
accel_diff_all = df_all['Accelerator Pedal Position D [%]'].diff()
accel_rate_all = accel_diff_all / dt  # units: % pedal / s

# speed acceleration (m/s^2): convert km/h -> m/s by /3.6, then diff/dt
speed_diff = df_all['Vehicle Speed Sensor [km/h]'].diff()
speed_accel_all = (speed_diff / 3.6) / dt  # m/s^2; negative = deceleration

# remove tiny near-zero noise
eps = 1e-3
accel_rate_nonzero = accel_rate_all[accel_rate_all.abs() > eps].dropna()
speed_accel_nonzero = speed_accel_all[speed_accel_all.abs() > eps].dropna()

# compute percentiles with sensible fallbacks
if len(accel_rate_nonzero) >= 10:
    ACCEL_RATE_THRESH = np.percentile(accel_rate_nonzero, 95)    # positive -> sudden accel
else:
    ACCEL_RATE_THRESH = 5.0   # default: 5 % pedal / s (tweak if needed)

if len(speed_accel_nonzero) >= 10:
    BRAKE_DECEL_THRESH = np.percentile(speed_accel_nonzero, 5)   # negative large -> sudden braking
else:
    BRAKE_DECEL_THRESH = -1.5  # default: -1.5 m/s^2 (~moderate braking)

print(f"Adaptive accel-rate threshold (pedal %/s): {ACCEL_RATE_THRESH:.3f}")
print(f"Adaptive braking threshold (m/s^2): {BRAKE_DECEL_THRESH:.3f}")

# Optional diagnostics
try:
    print("\naccel_rate_nonzero.describe():\n", accel_rate_nonzero.describe())
    print("\nspeed_accel_nonzero.describe():\n", speed_accel_nonzero.describe())
except Exception:
    pass

# -----------------------------
# 4. Sliding windows & feature extraction
# -----------------------------
window_size = 15.0
step_size = 7.5
max_time = df_all['Time_sec'].max()
windows = []

start = df_all['Time_sec'].min()
while start + window_size <= max_time:
    end = start + window_size
    win = df_all[(df_all['Time_sec'] >= start) & (df_all['Time_sec'] < end)]
    if not win.empty:
        windows.append(win)
    start += step_size

features = []
for win in windows:
    f = {}
    # basic stats
    f['rpm_mean'] = win['Engine RPM [RPM]'].mean()
    f['rpm_std'] = win['Engine RPM [RPM]'].std()
    f['speed_mean'] = win['Vehicle Speed Sensor [km/h]'].mean()
    f['speed_std'] = win['Vehicle Speed Sensor [km/h]'].std()
    f['throttle_mean'] = win['Absolute Throttle Position [%]'].mean()
    f['throttle_std'] = win['Absolute Throttle Position [%]'].std()
    f['accel_mean'] = win['Accelerator Pedal Position D [%]'].mean()
    f['accel_std'] = win['Accelerator Pedal Position D [%]'].std()
    f['engine_load_mean'] = win['Intake Manifold Absolute Pressure [kPa]'].mean()
    f['maf_mean'] = win['Air Flow Rate from Mass Flow Sensor [g/s]'].mean()

    # traffic state
    f['traffic_state'] = win['traffic_state'].iloc[0] if 'traffic_state' in win.columns else 'unknown'

    # compute window-wise dt and rates
    dt_win = win['Time_sec'].diff().replace(0, np.nan).fillna(method='bfill').fillna(1.0)
    accel_diff_win = win['Accelerator Pedal Position D [%]'].diff()
    accel_rate_win = accel_diff_win / dt_win

    speed_diff_win = win['Vehicle Speed Sensor [km/h]'].diff()
    speed_accel_win = (speed_diff_win / 3.6) / dt_win  # m/s^2

    # sudden acceleration (accelerator pedal rate) and sudden braking (speed deceleration)
    f['sudden_accel'] = accel_rate_win.gt(ACCEL_RATE_THRESH).any()
    f['sudden_brake'] = speed_accel_win.lt(BRAKE_DECEL_THRESH).any()

    # fuel efficiency indicator (lower = more eco-friendly)
    f['fuel_efficiency_indicator'] = (f['rpm_mean'] * (f['throttle_mean'] if not np.isnan(f['throttle_mean']) else 0)) / (f['maf_mean'] + 1e-3)

    features.append(f)

features_df = pd.DataFrame(features)

# if any NaNs in indicators, fill with reasonable defaults
features_df['fuel_efficiency_indicator'] = features_df['fuel_efficiency_indicator'].fillna(features_df['fuel_efficiency_indicator'].median())
features_df['rpm_std'] = features_df['rpm_std'].fillna(0)
features_df['speed_std'] = features_df['speed_std'].fillna(0)
features_df['throttle_std'] = features_df['throttle_std'].fillna(0)

# -----------------------------
# 5. Scale & KMeans clustering for driving style
# -----------------------------
cluster_features = ['rpm_mean','rpm_std','speed_mean','speed_std','throttle_mean','throttle_std']
X_scaled = StandardScaler().fit_transform(features_df[cluster_features])
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
features_df['cluster'] = kmeans.fit_predict(X_scaled)

# -----------------------------
# 6. Map clusters to driving style using weighted score (RELATIVE ranking)
# -----------------------------
cluster_stats = features_df.groupby('cluster')[cluster_features].mean()
cluster_stats_scaled = MinMaxScaler().fit_transform(cluster_stats)
cluster_stats_scaled_df = pd.DataFrame(cluster_stats_scaled, columns=cluster_stats.columns, index=cluster_stats.index)

cluster_scores = {}
for cluster_id, row in cluster_stats_scaled_df.iterrows():
    score = (0.4 * row['rpm_mean'] + 0.2 * row['rpm_std'] +
             0.2 * row['speed_mean'] + 0.1 * row['speed_std'] +
             0.05 * row['throttle_mean'] + 0.05 * row['throttle_std'])
    cluster_scores[cluster_id] = score

# RELATIVE ranking -> always produce calm/normal/aggressive
scores = pd.Series(cluster_scores)
ranked = scores.rank(method="dense", ascending=True).astype(int)  # 1 = lowest, 3 = highest

cluster_labels = {}
for cluster_id in scores.index:
    if ranked[cluster_id] == 1:
        cluster_labels[cluster_id] = 'calm'
    elif ranked[cluster_id] == 2:
        cluster_labels[cluster_id] = 'normal'
    else:
        cluster_labels[cluster_id] = 'aggressive'

features_df['driving_style'] = features_df['cluster'].map(cluster_labels)

print("\nCluster scores:", cluster_scores)
print("Cluster labels (relative):", cluster_labels)

# -----------------------------
# 7. Eco-driving detection
# -----------------------------
eco_threshold = np.percentile(features_df['fuel_efficiency_indicator'].dropna(), 33)
features_df['eco_driving'] = features_df['fuel_efficiency_indicator'] <= eco_threshold

# -----------------------------
# 8. Overall metrics
# -----------------------------
total_windows = len(features_df)
print(f"\nOverall sudden acceleration: {features_df['sudden_accel'].sum() / total_windows * 100:.2f}%")
print(f"Overall sudden braking: {features_df['sudden_brake'].sum() / total_windows * 100:.2f}%")
print(f"Overall eco-friendly driving: {features_df['eco_driving'].mean() * 100:.2f}%")

# -----------------------------
# 9. Prepare data for visualization
# -----------------------------
# sudden events per traffic state & driving style
sudden_data = []
for state in features_df['traffic_state'].unique():
    subset_state = features_df[features_df['traffic_state'] == state]
    for style in subset_state['driving_style'].unique():
        subset = subset_state[subset_state['driving_style'] == style]
        sudden_data.append({
            'traffic_state': state,
            'driving_style': style,
            'Sudden Accel %': subset['sudden_accel'].mean() * 100,
            'Sudden Brake %': subset['sudden_brake'].mean() * 100
        })
sudden_df = pd.DataFrame(sudden_data)

# eco-driving per traffic state & driving style
eco_data = []
for state in features_df['traffic_state'].unique():
    subset_state = features_df[features_df['traffic_state'] == state]
    for style in subset_state['driving_style'].unique():
        subset = subset_state[subset_state['driving_style'] == style]
        eco_data.append({
            'traffic_state': state,
            'driving_style': style,
            'Eco-driving %': subset['eco_driving'].mean() * 100
        })
eco_df = pd.DataFrame(eco_data)

# -----------------------------
# 10. Dashboard visualization (3 subplots)
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 18))
plt.subplots_adjust(hspace=0.4)

# 1) Driving style distribution
sns.countplot(data=features_df, x='traffic_state', hue='driving_style', ax=axes[0])
axes[0].set_title("Driving Style Distribution per Traffic State")
axes[0].set_ylabel("Number of Windows")
axes[0].set_xlabel("Traffic State")
axes[0].legend(title="Driving Style")

# 2) Sudden accel / brake
if not sudden_df.empty:
    sudden_melt = sudden_df.melt(id_vars=['traffic_state','driving_style'],
                                 value_vars=['Sudden Accel %','Sudden Brake %'],
                                 var_name='Event', value_name='Percentage')
    sns.barplot(data=sudden_melt, x='traffic_state', y='Percentage', hue='driving_style', ax=axes[1])
axes[1].set_title("Sudden Acceleration/Braking % per Traffic State & Driving Style")
axes[1].set_ylabel("Percentage (%)")
axes[1].set_xlabel("Traffic State")
axes[1].legend(title="Driving Style")

# 3) Eco-driving %
if not eco_df.empty:
    sns.barplot(data=eco_df, x='traffic_state', y='Eco-driving %', hue='driving_style', ax=axes[2])
axes[2].set_title("Eco-driving % per Traffic State & Driving Style")
axes[2].set_ylabel("Percentage (%)")
axes[2].set_xlabel("Traffic State")
axes[2].legend(title="Driving Style")

plt.show()
