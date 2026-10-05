import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_parquet("data/software-data.parquet")
df_filled = df.ffill()
print(df_filled)

speed_m_s = (
	df_filled["SME_TRQSPD_Speed"]
	* (12 / 41)
	* (2 * np.pi / 60)
	* 0.2
)
speed_at_10 = np.interp(10.0, df_filled["Time"], speed_m_s)
print(f"Car speed at 10 s: {speed_at_10:.2f} m/s ({speed_at_10 * 3.6:.2f} km/h)")

plt.plot(df_filled["Time"], speed_m_s, label="Car speed")
plt.axvline(10, color="gray", linestyle="--", linewidth=1)
plt.scatter([10], [speed_at_10], color="red", zorder=3, label=f"10 s: {speed_at_10:.2f} m/s")
plt.title("Car Speed vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

time_s = df_filled["Time"].to_numpy()
pedal_travel = df_filled["ETC_STATUS_PEDAL_TRAVEL"].to_numpy()
torque_demand = df_filled["SME_THROTL_TorqueDemand"].to_numpy()
brake_voltage = df_filled["ETC_STATUS_BRAKE_SENSE_VOLTAGE"].to_numpy()

throttle_applied = (pedal_travel > 0) | (torque_demand > 0)
brake_applied = brake_voltage > 360
vehicle_moving = speed_m_s.to_numpy() > 0

# Give braking precedence if both controls are reported at the same time.
braking = brake_applied & vehicle_moving
accelerating = throttle_applied & ~braking
coasting = vehicle_moving & ~throttle_applied & ~braking


def state_intervals(state):
	changes = np.diff(np.r_[False, state, False].astype(int))
	starts = np.flatnonzero(changes == 1)
	ends = np.flatnonzero(changes == -1) - 1
	intervals = []
	for start, end in zip(starts, ends):
		if intervals and time_s[start] - intervals[-1][1] <= 0.25:
			intervals[-1] = (intervals[-1][0], time_s[end])
		else:
			intervals.append((time_s[start], time_s[end]))
	return [interval for interval in intervals if interval[1] - interval[0] >= 0.5]


states = [
	("Accelerating", accelerating, "#d97706"),
	("Braking", braking, "#c2413b"),
	("Coasting", coasting, "#16827a"),
]

for state_name, state_mask, color in states:
	intervals = state_intervals(state_mask)
	print(f"\n{state_name} time ranges:")
	if intervals:
		for start, end in intervals:
			print(f"  {start:.2f}-{end:.2f} s ({end - start:.2f} s)")
	else:
		print("  No intervals found")

	plot_mask = np.zeros_like(state_mask, dtype=bool)
	for start, end in intervals:
		plot_mask |= (time_s >= start) & (time_s <= end)

	fig, ax = plt.subplots(figsize=(11, 4))
	ax.plot(time_s, speed_m_s, color="#aeb8bd", linewidth=1, label="Vehicle speed")
	ax.plot(
		time_s,
		np.where(plot_mask, speed_m_s, np.nan),
		color=color,
		linewidth=1.8,
		label=state_name,
	)
	for start, end in intervals:
		ax.axvspan(start, end, color=color, alpha=0.12)
	ax.set_title(f"{state_name} vs. Time")
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Speed (m/s)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()

plt.show()

gps = (
	df.dropna(subset=["VDM_GPS_Latitude", "VDM_GPS_Longitude"])
	.sort_values("Time")
	.reset_index(drop=True)
)
gps_time = gps["Time"].to_numpy()
gps_latitude = gps["VDM_GPS_Latitude"].to_numpy()
gps_longitude = gps["VDM_GPS_Longitude"].to_numpy()
gps_speed_mph = gps["VDM_GPS_SPEED"].fillna(0).to_numpy()
moving_fixes = gps_speed_mph > 1
first_moving_index = np.flatnonzero(moving_fixes)[0]

latitude_reference = np.deg2rad(gps_latitude[first_moving_index])
east_m = (gps_longitude - gps_longitude[first_moving_index]) * 111320 * np.cos(latitude_reference)
north_m = (gps_latitude - gps_latitude[first_moving_index]) * 111320
distance_from_start_m = np.hypot(east_m, north_m)
path_distance_m = np.r_[0, np.cumsum(np.hypot(np.diff(east_m), np.diff(north_m)))]

lap_start_time = gps_time[first_moving_index]
return_indices = np.flatnonzero(
	moving_fixes
	& (gps_time >= lap_start_time + 30)
	& (distance_from_start_m <= 8)
)

return_clusters = []
for index in return_indices:
	if return_clusters and gps_time[index] - gps_time[return_clusters[-1][-1]] <= 5:
		return_clusters[-1].append(index)
	else:
		return_clusters.append([index])

lap_finishes = []
first_lap_distance_m = None
last_lap_distance_m = path_distance_m[first_moving_index]
for cluster in return_clusters:
	closest_return = min(cluster, key=lambda index: distance_from_start_m[index])
	distance_since_lap_m = path_distance_m[closest_return] - last_lap_distance_m
	if first_lap_distance_m is None or distance_since_lap_m >= first_lap_distance_m * 0.75:
		lap_finishes.append(closest_return)
		if first_lap_distance_m is None:
			first_lap_distance_m = distance_since_lap_m
		last_lap_distance_m = path_distance_m[closest_return]

print(f"\nCompleted laps: {len(lap_finishes)}")
lap_ranges = list(zip(
	[gps_time[first_moving_index]] + [gps_time[index] for index in lap_finishes[:-1]],
	[gps_time[index] for index in lap_finishes],
))
for lap_number, (start_time, end_time) in enumerate(lap_ranges, start=1):
	print(f"  Lap {lap_number}: {start_time:.2f}-{end_time:.2f} s")

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(
	east_m,
	north_m,
	color="#cbd5e1",
	linewidth=1,
	label="Full GPS track",
)
lap_colors = plt.get_cmap("tab10")
for lap_number, (start_time, end_time) in enumerate(lap_ranges, start=1):
	lap_gps_mask = (gps_time >= start_time) & (gps_time <= end_time)
	lap_color = lap_colors((lap_number - 1) % 10)
	ax.plot(
		east_m[lap_gps_mask],
		north_m[lap_gps_mask],
		color=lap_color,
		linewidth=2,
		label=f"Lap {lap_number}: {end_time - start_time:.2f} s",
	)
	start_index = np.argmin(np.abs(gps_time - start_time))
	finish_index = lap_finishes[lap_number - 1]
	ax.scatter(
		east_m[start_index],
		north_m[start_index],
		color=lap_color,
		marker="o",
		s=45,
		zorder=3,
		label="Lap start" if lap_number == 1 else None,
	)
	ax.scatter(
		east_m[finish_index],
		north_m[finish_index],
		color=lap_color,
		marker="X",
		s=55,
		zorder=3,
		label="Lap finish" if lap_number == 1 else None,
	)
	ax.annotate(
		str(lap_number),
		(east_m[start_index], north_m[start_index]),
		xytext=(5, 5),
		textcoords="offset points",
		color=lap_color,
		fontweight="bold",
	)

ax.set_title("GPS Track by Lap")
ax.set_xlabel("East from lap start (m)")
ax.set_ylabel("North from lap start (m)")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()

sample_end_time = np.r_[time_s[1:], time_s[-1]]
for lap_number, (start_time, end_time) in enumerate(lap_ranges, start=1):
	lap_mask = (time_s < end_time) & (sample_end_time > start_time)
	lap_duration_per_sample = np.maximum(
		0,
		np.minimum(sample_end_time, end_time) - np.maximum(time_s, start_time),
	)
	peak_speed_index = np.flatnonzero(lap_mask)[np.argmax(speed_m_s.to_numpy()[lap_mask])]
	peak_speed = speed_m_s.iloc[peak_speed_index]

	acceleration_window_s = 1.0
	window_start_times = time_s[(time_s >= start_time) & (time_s + acceleration_window_s <= end_time)]
	window_start_speeds = np.interp(window_start_times, time_s, speed_m_s)
	window_end_speeds = np.interp(window_start_times + acceleration_window_s, time_s, speed_m_s)
	average_acceleration = (window_end_speeds - window_start_speeds) / acceleration_window_s
	peak_acceleration_index = np.argmax(average_acceleration)
	peak_acceleration = average_acceleration[peak_acceleration_index]
	peak_acceleration_time = window_start_times[peak_acceleration_index]

	accelerating_time = np.sum(lap_duration_per_sample * accelerating)
	coasting_time = np.sum(lap_duration_per_sample * coasting)
	imu_acceleration = df_filled["VDM_X_AXIS_ACCELERATION"].to_numpy() * 9.81
	peak_imu_acceleration = np.max(imu_acceleration[lap_mask])

	print(f"\nLap {lap_number} metrics:")
	print(f"  Maximum speed: {peak_speed:.2f} m/s ({peak_speed * 3.6:.2f} km/h)")
	print(
		f"  Maximum 1-second average acceleration: {peak_acceleration:.2f} m/s^2 "
		f"(starting at {peak_acceleration_time:.2f} s)"
	)
	print(f"  Accelerating time: {accelerating_time:.2f} s")
	print(f"  Coasting time: {coasting_time:.2f} s")
	print(f"  Peak VDM X-axis acceleration (cross-check): {peak_imu_acceleration:.2f} m/s^2")