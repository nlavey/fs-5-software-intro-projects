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