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