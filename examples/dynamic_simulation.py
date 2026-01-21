"""
Example: Dynamic Electrolyser Simulation with Startup/Standby
"""

from electrolyserlib import DynamicElectrolyser
import numpy as np
import pandas as pd

print("=" * 70)
print("Dynamic Electrolyser - Startup and Standby Simulation")
print("=" * 70)

# Create dynamic electrolyser
dyn_el = DynamicElectrolyser(
    nominal_power=1000,  # kW
    cold_startup_min=30,  # Cold start takes 30 minutes
    warm_startup_min=10,  # Warm start takes 10 minutes
    standby_limit_min=30,  # Stays warm for 30 minutes
    startup_power_frac=0.25  # Uses 25% power during startup
)

print("\nElectrolyser Configuration:")
print(f"  Nominal Power: {dyn_el.nominal_power} kW")
print(f"  Cold Start Time: {dyn_el.cold_startup_min} min")
print(f"  Warm Start Time: {dyn_el.warm_startup_min} min")
print(f"  Standby Limit: {dyn_el.standby_limit_min} min")
print(f"  Startup Power: {dyn_el.startup_power_frac * 100}%")

# Create realistic power timeseries (15-min resolution, 8 hours)
np.random.seed(42)

# Simulation of a typical daily pattern with wind/solar
hours = 8
intervals_per_hour = 4  # 15-min intervals
total_intervals = hours * intervals_per_hour

# Create variable power with phases
power_profile = []
for h in range(hours):
    if h < 2:
        # First 2 hours: Low power (cold)
        power_profile.extend([50, 80, 100, 120])
    elif h < 4:
        # Hours 2-4: Power increases (startup)
        base = 200 + (h - 2) * 200
        power_profile.extend([base + np.random.randint(0, 100) for _ in range(4)])
    elif h < 6:
        # Hours 4-6: Full power
        power_profile.extend([900 + np.random.randint(0, 200) for _ in range(4)])
    else:
        # Hours 6-8: Power decreases (into standby)
        base = 800 - (h - 6) * 300
        power_profile.extend([max(50, base + np.random.randint(-100, 50)) for _ in range(4)])

# Execute simulation
print("\n" + "-" * 70)
print("Starting Simulation...")
print("-" * 70)

results = dyn_el.simulate(
    timeseries=power_profile,
    unit="kW",
    resolution="15min",
    look_ahead_min=120  # Look ahead 2 hours
)

# Calculate statistics
print("\nSimulation Results:")
print(f"  Total Duration: {len(results) * 15 / 60:.1f} hours ({len(results)} intervals)")
print(f"  Total H2 Production: {results['h2_produced_m3'].sum():.2f} m³")
print(f"  Average Production: {results['h2_produced_m3'].mean():.3f} m³ per 15min")
print(f"  Total Energy Used: {(results['p_used_kW'] * 0.25).sum():.2f} kWh")
print(f"  Excess Energy: {(results['p_excess_kW'] * 0.25).sum():.2f} kWh")

# State statistics
state_counts = results['state'].value_counts()
print("\nState Distribution:")
for state, count in state_counts.items():
    percentage = (count / len(results)) * 100
    time_hours = (count * 15) / 60
    print(f"  {state:20s}: {count:3d} intervals ({percentage:5.1f}%, {time_hours:.1f}h)")

# Find state transitions
transitions = results[results['state'] != results['state_prev']]
print(f"\nNumber of State Transitions: {len(transitions)}")

# Show important events
print("\nImportant Events:")
print("-" * 70)

for idx, row in transitions.iterrows():
    time_h = (idx * 15) / 60
    print(f"  t={time_h:5.2f}h: {row['state_prev']:15s} -> {row['state']:15s} "
          f"(P_in={row['p_in_kW']:6.1f} kW)")

# Detailed timeseries for first 2 hours
print("\n" + "=" * 70)
print("Detailed Analysis - First 2 Hours (Cold Start Phase)")
print("=" * 70)

detail_df = results.head(8).copy()
detail_df['time_h'] = [i * 0.25 for i in range(len(detail_df))]

print("\n{:>6s}  {:>15s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format(
    "Time", "State", "P_in", "P_used", "P_excess", "H2"
))
print("-" * 70)

for _, row in detail_df.iterrows():
    print("{:5.2f}h  {:>15s}  {:7.1f}  {:7.1f}  {:7.1f}  {:7.3f}".format(
        row['time_h'],
        row['state'],
        row['p_in_kW'],
        row['p_used_kW'],
        row['p_excess_kW'],
        row['h2_produced_m3']
    ))

# Production analysis
print("\n" + "=" * 70)
print("Production Efficiency")
print("=" * 70)

# Only time points with production
productive_intervals = results[results['h2_produced_m3'] > 0]
non_productive_intervals = results[results['h2_produced_m3'] == 0]

print(f"\nProductive Intervals: {len(productive_intervals)} ({len(productive_intervals)/len(results)*100:.1f}%)")
print(f"Non-Productive Intervals: {len(non_productive_intervals)} ({len(non_productive_intervals)/len(results)*100:.1f}%)")

if len(productive_intervals) > 0:
    avg_production = productive_intervals['h2_produced_m3'].mean()
    print(f"Average Production (when active): {avg_production:.3f} m³ per 15min")
    print(f"Average Power (when active): {productive_intervals['p_used_kW'].mean():.1f} kW")

# Startup analysis
startup_intervals = results[results['state'] == 'STARTUP']
if len(startup_intervals) > 0:
    startup_energy = (startup_intervals['p_used_kW'] * 0.25).sum()
    print(f"\nTotal Startup Energy: {startup_energy:.2f} kWh")
    print(f"Total Startup Time: {len(startup_intervals) * 15} min")

print("\n" + "=" * 70)
print("Simulation completed!")
print("=" * 70)

# Optional: Save results as CSV
# results.to_csv('simulation_results.csv', index=False)
# print("\nResults saved to 'simulation_results.csv'")
