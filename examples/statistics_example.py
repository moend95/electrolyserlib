"""
Example: Calculate and display key performance indicators (KPIs) from electrolyser simulation.
"""
import numpy as np
from electrolyserlib import Electrolyser, DynamicElectrolyser

# === Example 1: Simple Electrolyser ===
print("=" * 60)
print("Example 1: Simple Electrolyser - Statistics")
print("=" * 60)

# Create electrolyser with 1000 kW nominal power
el = Electrolyser(nominal_power=1000)

# Simulate a day with varying load (in kW)
# 24 hours with hourly values
load_profile = [
    0, 0, 0, 0, 0, 100,      # Night, early morning
    300, 500, 700, 900,       # Morning ramp-up
    1000, 1100, 1200, 1100,  # Noon peak (excess at 1100, 1200)
    1000, 900, 700, 500,      # Afternoon
    300, 200, 100, 50,        # Evening
    0, 0                      # Night
]

# Calculate H2 production
result = el.calc_h2(load_profile, unit="kW", resolution="1h")

# Get statistics
stats = el.get_statistics(result, resolution="1h")

print("\nSimulation Results:")
print(f"  Total H2 produced:        {stats['total_h2_m3']:.2f} m³")
print(f"  Total H2 produced:        {stats['total_h2_kg']:.2f} kg")
print(f"  Average efficiency:       {stats['avg_efficiency_pct']:.2f} %")
print(f"  Full load hours:          {stats['full_load_hours']:.2f} h")
print(f"  Operating hours:          {stats['operating_hours']:.2f} h")
print(f"  Total energy used:        {stats['total_energy_used_kwh']:.2f} kWh")
print(f"  Total excess energy:      {stats['total_excess_energy_kwh']:.2f} kWh")
print(f"  Energy without production:{stats['energy_without_production_kwh']:.2f} kWh")

print("\n" + "=" * 60)
print("Example 2: Dynamic Electrolyser - Statistics")
print("=" * 60)

# Create dynamic electrolyser
dyn_el = DynamicElectrolyser(
    nominal_power=1000,
    cold_startup_min=30,
    warm_startup_min=10,
    standby_limit_min=30,
    startup_power_frac=0.25
)

# Simulate 4 hours with 15-min resolution (16 time steps)
# Scenario: Start-up, operation, short pause, operation again
load_15min = [
    500, 600, 700, 800,   # First hour: startup + operation
    900, 1000, 1000, 1000, # Second hour: full operation
    50, 50, 50, 50,        # Third hour: low load (standby)
    800, 900, 1000, 1000   # Fourth hour: operation again
]

# Run simulation
result_dyn = dyn_el.simulate(load_15min, unit="kW", resolution="15min", look_ahead_min=120)

# Get statistics
stats_dyn = dyn_el.get_statistics(result_dyn, resolution="15min")

print("\nSimulation Results (Dynamic):")
print(f"  Total H2 produced:        {stats_dyn['total_h2_m3']:.2f} m³")
print(f"  Total H2 produced:        {stats_dyn['total_h2_kg']:.2f} kg")
print(f"  Average efficiency:       {stats_dyn['avg_efficiency_pct']:.2f} %")
print(f"  Full load hours:          {stats_dyn['full_load_hours']:.2f} h")
print(f"  Operating hours:          {stats_dyn['operating_hours']:.2f} h")
print(f"  Total energy used:        {stats_dyn['total_energy_used_kwh']:.2f} kWh")
print(f"  Total excess energy:      {stats_dyn['total_excess_energy_kwh']:.2f} kWh")
print(f"  Startup energy:           {stats_dyn['startup_energy_kwh']:.2f} kWh")
print(f"  Standby time steps:       {stats_dyn['standby_steps']}")
print(f"  Cold starts:              {stats_dyn['cold_starts']}")
print(f"  Warm starts:              {stats_dyn['warm_starts']}")

# Show detailed state transitions
print("\nDetailed State Timeline (first 10 steps):")
print(result_dyn[['state', 'p_in_kW', 'p_used_kW', 'h2_produced_m3']].head(10).to_string(index=False))
