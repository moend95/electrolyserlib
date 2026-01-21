"""
Example: Simple H2 production calculation with the Electrolyser model
"""

from electrolyserlib import Electrolyser
import numpy as np

# Create an electrolyser with 1000 kW nominal power
electrolyser = Electrolyser(nominal_power=1000)

print("=" * 60)
print("Example 1: Hourly Hydrogen Production")
print("=" * 60)

# Power timeseries for one day (24 hours)
# Simulation of variable wind energy
np.random.seed(42)
power_input_kw = np.random.uniform(300, 1200, 24)

# Calculate H2 production
results = electrolyser.calc_h2(
    timeseries=power_input_kw,
    unit="kW",
    resolution="1h"
)

# Show results
print(f"\nInput data: {len(power_input_kw)} hours")
print(f"Average input power: {power_input_kw.mean():.1f} kW")
print(f"\nResults:")
print(f"  Total H2 production: {results['h2_produced_m3'].sum():.2f} m³")
print(f"  Average hourly production: {results['h2_produced_m3'].mean():.2f} m³/h")
print(f"  Maximum production: {results['h2_produced_m3'].max():.2f} m³/h")
print(f"  Total energy used: {results['used_power_kW'].sum():.2f} kWh")
print(f"  Excess energy: {results['excess_power_kW'].sum():.2f} kWh")

# Show first 5 hours in detail
print("\nFirst 5 hours in detail:")
print(results.head().to_string(index=False))

# Calculate efficiency
total_h2_m3 = results['h2_produced_m3'].sum()
total_energy_kwh = results['used_power_kW'].sum()
if total_h2_m3 > 0:
    avg_specific_consumption = total_energy_kwh / total_h2_m3
    print(f"\nAverage specific energy consumption: {avg_specific_consumption:.3f} kWh/m³")

print("\n" + "=" * 60)
print("Example 2: High-Resolution 15-Minute Simulation")
print("=" * 60)

# 15-minute resolution for 6 hours (24 values)
power_15min = [200, 350, 500, 750, 900, 1000, 1100, 950, 
               800, 600, 400, 300, 250, 400, 600, 800,
               950, 1000, 980, 850, 700, 550, 400, 300]

results_15min = electrolyser.calc_h2(
    timeseries=power_15min,
    unit="kW",
    resolution="15min"
)

print(f"\nInput data: {len(power_15min)} intervals of 15 minutes (6 hours)")
print(f"Total H2 production: {results_15min['h2_produced_m3'].sum():.2f} m³")
print(f"Average production: {results_15min['h2_produced_m3'].mean():.2f} m³ per 15min")

print("\n" + "=" * 60)
print("Example 3: Overload Scenario")
print("=" * 60)

# Test with power above nominal load
power_overload = [1200, 1500, 1800, 1100, 900]

results_overload = electrolyser.calc_h2(
    timeseries=power_overload,
    unit="kW",
    resolution="1h"
)

print("\nPower above nominal load (1000 kW):")
for i, row in results_overload.iterrows():
    print(f"  Hour {i+1}: Input={row['input_power_raw_kW']:.0f} kW, "
          f"Used={row['used_power_kW']:.0f} kW, "
          f"Excess={row['excess_power_kW']:.0f} kW, "
          f"H2={row['h2_produced_m3']:.2f} m³")

print("\n" + "=" * 60)
print("Example 4: Different Power Units")
print("=" * 60)

# Same timeseries in different units
power_values = [800000, 900000, 1000000]  # in Watts

# In Watts
results_w = electrolyser.calc_h2(power_values, unit="W", resolution="1h")
print(f"\nInput in Watts: {results_w['h2_produced_m3'].sum():.2f} m³")

# In Kilowatts
power_values_kw = [800, 900, 1000]
results_kw = electrolyser.calc_h2(power_values_kw, unit="kW", resolution="1h")
print(f"Input in kW:    {results_kw['h2_produced_m3'].sum():.2f} m³")

# In Megawatts
power_values_mw = [0.8, 0.9, 1.0]
results_mw = electrolyser.calc_h2(power_values_mw, unit="MW", resolution="1h")
print(f"Input in MW:    {results_mw['h2_produced_m3'].sum():.2f} m³")

print("\nAll three calculations should be identical!")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)

print("\n" + "=" * 60)
print("Example 5: Wind Power Data (24 Hours, 15-min Resolution)")
print("=" * 60)

# Example power data from wind turbine (kW)
# 15-minute intervals over 24 hours
real_power_data = [
    9.096066519723282, 19.564953283906075, 23.662994663571826, 3.5342807157466334,
    0.0, 0.0, 0.1970105604824339, 10.208374383123127,
    3.5342807157466334, 19.564953283906075, 21.73327938992974, 12.432942224112255,
    0.1970105604824339, 19.564953283906075, 12.432942224112248, 7.9837426943865735,
    0.0, 15.332397165430265, 21.73327938992974, 0.0,
    0.1970105604824339, 3.5342807157466334, 17.41572361651316, 10.208374383123127,
    11.320666284586116, 15.332397165430265, 1.3094495842914395, 0.0,
    17.41572361651316, 19.564953283906075, 21.73327938992974, 31.690399827124654,
    66.48231347771947, 81.64216576665963, 47.64734048457914, 51.42473347396749,
    93.02966442453588, 0.0, 0.0, 0.0,
    0.0, 3.5342807157466334, 17.41572361651316, 39.535601339777386,
    58.964994559743666, 43.9393006193307, 74.00888083080864, 58.964994559743666,
    147.4743808087102, 122.47917500458833, 43.9393006193307, 39.535601339777386,
    37.404216227583966, 27.796982320479756, 33.66431632350283, 97.7970140625708,
    166.46738543569242, 147.4743808087102, 134.97246604626002, 188.3252853450364,
    172.86535677734526, 134.97246604626002, 147.4743808087102, 97.7970140625708,
    74.00888083080864, 97.7970140625708, 153.74652476640512, 160.1056027963119,
    256.5924664967022, 239.41645043679313, 141.2109794119665, 160.1056027963119,
    134.97246604626002, 89.24098295052688, 97.7970140625708, 293.10676864071587,
    166.46738543569242, 74.00888083080864, 172.86535677734526, 160.1056027963119,
    147.4743808087102, 51.42473347396749, 160.1056027963119, 273.97495074895045,
    256.5924664967022, 239.41645043679313, 213.70673330934628, 239.41645043679313,
    205.2020180752704, 239.41645043679313, 248.00072795819915, 479.7577431701312,
    360.15150686905054, 273.97495074895045, 205.2020180752704, 188.3252853450364,
    147.4743808087102, 222.2691582885965, 273.97495074895045
]

# Time labels (15-min intervals)
time_labels = []
for hour in range(25):  # 0-24
    for minute in [0, 15, 30, 45]:
        if hour == 24 and minute > 0:
            break
        time_labels.append(f"{hour:02d}:{minute:02d}")

# Calculate H2 production with 500 kW electrolyser
electrolyser_500 = Electrolyser(nominal_power=500)
results_wind = electrolyser_500.calc_h2(
    timeseries=real_power_data,
    unit="kW",
    resolution="15min"
)

print(f"\nData: 24 hours, 15-minute resolution ({len(real_power_data)} data points)")
print(f"Electrolyser: 500 kW nominal power")
print(f"\nPower Statistics:")
print(f"  Average: {sum(real_power_data)/len(real_power_data):.2f} kW")
print(f"  Maximum: {max(real_power_data):.2f} kW")
print(f"  Minimum: {min(real_power_data):.2f} kW")

print(f"\nH2 Production Results:")
print(f"  Total H2 Production: {results_wind['h2_produced_m3'].sum():.2f} m³")
print(f"  Total Energy Used: {results_wind['used_power_kW'].sum() * 0.25:.2f} kWh")
print(f"  Total Excess Energy: {results_wind['excess_power_kW'].sum() * 0.25:.2f} kWh")
print(f"  Capacity Factor: {(results_wind['used_power_kW'].sum() * 0.25) / (500 * 24) * 100:.1f}%")

# Show peak production hours
print("\nTop 5 Production Intervals:")
top_5 = results_wind.nlargest(5, 'h2_produced_m3')
for idx, row in top_5.iterrows():
    time_label = time_labels[idx]
    print(f"  {time_label}: P_in={row['input_power_raw_kW']:6.2f} kW, "
          f"P_used={row['used_power_kW']:6.2f} kW, "
          f"H2={row['h2_produced_m3']:.3f} m³")

# Show periods with no production
no_production = results_wind[results_wind['h2_produced_m3'] == 0]
print(f"\nIntervals with no production: {len(no_production)}/{len(results_wind)} "
      f"({len(no_production)/len(results_wind)*100:.1f}%)")

print("\n" + "=" * 60)
print("All examples completed!")
print("=" * 60)
