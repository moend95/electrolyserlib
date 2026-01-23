# ElectrolyserLib

A Python library for calculating hydrogen production based on PEM (Proton Exchange Membrane) electrolyser models.

## Features

- **Standard Electrolyser Model**: Calculate H2 production based on efficiency curves and power input
- **Dynamic Electrolyser Model**: Advanced simulation with startup/standby states and dynamic behavior
- **Flexible Input**: Support for different power units (W, kW, MW) and time resolutions (1min, 5min, 15min, 1h)
- **Custom Efficiency Curves**: Use default PEM curves or provide your own CSV data
- **Production Optimization**: Automatic handling of minimum load, excess power, and operational states

## Installation

```bash
pip install electrolyserlib
```

## Quick Start

### Basic Electrolyser

```python
from electrolyserlib import Electrolyser
import numpy as np

# Create electrolyser with 1000 kW nominal power
electrolyser = Electrolyser(nominal_power=1000)

# Simulate with power timeseries
power_input = [800, 900, 1000, 950, 700]  # in kW
results = electrolyser.calc_h2(power_input, unit="kW", resolution="1h")

print(results)
# Output:
#    input_power_raw_kW  used_power_kW  excess_power_kW  h2_produced_m3
# 0                 800            800              0.0       168.99...
# 1                 900            900              0.0       190.12...
# 2                1000           1000              0.0       211.24...
# ...
```

### Dynamic Electrolyser with Startup/Standby

```python
from electrolyserlib import DynamicElectrolyser

# Create dynamic electrolyser
dyn_el = DynamicElectrolyser(
    nominal_power=1000,  # kW
    cold_startup_min=30,  # Cold startup takes 30 minutes
    warm_startup_min=10,  # Warm startup takes 10 minutes
    standby_limit_min=30,  # Stays warm for 30 minutes
    startup_power_frac=0.25  # Uses 25% power during startup
)

# Simulate with variable power
power_series = np.random.uniform(0, 1200, 100)  # 100 timesteps
results = dyn_el.simulate(
    power_series, 
    unit="kW", 
    resolution="15min",
    look_ahead_min=120  # Look ahead 2 hours for startup decisions
)

print(results.head())
# Output includes state transitions, power usage, and H2 production
```

## Custom Efficiency Curve

You can provide your own efficiency curve as a CSV file:

```python
# CSV format (semicolon-separated, comma as decimal):
# relative load [%];specific energy consumption [kWh/m3]
# 10;6.737
# 20;5.091
# 30;4.782
# ...

electrolyser = Electrolyser(
    csv_data="path/to/your/efficiency_curve.csv",
    nominal_power=500
)
```

Or as a string:

```python
csv_string = """relative load [%];specific energy consumption [kWh/m3]
10;6.737
20;5.091
30;4.782
40;4.677
50;4.650
60;4.656
70;4.685
80;4.734
90;4.798
100;4.871"""

electrolyser = Electrolyser(csv_data=csv_string, nominal_power=500)
```

## API Reference

### `Electrolyser`

**Constructor:**
```python
Electrolyser(csv_data=None, nominal_power=None)
```
- `csv_data`: Path to CSV file or CSV string with efficiency curve (optional, uses default if None)
- `nominal_power`: Nominal power in kW (required)

**Methods:**
- `calc_h2(timeseries, unit="kW", resolution="1h")`: Calculate H2 production
  - Returns: DataFrame with columns `input_power_raw_kW`, `used_power_kW`, `excess_power_kW`, `h2_produced_m3`

### `DynamicElectrolyser`

Inherits from `Electrolyser` with additional dynamic behavior.

**Constructor:**
```python
DynamicElectrolyser(
    csv_data=None, 
    nominal_power=None,
    cold_startup_min=30, 
    warm_startup_min=10,
    standby_limit_min=30, 
    startup_power_frac=0.25
)
```

**Methods:**
- `simulate(timeseries, unit="kW", resolution="15min", look_ahead_min=120)`: Simulate with dynamic states
  - Returns: DataFrame with state information, power flows, and H2 production

**States:**
- `COLD`: Electrolyser is off and cold
- `STARTUP`: Starting up (warm or cold)
- `RUNNING`: Normal operation
- `WARM_STANDBY`: Paused but still warm

## Supported Units and Resolutions

**Power Units:**
- `W` (Watts)
- `kW` (Kilowatts)
- `MW` (Megawatts)

**Time Resolutions:**
- `1min` (1 minute)
- `5min` (5 minutes)
- `10min` (10 minutes)
- `15min` (15 minutes)
- `1h` (1 hour)

## Use Cases

- **Renewable Energy Integration**: Calculate hydrogen production from variable renewable power sources
- **System Optimization**: Optimize electrolyser operation considering startup times and efficiency
- **Energy Storage Analysis**: Evaluate power-to-gas systems
- **Feasibility Studies**: Model hydrogen production facilities
- **Grid Services**: Analyze demand response and grid balancing with electrolysers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Moritz End (M.Sc)**  
University of Applied Sciences Cologne  
Cologne Institute for Renewable Energy (CIRE)

Developed for hydrogen production modeling and renewable energy research.

## Citation

If you use this library in your research, please cite:

```bibtex
@article{End2024,
  author = {End, Moritz},
  title = {Simulation einer effizienten Betriebsstrategie für systemdienliche PEM-Elektrolyse / Simulation of an efficient operating strategy for system-supporting PEM electrolysis},
  journal = {e \& i Elektrotechnik und Informationstechnik},
  volume = {141},
  number = {3},
  year = {2024},
  month = {August},
  doi = {10.1007/s00502-024-01230-z},
  url = {https://doi.org/10.1007/s00502-024-01230-z}
}
```

**Publication:**  
End, M. (2024). Simulation of an efficient operating strategy for system-supporting PEM electrolysis.  
*e & i Elektrotechnik und Informationstechnik*, 141(3).  
DOI: [10.1007/s00502-024-01230-z](https://doi.org/10.1007/s00502-024-01230-z)

**Repository:**
```
ElectrolyserLib (2026). Python library for PEM electrolyser modeling.
https://github.com/moend95/electrolyserlib
```