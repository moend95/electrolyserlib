import pandas as pd
import numpy as np
import io

# === DEFAULT EFFICIENCY CURVE ===
# PEM electrolyser efficiency curve (relative load vs. specific energy consumption)
DEFAULT_EFFICIENCY_CURVE = pd.DataFrame({
    'relative load [%]': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'specific energy consumption [kWh/m3]': [
        np.nan, np.nan, 6.736999752, 5.702394922, 5.09134582, 4.912176441, 4.78159199,
        4.722757406, 4.676690876, 4.658796512, 4.649861592, 4.649010393, 4.65566365,
        4.670746381, 4.685311582, 4.711295572, 4.733779855, 4.760749177, 4.797505549,
        4.830790097, 4.870956657
    ]
})

class Electrolyser:
    """PEM electrolyser model with efficiency curve."""
    
    # Resolution to time factor mapping
    RESOLUTION_MAP = {
        "1min": 1/60,
        "5min": 5/60,
        "10min": 10/60,
        "15min": 15/60,
        "1h": 1.0
    }
    
    def __init__(self, csv_data=None, nominal_power=None):
        """Initialize the electrolyser with efficiency curve and nominal power.
        
        Args:
            csv_data: Optional - CSV data (path or string) with efficiency curve
                     If None, the default efficiency curve is used
            nominal_power: Nominal power in kW
            
        Raises:
            ValueError: If nominal power <= 0 or data is invalid
        """
        if nominal_power is None:
            raise ValueError("nominal_power must be specified")
        
        if nominal_power <= 0:
            raise ValueError(f"Nominal power must be > 0, received: {nominal_power}")
        
        self.nominal_power = nominal_power
        
        # Read and prepare the efficiency curve
        if csv_data is None:
            # Use default efficiency curve
            self.df_curve = DEFAULT_EFFICIENCY_CURVE.copy()
        else:
            try:
                if isinstance(csv_data, str) and ("prozentual load" in csv_data or "relative load" in csv_data):
                    self.df_curve = pd.read_csv(io.StringIO(csv_data), sep=';', decimal=',')
                else:
                    self.df_curve = pd.read_csv(csv_data, sep=';', decimal=',')
            except Exception as e:
                raise ValueError(f"Error reading CSV data: {e}")
            
        self.df_curve = self.df_curve.dropna(subset=['specific energy consumption [kWh/m3]'])
        
        if self.df_curve.empty:
            raise ValueError("Efficiency curve is empty after removing NaN values")
        
        # Compatibility: detect alternatively named load column (e.g. 'Balance of Plant', 'relative load')
        expected_col = 'prozentual load [%]'
        if expected_col not in self.df_curve.columns and 'relative load [%]' not in self.df_curve.columns:
            # Check common alias names
            alias_candidates = [
                'relative load [%]', 'Balance of Plant', 'balance of plant', 'load', 'Load', 'Last [%]', 'last', 'prozentual load'
            ]
            load_col = None
            for col in self.df_curve.columns:
                if col in alias_candidates:
                    load_col = col
                    break
            # If no alias found: take first non-SEC column as load
            if load_col is None:
                sec_col = 'specific energy consumption [kWh/m3]'
                load_candidates = [c for c in self.df_curve.columns if c != sec_col]
                if not load_candidates:
                    raise ValueError("No suitable load column found in efficiency curve.")
                load_col = load_candidates[0]
            # Rename to expected column
            self.df_curve = self.df_curve.rename(columns={load_col: expected_col})
        elif 'relative load [%]' in self.df_curve.columns and expected_col not in self.df_curve.columns:
            # Standardize to 'prozentual load [%]'
            self.df_curve = self.df_curve.rename(columns={'relative load [%]': expected_col})
        
        self.df_curve = self.df_curve.sort_values(expected_col)
        self.min_load_pct = self.df_curve[expected_col].min()

    def _convert_to_kw(self, data, unit):
        """Convert power to kW unit.
        
        Args:
            data: Power value(s) in the given unit
            unit: Unit ("W", "kW" or "MW")
            
        Returns:
            Power in kW
            
        Raises:
            ValueError: If unit is not supported
        """
        factors = {"W": 0.001, "kW": 1.0, "MW": 1000.0}
        if unit not in factors:
            raise ValueError(f"Unknown unit '{unit}'. Supported units: {list(factors.keys())}")
        return data * factors[unit]

    def calc_h2(self, timeseries, unit="kW", resolution="1h"):
        """Calculate H2 production from input load and efficiency curve.
        
        Args:
            timeseries: List/array of power values
            unit: Unit of input data ("W", "kW", "MW"). Default: "kW"
            resolution: Time resolution ("1min", "5min", "10min", "15min", "1h"). Default: "1h"
            
        Returns:
            DataFrame with columns: input_power_raw_kW, used_power_kW, excess_power_kW, h2_produced_m3
            
        Raises:
            ValueError: If resolution is not supported
        """
        # 1. Input power in kW
        raw_power_kw = self._convert_to_kw(pd.Series(timeseries), unit)
        
        # 2. Limit to nominal load (clipping)
        # Everything above nominal_power is clipped
        actual_power_kw = raw_power_kw.clip(upper=self.nominal_power)
        
        # 3. Calculate excess power
        excess_power = raw_power_kw - actual_power_kw
        
        # 4. Percentage load for the curve (0 - 100%)
        load_pct = (actual_power_kw / self.nominal_power) * 100
        
        # 5. Interpolate specific consumption
        spec_cons = np.interp(
            load_pct, 
            self.df_curve['prozentual load [%]'], 
            self.df_curve['specific energy consumption [kWh/m3]']
        )
        
        # 6. Calculate production
        if resolution not in self.RESOLUTION_MAP:
            raise ValueError(f"Unknown resolution '{resolution}'. Supported values: {list(self.RESOLUTION_MAP.keys())}")
        
        time_factor = self.RESOLUTION_MAP[resolution]
        energy_used_kwh = actual_power_kw * time_factor
        
        # Prevent division by zero when spec_cons = 0
        h2_produced_m3 = np.divide(energy_used_kwh, spec_cons, 
                                   where=(spec_cons != 0), 
                                   out=np.zeros_like(energy_used_kwh))
        
        # 7. Minimum load logic (no production below minimum load)
        h2_produced_m3 = np.where(load_pct < self.min_load_pct, 0.0, h2_produced_m3)
        # If not producing, no power is consumed -> excess increases
        final_excess = np.where(load_pct < self.min_load_pct, raw_power_kw, excess_power)
        
        # Result DataFrame
        return pd.DataFrame({
            'input_power_raw_kW': raw_power_kw,
            'used_power_kW': np.where(load_pct < self.min_load_pct, 0.0, actual_power_kw),
            'excess_power_kW': final_excess,
            'h2_produced_m3': h2_produced_m3
        })
    
    def get_statistics(self, result_df, resolution="1h"):
        """Calculate key performance indicators from simulation results.
        
        Args:
            result_df: DataFrame returned by calc_h2() method
            resolution: Time resolution used in simulation ("1min", "5min", "10min", "15min", "1h")
            
        Returns:
            dict with the following keys:
                - total_h2_m3: Total H2 produced [m³]
                - total_h2_kg: Total H2 produced [kg] (at 0°C, 1 bar: 0.08988 kg/m³)
                - avg_efficiency_pct: Average efficiency [%] (based on LHV = 3.0 kWh/m³)
                - full_load_hours: Equivalent full load hours [h]
                - operating_hours: Operating hours (when producing H2) [h]
                - total_energy_used_kwh: Total energy consumed [kWh]
                - total_excess_energy_kwh: Total excess energy [kWh]
                - energy_without_production_kwh: Energy used without H2 production [kWh]
                
        Raises:
            ValueError: If resolution is not supported or result_df is invalid
        """
        if resolution not in self.RESOLUTION_MAP:
            raise ValueError(f"Unknown resolution '{resolution}'. Supported values: {list(self.RESOLUTION_MAP.keys())}")
        
        time_factor = self.RESOLUTION_MAP[resolution]
        
        # Required columns
        required_cols = ['used_power_kW', 'excess_power_kW', 'h2_produced_m3']
        for col in required_cols:
            if col not in result_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 1. Total H2 production
        total_h2_m3 = result_df['h2_produced_m3'].sum()
        # Conversion: 1 m³ H2 (at 0°C, 1 bar) = 0.08988 kg
        total_h2_kg = total_h2_m3 * 0.08988
        
        # 2. Total energy
        total_energy_used_kwh = (result_df['used_power_kW'] * time_factor).sum()
        total_excess_energy_kwh = (result_df['excess_power_kW'] * time_factor).sum()
        
        # 3. Average efficiency
        # LHV (Lower Heating Value) of H2: ~3.0 kWh/m³
        LHV_H2 = 3.0  # kWh/m³
        theoretical_energy_kwh = total_h2_m3 * LHV_H2
        
        if total_energy_used_kwh > 0:
            avg_efficiency_pct = (theoretical_energy_kwh / total_energy_used_kwh) * 100
        else:
            avg_efficiency_pct = 0.0
        
        # 4. Full load hours
        if self.nominal_power > 0:
            full_load_hours = total_energy_used_kwh / self.nominal_power
        else:
            full_load_hours = 0.0
        
        # 5. Operating hours (when producing H2)
        operating_steps = (result_df['h2_produced_m3'] > 0).sum()
        operating_hours = operating_steps * time_factor
        
        # 6. Energy without production (used power but no H2 output)
        # This happens during startup, heat-up, or below minimum load
        mask_no_production = (result_df['used_power_kW'] > 0) & (result_df['h2_produced_m3'] == 0)
        energy_without_production_kwh = (result_df.loc[mask_no_production, 'used_power_kW'] * time_factor).sum()
        
        return {
            'total_h2_m3': round(total_h2_m3, 2),
            'total_h2_kg': round(total_h2_kg, 2),
            'avg_efficiency_pct': round(avg_efficiency_pct, 2),
            'full_load_hours': round(full_load_hours, 2),
            'operating_hours': round(operating_hours, 2),
            'total_energy_used_kwh': round(total_energy_used_kwh, 2),
            'total_excess_energy_kwh': round(total_excess_energy_kwh, 2),
            'energy_without_production_kwh': round(energy_without_production_kwh, 2)
        }
    
class DynamicElectrolyser(Electrolyser):
    """Electrolyser with dynamic startup/standby characteristics."""
    
    # State constants
    STATE_COLD = "COLD"
    STATE_STARTUP = "STARTUP"
    STATE_RUNNING = "RUNNING"
    STATE_WARM_STANDBY = "WARM_STANDBY"
    
    def __init__(self, csv_data=None, nominal_power=None, 
                 cold_startup_min=30, warm_startup_min=10, 
                 standby_limit_min=30, startup_power_frac=0.25):
        """
        Args:
            csv_data: Efficiency curve (CSV path or string)
            nominal_power: Nominal power [kW]
            cold_startup_min: Time for cold starts [min]
            warm_startup_min: Time for warm starts [min]
            standby_limit_min: How long to stay warm before cold start [min]
            startup_power_frac: Power fraction during startup (e.g. 0.25 = 25% P_nom)
        """
        super().__init__(csv_data, nominal_power)
        
        # Startup parameters in minutes (not kWh!)
        self.cold_startup_min = cold_startup_min
        self.warm_startup_min = warm_startup_min
        self.startup_power_frac = startup_power_frac
        
        # Standby parameters
        self.standby_limit_min = standby_limit_min
        self.standby_maintenance_power = 0.01 * nominal_power  # 1% for heat maintenance (optional)
        
        # State variables
        self.state = self.STATE_COLD
        self.startup_counter = 0  # Counter for startup minutes instead of energy
        self.is_warm_start = False  # Remember: warm or cold start?
        self.standby_timer = 0  # [min] Remaining standby time
        
    def _get_resolution_min(self, resolution):
        """Convert resolution string to minutes."""
        resolution = resolution.lower().strip()
        if resolution.endswith("min"):
            return float(resolution.replace("min", ""))
        elif resolution.endswith("h"):
            return float(resolution.replace("h", "")) * 60
        else:
            raise ValueError(f"Unknown resolution '{resolution}'. Use e.g. '15min' or '1h'")
    
    def _check_look_ahead(self, future_series, resolution_min, horizon_min=60):
        """
        Check if sufficient power is available in the next 'horizon_min' minutes
        for startup + operation.
        
        Args:
            future_series: Future power values (from current index)
            resolution_min: Time resolution [min]
            horizon_min: Look-ahead horizon [min]
            
        Returns:
            bool: True if average power > 20% nominal power
        """
        steps_to_look = int(np.ceil(horizon_min / resolution_min))
        if steps_to_look <= 0 or len(future_series) == 0:
            return False
        
        future_data = future_series[:min(steps_to_look, len(future_series))]
        avg_power = future_data.mean()
        min_threshold = self.nominal_power * 0.2
        
        return avg_power > min_threshold
    
    def _calculate_h2_from_power(self, p_used_kw, time_factor, load_pct_override=None):
        """
        Calculate H2 production from consumed power using the efficiency curve.
        
        Args:
            p_used_kw: Consumed power [kW]
            time_factor: Time counter for energy calculation (e.g. 0.25 for 15min)
            load_pct_override: Optional: Force load percentage (instead of calculating from power)
            
        Returns:
            float: H2 production [m³]
        """
        if p_used_kw <= 0:
            return 0.0
        
        # Calculate load percentage
        if load_pct_override is not None:
            load_pct = load_pct_override
        else:
            load_pct = (p_used_kw / self.nominal_power) * 100
        
        # Interpolate specific consumption
        spec_cons = np.interp(
            load_pct,
            self.df_curve['prozentual load [%]'],
            self.df_curve['specific energy consumption [kWh/m3]']
        )
        
        # H2 from energy
        if spec_cons > 0:
            energy_kwh = p_used_kw * time_factor
            h2_m3 = energy_kwh / spec_cons
            return h2_m3
        
        return 0.0
    
    def simulate(self, timeseries, unit="kW", resolution="15min", look_ahead_min=120):
        """
        Simulate electrolyser with dynamic states.
        
        Args:
            timeseries: Array of input load values
            unit: Unit of input load
            resolution: Time resolution (e.g. '15min', '1h')
            look_ahead_min: Horizon for look-ahead for startup [min]
            
        Returns:
            DataFrame with columns: state, p_in, p_used, p_excess, h2_produced, startup_counter
        """
        # Input preparation
        raw_power_kw = self._convert_to_kw(pd.Series(timeseries), unit)
        res_min = self._get_resolution_min(resolution)
        time_factor = res_min / 60  # for energy calculation in kWh
        
        results = []
        
        for i, p_in in enumerate(raw_power_kw):
            p_used = 0.0
            h2_out = 0.0
            prev_state = self.state
            
            # === STATE TRANSITIONS AND LOGIC ===
            
            if self.state == self.STATE_COLD:
                # Consider: Is startup worthwhile?
                has_enough_future_power = self._check_look_ahead(
                    raw_power_kw[i:].values, res_min, horizon_min=look_ahead_min
                )
                # Criterion: Current power > 10% nominal load AND good future forecast
                if p_in >= (self.nominal_power * 0.1) and has_enough_future_power:
                    self.state = self.STATE_STARTUP
                    self.startup_counter = 0  # Reset startup counter
                    self.is_warm_start = False
            
            elif self.state == self.STATE_STARTUP:
                # Startup with max. 25% nominal load
                p_used = min(p_in, self.nominal_power * self.startup_power_frac)
                # Accumulate startup time
                self.startup_counter += res_min
                
                # Determine required startup time
                required_startup_min = self.cold_startup_min
                
                # Check if startup completed
                if self.startup_counter >= required_startup_min:
                    # Transition to RUNNING
                    self.state = self.STATE_RUNNING
                    self.startup_counter = 0
            
            elif self.state == self.STATE_RUNNING:
                # Normal operation
                if p_in < (self.nominal_power * self.min_load_pct / 100):
                    # Too little power (below minimum load) -> Standby mode
                    self.state = self.STATE_WARM_STANDBY
                    self.standby_timer = self.standby_limit_min
                    # No production in this step yet
                else:
                    # Normal H2 production
                    p_used = min(p_in, self.nominal_power)
                    h2_out = self._calculate_h2_from_power(p_used, time_factor)
            
            elif self.state == self.STATE_WARM_STANDBY:
                # Wait for restart or cooldown
                self.standby_timer -= res_min
                
                if p_in >= (self.nominal_power * self.min_load_pct / 100):
                    # Enough power! -> Quick start (warm start, only 10 min instead of 30)
                    self.state = self.STATE_STARTUP
                    self.startup_counter = 0
                    self.is_warm_start = True
                
                elif self.standby_timer <= 0:
                    # No power for too long -> Cold start mode
                    self.state = self.STATE_COLD
            
            # Calculate excess
            p_excess = p_in - p_used
            
            # Logging
            results.append({
                'state': self.state,
                'state_prev': prev_state,
                'p_in_kW': p_in,
                'p_used_kW': p_used,
                'p_excess_kW': p_excess,
                'h2_produced_m3': h2_out,
                'startup_counter_min': self.startup_counter if self.state == self.STATE_STARTUP else 0,
                'standby_timer_min': self.standby_timer if self.state == self.STATE_WARM_STANDBY else 0
            })
        
        return pd.DataFrame(results)
    
    def get_statistics(self, result_df, resolution="15min"):
        """Calculate key performance indicators from dynamic simulation results.
        
        Args:
            result_df: DataFrame returned by simulate() method
            resolution: Time resolution used in simulation ("1min", "5min", "10min", "15min", "1h")
            
        Returns:
            dict with the following keys:
                - total_h2_m3: Total H2 produced [m³]
                - total_h2_kg: Total H2 produced [kg] (at 0°C, 1 bar: 0.08988 kg/m³)
                - avg_efficiency_pct: Average efficiency [%] (based on LHV = 3.0 kWh/m³)
                - full_load_hours: Equivalent full load hours [h]
                - operating_hours: Operating hours (when producing H2) [h]
                - total_energy_used_kwh: Total energy consumed [kWh]
                - total_excess_energy_kwh: Total excess energy [kWh]
                - startup_energy_kwh: Energy used during startup phases [kWh]
                - standby_steps: Number of time steps in warm standby mode
                - cold_starts: Number of cold starts
                - warm_starts: Number of warm starts
                
        Raises:
            ValueError: If resolution is not supported or result_df is invalid
        """
        res_min = self._get_resolution_min(resolution)
        time_factor = res_min / 60  # for energy calculation in kWh
        
        # Required columns
        required_cols = ['p_used_kW', 'p_excess_kW', 'h2_produced_m3', 'state']
        for col in required_cols:
            if col not in result_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 1. Total H2 production
        total_h2_m3 = result_df['h2_produced_m3'].sum()
        # Conversion: 1 m³ H2 (at 0°C, 1 bar) = 0.08988 kg
        total_h2_kg = total_h2_m3 * 0.08988
        
        # 2. Total energy
        total_energy_used_kwh = (result_df['p_used_kW'] * time_factor).sum()
        total_excess_energy_kwh = (result_df['p_excess_kW'] * time_factor).sum()
        
        # 3. Average efficiency
        # LHV (Lower Heating Value) of H2: ~3.0 kWh/m³
        LHV_H2 = 3.0  # kWh/m³
        theoretical_energy_kwh = total_h2_m3 * LHV_H2
        
        if total_energy_used_kwh > 0:
            avg_efficiency_pct = (theoretical_energy_kwh / total_energy_used_kwh) * 100
        else:
            avg_efficiency_pct = 0.0
        
        # 4. Full load hours
        if self.nominal_power > 0:
            full_load_hours = total_energy_used_kwh / self.nominal_power
        else:
            full_load_hours = 0.0
        
        # 5. Operating hours (when producing H2)
        operating_steps = (result_df['h2_produced_m3'] > 0).sum()
        operating_hours = operating_steps * time_factor
        
        # 6. Energy during startup
        startup_energy_kwh = (result_df[result_df['state'] == self.STATE_STARTUP]['p_used_kW'] * time_factor).sum()
        
        # 7. Standby statistics
        standby_steps = (result_df['state'] == self.STATE_WARM_STANDBY).sum()
        
        # 8. Count state transitions (cold/warm starts)
        cold_starts = 0
        warm_starts = 0
        
        # Detect transitions from COLD/WARM_STANDBY to STARTUP
        for i in range(1, len(result_df)):
            prev_state = result_df.iloc[i-1]['state']
            curr_state = result_df.iloc[i]['state']
            
            if curr_state == self.STATE_STARTUP and prev_state == self.STATE_COLD:
                cold_starts += 1
            elif curr_state == self.STATE_STARTUP and prev_state == self.STATE_WARM_STANDBY:
                warm_starts += 1
        
        return {
            'total_h2_m3': round(total_h2_m3, 2),
            'total_h2_kg': round(total_h2_kg, 2),
            'avg_efficiency_pct': round(avg_efficiency_pct, 2),
            'full_load_hours': round(full_load_hours, 2),
            'operating_hours': round(operating_hours, 2),
            'total_energy_used_kwh': round(total_energy_used_kwh, 2),
            'total_excess_energy_kwh': round(total_excess_energy_kwh, 2),
            'startup_energy_kwh': round(startup_energy_kwh, 2),
            'standby_steps': int(standby_steps),
            'cold_starts': int(cold_starts),
            'warm_starts': int(warm_starts)
        }