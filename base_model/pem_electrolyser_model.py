"""
PEM Electrolyser Stack Model

This module contains a detailed electrochemical model for PEM (Proton Exchange Membrane)
electrolysers. It calculates cell voltage based on the Nernst equation, activation
overpotential (Butler-Volmer kinetics), and ohmic losses. The model includes balance
of plant components such as power electronics, gas drying, compression, heat management,
and pumps.

This model is used to generate efficiency curves for the electrolyserlib package.

References:
[1] End, M. (2024). Simulation einer effizienten Betriebsstrategie für 
    systemdienliche PEM-Elektrolyse. e & i Elektrotechnik und Informationstechnik, 
    141(3). DOI: 10.1007/s00502-024-01230-z
[2] Tjarks, G. (2017). PEM-Elektrolyse-Systeme zur Anwendung in Power-to-Gas Anlagen. 
    Dissertation, RWTH Aachen, Forschungszentrum Jülich.
[3] Olivier, P., Bourasseau, C., & Bouamama, B. (2017). Low-temperature electrolysis 
    system modelling: A review. Renewable and Sustainable Energy Reviews, 78, 280-300.
[4] Springer, T.E., et al. (1991). Polymer electrolyte fuel cell model. 
    Journal of the Electrochemical Society, 138(8), 2334-2342.

Author: Moritz End
Institution: University of Applied Sciences Cologne, 
             Cologne Institute for Renewable Energy (CIRE)
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class PEMElectrolyser:
    """
    PEM Electrolyser stack model for hydrogen production calculations.
    
    This class models the electrochemical behavior of a PEM electrolyser stack,
    including cell voltage calculation, hydrogen production rate, and balance
    of plant components.
    
    Parameters
    ----------
    nominal_power : float
        Nominal electrical input power in kW. Must be a multiple of 500 kW
        (single stack power).
    timestep : float
        Time step duration for calculations.
    timestep_unit : str
        Unit of the time step: 's' (seconds), 'min' (minutes), or 'h' (hours).
    compression_pressure : float, optional
        Target compression pressure in bar. Default is 0 (no compression).
        If specified, must be >= 30 bar.
    
    Attributes
    ----------
    F : float
        Faraday's constant [C/mol]
    R : float
        Ideal gas constant [J/(mol*K)]
    n : int
        Number of electrons transferred in reaction
    E_th_0 : float
        Thermoneutral voltage at standard state [V]
    lhv : float
        Lower heating value of H2 [kWh/kg]
    hhv : float
        Higher heating value of H2 [kWh/kg]
    n_cells : int
        Number of cells per stack
    cell_area : float
        Cell active area [cm²]
    temperature : float
        Stack operating temperature [°C]
    
    Examples
    --------
    >>> electrolyser = PEMElectrolyser(
    ...     nominal_power=500,
    ...     timestep=15,
    ...     timestep_unit='min'
    ... )
    >>> h2_production = electrolyser.calc_H2_mass_flow_rate(400)
    >>> print(f"H2 production: {h2_production:.2f} kg/h")
    """

    def __init__(self, nominal_power, timestep, timestep_unit, compression_pressure=0):
        """
        Initialize the PEM Electrolyser model.
        
        Parameters
        ----------
        nominal_power : float
            Nominal electrical input power in kW.
        timestep : float
            Time step duration for calculations.
        timestep_unit : str
            Unit of the time step: 's', 'min', or 'h'.
        compression_pressure : float, optional
            Target compression pressure in bar. Default is 0 (no compression).
        """
        # Store raw input parameters
        self._nominal_power_input = nominal_power
        self._timestep_input = timestep
        self._timestep_unit_input = timestep_unit
        self._compression_pressure_input = compression_pressure
        
        # Physical constants
        self.F = 96485.34       # Faraday's constant [C/mol]
        self.R = 8.314          # Ideal gas constant [J/(mol*K)]
        self.n = 2              # Number of electrons transferred in reaction
        self.gibbs = 237.24e3   # Gibbs free energy [J/mol]
        self.E_th_0 = 1.481     # Thermoneutral voltage at standard state [V]
        self.M = 2.016          # Molecular weight of H2 [g/mol]
        self.lhv = 33.33        # Lower heating value of H2 [kWh/kg]
        self.hhv = 39.41        # Higher heating value of H2 [kWh/kg]
        self.rho_H2 = 0.08988   # Density of H2 at STP [kg/m³]
        self.rho_O2 = 1.429     # Density of O2 at STP [kg/m³]
        self.T_ambient = 20     # Ambient temperature [°C]
        
        # Validate inputs
        self._validate_inputs()
        
        # Operating parameters (after validation)
        self.P_nominal = float(self._nominal_power_input)  # Nominal power [kW]
        self.dt = float(self._timestep_input)              # Time step value
        self.timestep_unit = self._timestep_unit_input     # Time step unit
        self.compression_pressure = float(self._compression_pressure_input)  # [bar]
        
        # Pressure parameters (must be set before stack calculations)
        self.p_atmo = 101325               # Atmospheric pressure [Pa]
        self.p_anode = self.p_atmo         # Anode pressure [Pa]
        
        # Stack parameters (based on typical commercial PEM electrolyser)
        self.n_cells = 56                  # Number of cells per stack
        self.cell_area = 2500              # Cell active area [cm²]
        self.max_current_density = 2.1     # Maximum current density [A/cm²]
        self.temperature = 50              # Stack operating temperature [°C]
        self.stack_power = 500             # Nominal power per stack [kW]
        
        # Calculate number of stacks (at least 1)
        self.n_stacks = max(1, int(self.P_nominal / self.stack_power))
        
        # Recalculate power limits based on actual stack configuration
        self.P_min = self.P_nominal * 0.1  # Minimum power (10% of nominal) [kW]
        self.P_max = self.P_nominal        # Maximum power [kW]
    
    def _validate_inputs(self):
        """
        Validate all input parameters.
        
        Raises
        ------
        ValueError
            If any input parameter is invalid.
        TypeError
            If input types are incorrect.
        """
        # Validate nominal power
        try:
            power = float(self._nominal_power_input)
            if power <= 0:
                raise ValueError("Nominal power must be positive.")
        except (ValueError, TypeError):
            raise TypeError(
                "Nominal power must be a positive number. "
                f"Received: {self._nominal_power_input}"
            )
        
        # Validate timestep
        try:
            timestep = float(self._timestep_input)
            if timestep <= 0:
                raise ValueError("Timestep must be positive.")
        except (ValueError, TypeError):
            raise TypeError(
                "Timestep must be a positive number. "
                f"Received: {self._timestep_input}"
            )
        
        # Validate timestep unit
        valid_units = {
            's': 'seconds',
            'sec': 'seconds',
            'second': 'seconds',
            'seconds': 'seconds',
            'min': 'minutes',
            'm': 'minutes',
            'minute': 'minutes',
            'minutes': 'minutes',
            'h': 'hours',
            'hr': 'hours',
            'hour': 'hours',
            'hours': 'hours'
        }
        
        unit = str(self._timestep_unit_input).lower()
        if unit not in valid_units:
            raise ValueError(
                f"Invalid timestep unit: '{self._timestep_unit_input}'. "
                f"Valid options are: 's' (seconds), 'min' (minutes), 'h' (hours)"
            )
        self._timestep_unit_normalized = valid_units[unit]
        
        # Validate compression pressure
        try:
            pressure = float(self._compression_pressure_input)
            if pressure < 0:
                raise ValueError("Compression pressure cannot be negative.")
            if pressure > 0 and pressure < 30:
                raise ValueError(
                    "Compression pressure must be at least 30 bar if compression is enabled. "
                    "Set to 0 to disable compression."
                )
        except (ValueError, TypeError) as e:
            if "at least 30 bar" in str(e) or "cannot be negative" in str(e):
                raise
            raise TypeError(
                "Compression pressure must be a number. "
                f"Received: {self._compression_pressure_input}"
            )
        
        # Validate that nominal power is a multiple of stack power (500 kW)
        stack_power = 500  # kW per stack
        if float(self._nominal_power_input) % stack_power != 0:
            raise ValueError(
                f"Nominal power must be a multiple of {stack_power} kW "
                f"(single stack power). Received: {self._nominal_power_input} kW"
            )
    
    def _calc_stack_nominal_power(self):
        """
        Calculate the nominal power of a single stack.
        
        Returns
        -------
        float
            Nominal stack power in kW.
        """
        # Maximum current for the stack
        I_max = self.max_current_density * self.cell_area  # [A]
        
        # Cell voltage at maximum current
        U_cell_max = self.calc_cell_voltage(I_max, self.temperature)
        
        # Stack power
        P_stack = (self.n_cells * U_cell_max * I_max) / 1000  # [kW]
        
        return P_stack
    
    def stack_nominal_power(self):
        """
        Get the nominal power of a single stack.
        
        Returns
        -------
        float
            Nominal stack power in kW.
        """
        return self._calc_stack_nominal_power()
    
    def calc_cell_voltage(self, current, temperature):
        """
        Calculate the cell voltage using the electrochemical model.
        
        The cell voltage is calculated as:
        U_cell = E_rev + eta_act_anode + eta_act_cathode + eta_ohm
        
        Where:
        - E_rev: Reversible (Nernst) voltage with pressure correction
        - eta_act: Activation overpotential (Tafel equation)
        - eta_ohm: Ohmic overpotential (electronic + ionic resistance)
        
        Parameters
        ----------
        current : float
            Cell current in Amperes [A].
        temperature : float
            Operating temperature in Celsius [°C].
        
        Returns
        -------
        float
            Cell voltage in Volts [V].
        
        Notes
        -----
        Model parameters based on literature values and validated against
        commercial PEM electrolyser data. Typical cell voltages range
        from ~1.5V at low current to ~1.77V at 2 A/cm² current density.
        
        References
        ----------
        [1] End, M. (2024). DOI: 10.1007/s00502-024-01230-z - Model description
        [2] Tjarks, G. (2017). FZ Jülich - Tafel equation parameters (Eq. 3.11)
        [3] Olivier et al. (2017). Review - Butler-Volmer/Tafel kinetics
        [4] Springer et al. (1991). Nafion conductivity model (Eq. 14)
        
        Model validated against commercial PEM electrolyser specifications
        """
        # Convert temperature to Kelvin
        T_K = temperature + 273.15  # [K]
        
        # Current density [A/cm²]
        i = current / self.cell_area
        
        # Handle zero or very low current
        if i < 1e-6:
            return 1.229  # Return reversible voltage at zero current
        
        # === Reversible voltage (Nernst equation) ===
        E_rev_0 = self.gibbs / (self.n * self.F)  # ~1.229 V
        
        # Pressure at electrodes [Pa]
        p_atmo = self.p_atmo  # 101325 Pa
        p_anode = 200000      # 2 bar [Pa]
        p_cathode = 3000000   # 30 bar [Pa]
        
        # Saturation pressure of water (Arden Buck equation) [Pa]
        T_celsius = temperature
        p_H2O_sat = 0.61121 * np.exp((18.678 - T_celsius/234.5) * T_celsius / (257.14 + T_celsius)) * 1e3  # [Pa]
        
        # General Nernst equation
        E_rev = E_rev_0 + (self.R * T_K / (self.n * self.F)) * np.log(
            ((p_anode - p_H2O_sat) / p_atmo) * np.sqrt((p_cathode - p_H2O_sat) / p_atmo)
        )
        
        # === Activation overpotential (Tafel equation) ===
        # Tafel equation: eta = (R*T)/(alpha*z*F) * ln(i/i_0)
        # Based on: Tjarks (2017) Eq. 3.11, Olivier et al. (2017) Table 2
        T_anode = T_K
        T_cathode = T_K
        
        # Charge transfer coefficients [-]
        # Ref: Tjarks (2017), Olivier et al. (2017) - ACT V formulation
        alpha_a = 2    # Anode charge transfer coefficient
        alpha_c = 0.5  # Cathode charge transfer coefficient
        
        # Stoichiometric coefficients of electrons transferred [-]
        # Based on half-reactions: 2H2O -> O2 + 4H+ + 4e- (anode)
        #                          2H+ + 2e- -> H2 (cathode)
        z_a = 4  # Anode electron transfer number
        z_c = 2  # Cathode electron transfer number
        
        # Exchange current densities [A/cm²]
        # Ref: Fitted from Tjarks (2017) Tab. 3.1, typical range 10^-9 to 10^-8
        i_0_a = 1e-9   # Anode exchange current density
        i_0_c = 1e-3   # Cathode exchange current density
        
        # Tafel equation: eta = (R*T)/(alpha*z*F) * ln(i/i_0)
        V_act_a = (self.R * T_anode / (alpha_a * z_a * self.F)) * np.log(i / i_0_a)
        V_act_c = (self.R * T_cathode / (alpha_c * z_c * self.F)) * np.log(i / i_0_c)
        
        # === Ohmic overpotential ===
        # Nafion membrane parameters
        # Ref: Springer et al. (1991), used in Tjarks (2017) Eq. 3.13,
        #      Olivier et al. (2017) Eq. 14
        lambda_nafion = 25      # Water content parameter [-] (Ref: Tjarks 2017)
        t_nafion = 0.01         # Membrane thickness [cm] (100 μm, typical range: 50-200 μm)
        
        # Nafion conductivity [S/cm] - Springer correlation
        # sigma = (0.005139*lambda - 0.00326) * exp(1268*(1/303 - 1/T))
        sigma_nafion = (0.005139 * lambda_nafion - 0.00326) * np.exp(1268 * (1/303 - 1/T_K))
        
        # Ionic resistance [Ohm*cm²]
        R_ohmic_ionic = t_nafion / sigma_nafion
        
        # Electronic resistance [Ohm*cm²]
        # Ref: Fitted from experimental data (Tjarks 2017, Tab. 3.1: ~50-160 mΩ·cm²)
        R_ohmic_elec = 50e-3
        
        # Ohmic overpotential
        V_ohmic = i * (R_ohmic_elec + R_ohmic_ionic)
        
        # === Total cell voltage ===
        V_cell = E_rev + V_act_a + V_act_c + V_ohmic
        
        return V_cell
    
    def create_polarization_curve(self, temperature=None, n_points=100):
        """
        Generate a polarization curve (U-I characteristic) for the cell.
        
        Parameters
        ----------
        temperature : float, optional
            Operating temperature in °C. Default is the stack temperature.
        n_points : int, optional
            Number of data points. Default is 100.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'current_density': Array of current densities [A/cm²]
            - 'cell_voltage': Array of cell voltages [V]
            - 'current': Array of currents [A]
            - 'power_density': Array of power densities [W/cm²]
        """
        if temperature is None:
            temperature = self.temperature
        
        # Current density range (0 to max)
        j_range = np.linspace(0.001, self.max_current_density, n_points)
        
        # Calculate corresponding currents
        I_range = j_range * self.cell_area
        
        # Calculate cell voltage for each current
        U_cell = np.array([self.calc_cell_voltage(I, temperature) for I in I_range])
        
        # Power density
        P_density = j_range * U_cell  # [W/cm²]
        
        return {
            'current_density': j_range,
            'cell_voltage': U_cell,
            'current': I_range,
            'power_density': P_density
        }
    
    def plot_polarization_curve(self, temperature=None, save_path=None):
        """
        Plot the polarization curve.
        
        Parameters
        ----------
        temperature : float, optional
            Operating temperature in °C. Default is the stack temperature.
        save_path : str, optional
            Path to save the figure. If None, the figure is displayed.
        
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        data = self.create_polarization_curve(temperature)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot cell voltage
        color1 = 'tab:blue'
        ax1.set_xlabel('Current Density [A/cm²]')
        ax1.set_ylabel('Cell Voltage [V]', color=color1)
        ax1.plot(data['current_density'], data['cell_voltage'], color=color1, linewidth=2, label='Cell Voltage')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim([1.2, 2.5])
        ax1.grid(True, alpha=0.3)
        
        # Plot power density on secondary axis
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Power Density [W/cm²]', color=color2)
        ax2.plot(data['current_density'], data['power_density'], color=color2, linewidth=2, linestyle='--', label='Power Density')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Title and legend
        temp = temperature if temperature is not None else self.temperature
        plt.title(f'PEM Electrolyser Polarization Curve (T = {temp}°C)')
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def calculate_cell_current(self, P_dc):
        """
        Calculate the cell current for a given DC power input.
        
        Parameters
        ----------
        P_dc : float
            DC power input in kW.
        
        Returns
        -------
        float
            Cell current in Amperes [A].
        """
        # Power per stack
        P_stack = P_dc / max(self.n_stacks, 1)
        
        # Iterative solution to find current
        # P = n_cells * U_cell * I
        # Start with initial guess
        I_guess = 1000  # [A]
        
        for _ in range(50):  # Maximum 50 iterations
            U_cell = self.calc_cell_voltage(I_guess, self.temperature)
            P_calc = (self.n_cells * U_cell * I_guess) / 1000  # [kW]
            
            # Update current estimate
            if P_calc > 0:
                I_new = I_guess * (P_stack / P_calc)
            else:
                I_new = I_guess
            
            # Check convergence
            if abs(I_new - I_guess) < 0.1:
                break
            
            I_guess = I_new
        
        return I_guess
    
    def calc_faradaic_efficiency(self, current):
        """
        Calculate the Faradaic efficiency based on current density and pressure.
        
        The Faradaic efficiency accounts for parasitic reactions and
        gas crossover through the membrane.
        
        Parameters
        ----------
        current : float
            Cell current in Amperes [A].
        
        Returns
        -------
        float
            Faradaic efficiency as a fraction (0-1).
        
        Notes
        -----
        Empirical model from literature:
        eta_F = (a1*p + a2) * i^b + c
        
        with:
        - a1 = -0.0034
        - a2 = -0.001711
        - b = -1
        - c = 1
        - p = 20 bar (electrolyser operating pressure)
        
        Reference: https://res.mdpi.com/d_attachment/energies/energies-13-04792
        """
        # Current density [A/cm²]
        i = current / self.cell_area
        
        # Avoid division by zero at very low current
        if i < 1e-6:
            return 0.5  # Low efficiency at near-zero current
        
        # Electrolyser operating pressure [bar]
        p = 20
        
        # Empirical parameters
        a_1 = -0.0034
        a_2 = -0.001711
        b = -1
        c = 1
        
        # Faradaic efficiency
        eta_F = (a_1 * p + a_2) * (i ** b) + c
        
        # Limit to realistic range
        eta_F = max(0.5, min(1.0, eta_F))
        
        return eta_F
    
    def calc_H2_mass_flow_rate(self, P_ac):
        """
        Calculate the hydrogen mass flow rate.
        
        Parameters
        ----------
        P_ac : float
            AC power input in kW.
        
        Returns
        -------
        float
            Hydrogen mass flow rate in kg/h.
        """
        # Limit power to operating range
        P_ac = max(0, min(P_ac, self.P_max))
        
        if P_ac < self.P_min:
            return 0.0
        
        # DC power after power electronics losses
        P_dc = self.power_dc(P_ac)
        
        # Cell current
        I = self.calculate_cell_current(P_dc)
        
        # Faradaic efficiency
        eta_F = self.calc_faradaic_efficiency(I)
        
        # Hydrogen production rate from Faraday's law
        # n_H2 = (I * eta_F) / (n * F) [mol/s]
        # m_H2 = n_H2 * M [g/s]
        
        n_H2_rate = (self.n_stacks * self.n_cells * I * eta_F) / (self.n * self.F)  # [mol/s]
        m_H2_rate = n_H2_rate * self.M  # [g/s]
        m_H2_rate_kg_h = m_H2_rate * 3.6  # Convert g/s to kg/h
        
        return m_H2_rate_kg_h
    
    def calc_O2_mass_flow_rate(self, H2_mass_flow_rate):
        """
        Calculate the oxygen mass flow rate based on stoichiometry.
        
        Parameters
        ----------
        H2_mass_flow_rate : float
            Hydrogen mass flow rate in kg/h.
        
        Returns
        -------
        float
            Oxygen mass flow rate in kg/h.
        
        Notes
        -----
        From the reaction: 2H2O -> 2H2 + O2
        Mass ratio: m_O2 / m_H2 = (1 * 32) / (2 * 2) = 8
        """
        # Stoichiometric ratio: O2/H2 = 0.5 (molar)
        # Mass ratio: (0.5 * 32) / 2.016 = 7.94
        mass_ratio = (0.5 * 32) / 2.016
        
        return H2_mass_flow_rate * mass_ratio
    
    def calc_H2O_mass_flow_rate(self, H2_mass_flow_rate):
        """
        Calculate the water consumption rate based on stoichiometry.
        
        Parameters
        ----------
        H2_mass_flow_rate : float
            Hydrogen mass flow rate in kg/h.
        
        Returns
        -------
        float
            Water mass flow rate in kg/h.
        
        Notes
        -----
        From the reaction: 2H2O -> 2H2 + O2
        Mass ratio: m_H2O / m_H2 = (2 * 18) / (2 * 2) = 9
        Including 1.5x factor for transport water.
        """
        # Stoichiometric ratio: H2O/H2 = 1 (molar)
        # Mass ratio: 18 / 2.016 = 8.93
        # Factor 1.5 accounts for transport water in the cell
        mass_ratio = (18 / 2.016) * 1.5
        
        return H2_mass_flow_rate * mass_ratio
    
    def power_electronics(self, P_nominal, P_ac):
        """
        Calculate power electronics losses (AC/DC conversion).
        
        Parameters
        ----------
        P_nominal : float
            Nominal power in kW.
        P_ac : float
            AC power input in kW.
        
        Returns
        -------
        float
            Power electronics losses in kW.
        
        Notes
        -----
        Efficiency curve based on typical industrial rectifier data.
        Maximum efficiency ~97.7% at nominal load.
        
        Formula: P_loss = P_ac * (1 - eta)
        Source: Master thesis, validated against [22, 37]
        """
        if P_ac <= 0:
            return 0.0
        
        # Efficiency curve from master thesis
        relative_performance = [0.0, 0.09, 0.12, 0.15, 0.189, 0.209, 0.24, 0.3, 0.4, 0.54, 0.7, 1.001]
        eta = [0.86, 0.91, 0.928, 0.943, 0.949, 0.95, 0.954, 0.96, 0.965, 0.97, 0.973, 0.977]
        
        # Create interpolation function
        f_eta = interp1d(relative_performance, eta, kind='linear', fill_value='extrapolate')
        
        # Calculate efficiency at current load
        load_fraction = min(max(P_ac / P_nominal, 0.0), 1.0)
        eta_interp = float(f_eta(load_fraction))
        eta_interp = max(0.86, min(0.977, eta_interp))  # Limit to realistic range
        
        # Power loss
        P_electronics = P_ac * (1 - eta_interp)
        
        return P_electronics
    
    def power_dc(self, P_ac):
        """
        Calculate DC power after rectifier losses.
        
        Parameters
        ----------
        P_ac : float
            AC power input in kW.
        
        Returns
        -------
        float
            DC power in kW.
        """
        P_loss = self.power_electronics(self.P_nominal, P_ac)
        return P_ac - P_loss
    
    def gas_drying(self, H2_mass_flow_rate):
        """
        Calculate power consumption for gas drying (TSA - Temperature Swing Adsorption).
        
        The produced hydrogen leaves the stack in a hydrogen-water vapor mixture.
        A TSA system with two adsorption beds is used for drying, where one bed
        adsorbs while the other regenerates at elevated temperature.
        
        Parameters
        ----------
        H2_mass_flow_rate : float
            Hydrogen mass flow rate in kg/h.
        
        Returns
        -------
        float
            Gas drying power consumption in kW.
        
        Notes
        -----
        Model based on master thesis:
        P_HZ = cp_H2 * M_H2 * n * dT + Q_des
        
        - cp_H2 = 14300 J/(kg*K): Specific heat capacity of H2
        - M_H2 = 2.016e-3 kg/mol: Molar mass of H2
        - Regeneration temperature: 300°C
        - Ambient temperature: 20°C
        - Desorption enthalpy: 48600 J/mol
        - Moisture content: X_in=0.1, X_out=1.0 mol H2O/mol H2
        """
        if H2_mass_flow_rate <= 0:
            return 0.0
        
        # Physical constants
        M_H2 = 2.016e-3   # Molar mass of H2 [kg/mol]
        cp_H2 = 14300     # Specific heat capacity of H2 [J/(kg*K)]
        
        # Convert mass flow rate to molar flow rate [mol/s]
        # H2_mass_flow_rate is in kg/h, convert to kg/s then to mol/s
        n_H2 = (H2_mass_flow_rate / 3600) / M_H2  # [mol/s]
        
        # Moisture content [mol H2O / mol H2]
        X_in = 0.1   # After electrolysis, entering 1st bed
        X_out = 1.0  # After regeneration bed (saturated)
        
        # Desorption gas stream [mol/s]
        n_desorption = (X_in / (X_out - X_in)) * n_H2  # [mol/s]
        
        # Temperature difference for regeneration [K]
        dT = 300 - 20  # Regeneration temp - Ambient temp
        
        # Heating power [W]
        P_hz = cp_H2 * M_H2 * n_desorption * dT  # [W]
        
        # Desorption enthalpy [W]
        Q_des = 48600 * n_desorption  # [W] (48600 J/mol)
        
        # Total gas drying power [kW]
        P_gasdrying = (P_hz + Q_des) / 1000  # [kW]
        
        return P_gasdrying
    
    def compression(self, H2_mass_flow_rate):
        """
        Calculate power consumption for hydrogen compression.
        
        Uses isentropic compression model with compressibility factor.
        
        Parameters
        ----------
        H2_mass_flow_rate : float
            Hydrogen mass flow rate in kg/h.
        
        Returns
        -------
        float
            Compression power in kW.
        
        Notes
        -----
        Assumes compression from 30 bar (electrolyser outlet) to target pressure.
        Uses isentropic efficiency of 75% for reciprocating compressor.
        """
        if self.compression_pressure == 0:
            return 0.0
        
        # Convert mass flow rate to kg/s
        m_dot = H2_mass_flow_rate / 3600  # [kg/s]
        
        # Compressor inlet conditions
        T_in = self.temperature + 273.15  # [K]
        p_in = 30  # [bar] - electrolyser outlet pressure
        p_out = self.compression_pressure  # [bar]
        
        # Gas properties
        Z = 0.95  # Compressibility factor for H2
        k = 1.4   # Specific heat ratio (cp/cv) for H2
        R_specific = self.R / (self.M / 1000)  # [J/(kg*K)]
        
        # Isentropic compression work
        # w = k/(k-1) * R * T * Z * [(p2/p1)^((k-1)/k) - 1]
        pressure_ratio = p_out / p_in
        exponent = (k - 1) / k
        
        w_isentropic = (k / (k - 1)) * R_specific * T_in * Z * (pressure_ratio**exponent - 1)  # [J/kg]
        
        # Isentropic efficiency
        eta_isentropic = 0.75
        
        # Actual compression work
        w_actual = w_isentropic / eta_isentropic  # [J/kg]
        
        # Compression power
        P_compression = m_dot * w_actual / 1000  # [kW]
        
        return P_compression
    
    def heat_stack(self, P_dc):
        """
        Calculate heat generated in the stack.
        
        Parameters
        ----------
        P_dc : float
            DC power input in kW.
        
        Returns
        -------
        float
            Stack heat generation in kW.
        
        Notes
        -----
        Heat is generated due to the difference between cell voltage
        and thermoneutral voltage: Q = (U_cell - U_th) * I * n_cells
        """
        I = self.calculate_cell_current(P_dc)
        U_cell = self.calc_cell_voltage(I, self.temperature)
        
        # Heat per stack
        Q_stack = self.n_stacks * self.n_cells * (U_cell - self.E_th_0) * I / 1000  # [kW]
        
        return Q_stack
    
    def heat_system(self, Q_stack, H2O_mass_flow_rate):
        """
        Calculate system heat balance.
        
        Parameters
        ----------
        Q_stack : float
            Stack heat generation in kW.
        H2O_mass_flow_rate : float
            Water mass flow rate in kg/h.
        
        Returns
        -------
        float
            Net system heat to be removed in kW.
        """
        # Specific heat capacity of water [kWh/(kg*K)] * 60 for per hour
        c_p_H2O = 0.001162  # kW*h/(kg*K)
        
        # Temperature difference (operating - ambient)
        dT = self.temperature - self.T_ambient  # [K]
        
        # Heat absorbed by fresh water
        # Convert mass flow rate to kg/min for consistency
        Q_H2O = c_p_H2O * (H2O_mass_flow_rate / 60) * dT * 60  # [kW]
        
        # Net heat to be removed
        Q_system = Q_stack - Q_H2O
        
        return Q_system
    
    def calc_cooling_water_flow(self, Q_system):
        """
        Calculate required cooling water flow rate.
        
        Parameters
        ----------
        Q_system : float
            Heat to be removed in kW.
        
        Returns
        -------
        float
            Cooling water mass flow rate in kg/h.
        """
        # Specific heat capacity of water
        c_p_H2O = 4.186  # [kJ/(kg*K)]
        
        # Cooling water temperature rise (typically 10°C)
        dT_cooling = 10  # [K]
        
        # Mass flow rate [kg/s]
        if Q_system <= 0:
            return 0.0
        
        m_dot_cooling = (Q_system * 1000) / (c_p_H2O * dT_cooling)  # [kg/s] -> Note: Q in kW = kJ/s
        
        # Convert to kg/h
        return m_dot_cooling * 3600
    
    def calc_pump_power(self, H2O_mass_flow_rate, P_in):
        """
        Calculate pump power consumption for water circulation.
        
        Two main pumps are modeled:
        1. Fresh water supply pump (high pressure: 20 bar system pressure)
        2. Cooling water pump (pressure drop only)
        
        Parameters
        ----------
        H2O_mass_flow_rate : float
            Process water mass flow rate in kg/h.
        P_in : float
            Input power in kW.
        
        Returns
        -------
        tuple
            (total_pump_power, fresh_water_pump, cooling_water_pump) in kW.
        
        Notes
        -----
        Pump efficiency curve based on centrifugal pump characteristics.
        Maximum efficiency of 80% at nominal load.
        Pressure drop characteristics from literature [22].
        
        Formula: P_pump = V_dot * dp * (1 - eta_pump)
        Source: https://doi.org/10.1007/978-3-642-40032-2
        """
        # Pump efficiency curve (centrifugal pump)
        relative_performance_pump = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
        eta_pump = [0.627, 0.644, 0.661, 0.677, 0.691, 0.704, 0.715, 0.738, 0.754, 0.769, 0.782, 0.792, 0.797, 0.80]
        
        f_eta_pump = interp1d(relative_performance_pump, eta_pump, kind='linear', fill_value='extrapolate')
        
        # Pressure drop curve [Pa]
        relative_performance_pressure = [0.0, 0.02, 0.07, 0.12, 0.16, 0.2, 0.25, 0.32, 0.36, 0.4, 0.47, 0.54, 0.59,
                                         0.63, 0.67, 0.71, 0.74, 0.77, 0.8, 0.83, 0.86, 0.89, 0.92, 0.95, 0.98, 1.01]
        dt_pressure = [0.0, 330, 1870, 3360, 5210, 8540, 12980, 21850, 27020, 32930, 44000, 59500, 70190, 80520, 90850,
                       100810, 110400, 119990, 128840, 138420, 148010, 158330, 169760, 181190, 191890, 200000]
        
        f_dt_pressure = interp1d(relative_performance_pressure, dt_pressure, kind='linear', fill_value='extrapolate')
        
        # Calculate load fraction
        P_stack = self.stack_nominal_power()
        load_fraction = min(max(P_in / P_stack, 0.0), 1.0)
        
        # Get interpolated values
        eta_interp_pump = float(f_eta_pump(load_fraction))
        dt_interp_pressure = float(f_dt_pressure(load_fraction))  # [Pa]
        
        # Water density [kg/m³]
        rho_water = 997
        
        # Volume flow rate [m³/s]
        vfr_H2O = (H2O_mass_flow_rate / rho_water) / 3600  # [m³/s]
        
        # Fresh water pump: 20 bar system pressure
        P_pump_fresh = vfr_H2O * 2000000 * (1 - eta_interp_pump) / 1000  # [kW]
        
        # Cooling water pump: pressure drop only
        P_pump_cool = vfr_H2O * dt_interp_pressure * (1 - eta_interp_pump) / 1000  # [kW]
        
        # Total pump power
        P_pump_total = P_pump_fresh + P_pump_cool
        
        return P_pump_total, P_pump_fresh, P_pump_cool
    
    def calculate_efficiency(self, P_ac):
        """
        Calculate the overall system efficiency at a given power input.
        
        Parameters
        ----------
        P_ac : float
            AC power input in kW.
        
        Returns
        -------
        float
            System efficiency as a fraction (0-1).
        
        Notes
        -----
        Efficiency = (H2 energy output) / (Total energy input)
        Where total energy input includes auxiliaries (pumps, gas drying, etc.)
        """
        if P_ac < self.P_min:
            return 0.0
        
        # Hydrogen production
        H2_mfr = self.calc_H2_mass_flow_rate(P_ac)  # [kg/h]
        
        # Energy content of hydrogen (using LHV)
        E_H2 = H2_mfr * self.lhv  # [kWh/h] = [kW]
        
        # Auxiliary power consumption
        H2O_mfr = self.calc_H2O_mass_flow_rate(H2_mfr)
        P_gas_drying = self.gas_drying(H2_mfr)
        P_pump, _, _ = self.calc_pump_power(H2O_mfr, P_ac)
        P_compression = self.compression(H2_mfr)
        
        # Total power input
        P_total = P_ac + P_pump + P_gas_drying + P_compression
        
        # Efficiency
        eta = E_H2 / P_total if P_total > 0 else 0.0
        
        return eta
    
    def generate_efficiency_curve(self, n_points=20):
        """
        Generate an efficiency curve over the operating range.
        
        Parameters
        ----------
        n_points : int, optional
            Number of data points. Default is 20.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'load_fraction': Array of load fractions (0-1)
            - 'efficiency': Array of efficiencies (0-1)
            - 'power_kW': Array of power inputs [kW]
            - 'H2_production_kg_h': Array of H2 production rates [kg/h]
        
        Examples
        --------
        >>> electrolyser = PEMElectrolyser(500, 15, 'min')
        >>> curve = electrolyser.generate_efficiency_curve()
        >>> print(curve['efficiency'])
        """
        # Power range from minimum to maximum
        P_range = np.linspace(self.P_min, self.P_max, n_points)
        
        # Calculate efficiency at each power level
        efficiencies = []
        H2_production = []
        
        for P in P_range:
            eta = self.calculate_efficiency(P)
            H2 = self.calc_H2_mass_flow_rate(P)
            efficiencies.append(eta)
            H2_production.append(H2)
        
        return {
            'load_fraction': P_range / self.P_max,
            'efficiency': np.array(efficiencies),
            'power_kW': P_range,
            'H2_production_kg_h': np.array(H2_production)
        }
    
    def plot_efficiency_curve(self, save_path=None):
        """
        Plot the efficiency curve.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure. If None, the figure is displayed.
        
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        data = self.generate_efficiency_curve(n_points=50)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot efficiency
        color1 = 'tab:blue'
        ax1.set_xlabel('Load Fraction [-]')
        ax1.set_ylabel('Efficiency [-]', color=color1)
        ax1.plot(data['load_fraction'], data['efficiency'], color=color1, linewidth=2, label='Efficiency')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Plot H2 production on secondary axis
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('H2 Production [kg/h]', color=color2)
        ax2.plot(data['load_fraction'], data['H2_production_kg_h'], color=color2, linewidth=2, linestyle='--', label='H2 Production')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title(f'PEM Electrolyser Efficiency Curve (P_nom = {self.P_nominal} kW)')
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def generate_default_efficiency_curve():
    """
    Generate the default efficiency curve used in electrolyserlib.
    
    This function creates a PEM electrolyser with typical parameters
    and generates an efficiency curve that can be used for hydrogen
    production calculations.
    
    Returns
    -------
    list
        List of tuples (load_fraction, efficiency) for use in electrolyserlib.
    
    Examples
    --------
    >>> curve = generate_default_efficiency_curve()
    >>> print(curve[:3])
    [(0.1, 0.65), (0.15, 0.68), (0.2, 0.70)]
    """
    # Create electrolyser with typical parameters
    electrolyser = PEMElectrolyser(
        nominal_power=500,      # 500 kW (single stack)
        timestep=15,            # 15 minutes
        timestep_unit='min',    # Minutes
        compression_pressure=0  # No compression (included in BoP)
    )
    
    # Generate efficiency curve
    data = electrolyser.generate_efficiency_curve(n_points=20)
    
    # Format as list of tuples for electrolyserlib
    curve = list(zip(data['load_fraction'], data['efficiency']))
    
    return curve


def print_example_results():
    """
    Print example results from the PEM electrolyser model.
    
    This function demonstrates the model capabilities by calculating
    key parameters for a typical electrolyser configuration.
    """
    print("=" * 60)
    print("PEM Electrolyser Model - Example Results")
    print("=" * 60)
    
    # Create electrolyser
    electrolyser = PEMElectrolyser(
        nominal_power=500,
        timestep=15,
        timestep_unit='min',
        compression_pressure=0
    )
    
    print(f"\nStack Configuration:")
    print(f"  - Nominal Power: {electrolyser.P_nominal} kW")
    print(f"  - Number of Cells: {electrolyser.n_cells}")
    print(f"  - Cell Area: {electrolyser.cell_area} cm²")
    print(f"  - Operating Temperature: {electrolyser.temperature}°C")
    print(f"  - Number of Stacks: {electrolyser.n_stacks}")
    
    print(f"\nPerformance at Different Load Points:")
    print("-" * 60)
    print(f"{'Load [%]':<12} {'Power [kW]':<12} {'H2 [kg/h]':<12} {'Efficiency [%]':<15}")
    print("-" * 60)
    
    for load in [0.2, 0.4, 0.6, 0.8, 1.0]:
        P = load * electrolyser.P_nominal
        H2 = electrolyser.calc_H2_mass_flow_rate(P)
        eta = electrolyser.calculate_efficiency(P)
        print(f"{load*100:<12.0f} {P:<12.1f} {H2:<12.2f} {eta*100:<15.1f}")
    
    print("-" * 60)
    
    # Show efficiency curve
    print("\nEfficiency Curve (for electrolyserlib):")
    curve = generate_default_efficiency_curve()
    print("load_fraction, efficiency")
    for load, eta in curve[::4]:  # Print every 4th point
        print(f"  {load:.2f}, {eta:.3f}")
    
    print("\n" + "=" * 60)


# Run example when script is executed directly
if __name__ == "__main__":
    print_example_results()
    
    # Create electrolyser and plot curves
    electrolyser = PEMElectrolyser(500, 15, 'min')
    
    print("\nGenerating plots...")
    electrolyser.plot_polarization_curve()
    electrolyser.plot_efficiency_curve()
    plt.show()
