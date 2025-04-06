import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List


@dataclass
class ChilledWaterRoomParameters:
    length: float  # meters
    breadth: float  # meters
    height: float  # meters
    current_temp: float  # °C
    target_temp: float  # °C
    external_temp: float = 35.0  # °C
    wall_insulation: str = "medium"  # Insulation level (low/medium/high)
    humidity: float = 50.0  # Target humidity (%)
    num_people: int = 0  # Number of people in room
    # External heat gain in Watts (sunlight, appliances)
    heat_gain_external: float = 0.0
    mode: str = "cooling"  # cooling/heating mode
    fan_coil_units: int = 1  # Number of fan coil units in the room


@dataclass
class ChilledWaterHVACParameters:
    power: float  # kW
    cop: float = 3.0  # Coefficient of Performance
    air_flow_rate: float = 0.5  # m³/s
    supply_temp: float = 12.0  # °C - Air supply temperature
    fan_speed: float = 100.0  # Fan speed percentage
    time_interval: float = 1.0  # Simulation update interval in seconds

    # Chilled water system specific parameters
    chilled_water_flow_rate: float = 0.5  # L/s
    chilled_water_supply_temp: float = 7.0  # °C
    chilled_water_return_temp: float = 12.0  # °C
    pump_power: float = 0.75  # kW
    # True if using primary/secondary loop configuration
    primary_secondary_loop: bool = True
    glycol_percentage: float = 0  # Percentage of glycol in water (0-100)
    # Water-to-air heat exchanger efficiency
    heat_exchanger_efficiency: float = 0.85


class ChilledWaterSystemSimulator:
    def __init__(self, room: ChilledWaterRoomParameters, hvac: ChilledWaterHVACParameters):
        self.room = room
        self.hvac = hvac
        self.specific_heat_air = 1.005  # kJ/kg·K
        self.air_density = 1.225  # kg/m³
        self.debug_info = []  # To store debug information during calculations

        # Chilled water properties
        self.specific_heat_water = 4.18  # kJ/kg·K
        self.water_density = 1000  # kg/m³
        self.initialize_water_properties()

        # Fan power parameters
        self.fan_power_factor = 0.06  # kW per m³/s of air flow at full speed

    def initialize_water_properties(self):
        """Initialize water properties based on glycol percentage."""
        # Adjust specific heat and density based on glycol percentage
        if self.hvac.glycol_percentage > 0:
            # More accurate correction factors for glycol-water mixture
            glycol_factor = self.hvac.glycol_percentage / 100

            # Specific heat decreases with glycol concentration
            # More accurate formula based on glycol concentration
            self.specific_heat_water = 4.18 * (1 - glycol_factor * 0.6)

            # Density increases with glycol concentration
            self.water_density = 1000 * (1 + glycol_factor * 0.15)

            # Adjust COP if glycol is used (glycol reduces efficiency)
            if self.hvac.glycol_percentage > 20:
                self.hvac.cop *= 0.95  # 5% penalty for high glycol concentration

    @property
    def room_volume(self) -> float:
        """Calculate room volume in cubic meters."""
        return self.room.length * self.room.breadth * self.room.height

    @property
    def room_air_mass(self) -> float:
        """Calculate mass of air in the room."""
        return self.room_volume * self.air_density

    def calculate_cooling_capacity(self, at_temp=None) -> float:
        """Calculate cooling capacity in Watts with improved chilled water system logic."""
        temp = at_temp if at_temp is not None else self.room.current_temp

        # Calculate theoretical maximum capacity based on rated power
        rated_capacity = self.hvac.power * 1000  # Convert kW to Watts

        # Calculate capacity based on air-side heat transfer
        # Q = m * cp * ΔT (mass flow rate * specific heat * temperature difference)
        air_mass_flow = self.hvac.air_flow_rate * self.air_density
        airflow_capacity = (
            air_mass_flow
            * self.specific_heat_air
            * 1000  # Convert to Watts
            * abs(temp - self.hvac.supply_temp)
        )

        # Calculate capacity based on water-side heat transfer
        water_capacity = self.calculate_water_heat_capacity()

        # Log capacities for debugging
        self.debug_info.append({
            "rated_capacity": rated_capacity,
            "airflow_capacity": airflow_capacity,
            "water_capacity": water_capacity
        })

        # For chilled water systems, consider both coil and water limitations
        # Water capacity is already adjusted for flow rate and temperatures
        heat_exchanger_efficiency = self.hvac.heat_exchanger_efficiency

        # Calculate the actual capacity based on the limiting factor
        if self.hvac.primary_secondary_loop:
            # With primary/secondary loop, water capacity is decoupled from coil capacity
            coil_capacity = min(rated_capacity, airflow_capacity)
            effective_capacity = min(
                coil_capacity, water_capacity * heat_exchanger_efficiency)
        else:
            # With direct loop, water capacity directly limits the coil capacity
            effective_capacity = min(
                rated_capacity, airflow_capacity, water_capacity * heat_exchanger_efficiency)

        # Scale by fan speed for the air-side delivery
        # Cubic relationship between fan speed and capacity (airflow is roughly cubic with fan speed)
        fan_factor = (self.hvac.fan_speed / 100.0) ** 0.67
        effective_capacity = effective_capacity * fan_factor

        # Scale by number of fan coil units
        effective_capacity = effective_capacity * self.room.fan_coil_units

        # In heating mode, reverse the sign of the capacity
        if self.room.mode.lower() == "heating":
            # In heating mode, we're using hot water instead of chilled water
            # Typically heating capacity is higher than cooling capacity for same system
            # Heating is typically 20% more efficient
            effective_capacity = effective_capacity * 1.2

        return effective_capacity

    def calculate_heat_gain(self, at_temp=None) -> float:
        """Calculate total heat gain in Watts with improved thermal calculations."""
        temp = at_temp if at_temp is not None else self.room.current_temp

        # Calculate wall area and floor/ceiling area separately
        wall_area = 2 * (self.room.length * self.room.height +
                         self.room.breadth * self.room.height)
        floor_ceiling_area = 2 * (self.room.length * self.room.breadth)

        # Get insulation factor based on level - more accurate values
        insulation_factors = {
            "low": 1.2,     # U-value of ~1.2 W/m²K
            "medium": 0.6,  # U-value of ~0.6 W/m²K
            "high": 0.3     # U-value of ~0.3 W/m²K
        }

        if isinstance(self.room.wall_insulation, str):
            insulation_factor = insulation_factors.get(
                self.room.wall_insulation.lower(), 0.6)
        else:
            insulation_factor = 0.6  # Default insulation factor if invalid

        # Different insulation properties for floor/ceiling
        floor_ceiling_factor = insulation_factor * 0.8  # Typically better insulated

        # Calculate environmental heat gain
        temperature_difference = self.room.external_temp - temp
        wall_heat_gain = wall_area * insulation_factor * temperature_difference
        floor_ceiling_heat_gain = floor_ceiling_area * \
            floor_ceiling_factor * temperature_difference
        environmental_gain = wall_heat_gain + floor_ceiling_heat_gain

        # Add heat from people (approximately 115W per person at rest)
        people_heat = self.room.num_people * 115

        # Add external heat gain (from sunlight, appliances, etc)
        total_heat_gain = environmental_gain + \
            people_heat + self.room.heat_gain_external

        return total_heat_gain

    def calculate_energy_consumption(self, cooling_capacity: float) -> float:
        """Calculate energy consumption in Watts with improved modeling and hard limits."""
        # Account for part-load efficiency
        part_load_factor = min(
            1.0, abs(cooling_capacity) / (self.hvac.power * 1000))

        # At part load, COP is often better than at full load
        adjusted_cop = self.hvac.cop * (1 + 0.1 * (1 - part_load_factor))

        # Chiller energy consumption (based on cooling capacity and COP)
        # Limit the maximum cooling capacity that a chiller can handle
        max_chiller_capacity = 20000  # 20kW for a typical small commercial chiller
        limited_cooling_capacity = min(
            abs(cooling_capacity), max_chiller_capacity)
        chiller_energy = limited_cooling_capacity / adjusted_cop

        # Calculate pump energy with more realistic constraints
        # Calculate base pump energy
        pump_energy = self.calculate_pump_energy()

        # Practical limit for pump power - typical commercial pumps
        # Even at maximum flow, the pump shouldn't exceed a reasonable percentage of total system power
        max_pump_percentage = 0.25  # Maximum 25% of total energy for pumping
        max_reasonable_pump = self.hvac.power * 1000 * max_pump_percentage

        # Apply a hard cap on pump energy
        pump_energy = min(pump_energy, max_reasonable_pump)

        # Fan energy consumption - using fan affinity laws
        fan_energy = self.calculate_fan_energy()

        # Total energy is the sum of all components
        total_energy = chiller_energy + pump_energy + fan_energy

        # Implement a hard cap on total energy based on equipment size
        # For a typical small commercial chiller system, it would be unrealistic
        # to exceed about 5x the rated power
        absolute_max_energy = self.hvac.power * 1000 * 5
        if total_energy > absolute_max_energy:
            scaling_factor = absolute_max_energy / total_energy
            chiller_energy *= scaling_factor
            pump_energy *= scaling_factor
            fan_energy *= scaling_factor
            total_energy = absolute_max_energy

        # Store components for debugging
        self.debug_info.append({
            "chiller_energy": chiller_energy,
            "pump_energy": pump_energy,
            "fan_energy": fan_energy,
            "adjusted_cop": adjusted_cop,
            "part_load_factor": part_load_factor,
            "limited_cooling_capacity": limited_cooling_capacity
        })

        return total_energy

    def calculate_fan_energy(self) -> float:
        """Calculate fan energy consumption based on fan laws."""
        # Fan power is proportional to the cube of speed
        baseline_fan_power = self.fan_power_factor * \
            self.hvac.air_flow_rate * 1000  # Convert to Watts
        fan_power = baseline_fan_power * (self.hvac.fan_speed / 100.0) ** 3

        # Multiply by number of fan coil units
        fan_power *= self.room.fan_coil_units

        return fan_power

    def calculate_refrigerant_flow(self, cooling_capacity: float) -> float:
        """Calculate refrigerant flow rate in g/s based on refrigerant properties."""
        # This is only relevant for the chiller's refrigeration cycle
        # Not directly used in the room-level chilled water calculations

        # R-410A properties at typical evaporating/condensing temperatures
        # Latent heat of vaporization depends on operating conditions
        evaporating_temp = 5.0  # °C (typical for chilled water system)
        condensing_temp = 45.0  # °C (typical for air-cooled condenser)

        # Enthalpy difference varies with temperatures
        enthalpy_difference = 170  # kJ/kg (more accurate for R-410A)

        # Calculate mass flow rate
        # Convert to g/s
        return (cooling_capacity / 1000) / enthalpy_difference * 1000

    def calculate_net_heat_at_temp(self, temp: float) -> float:
        """Calculate net heat transfer (in Watts) at a specific temperature."""
        # Calculate components at specified temperature
        cooling_capacity = self.calculate_cooling_capacity(at_temp=temp)
        heat_gain = self.calculate_heat_gain(at_temp=temp)

        # Calculate net heat based on mode
        if self.room.mode.lower() == "heating":
            # In heating mode, cooling_capacity is actually heating capacity (positive)
            net_heat = heat_gain + cooling_capacity
        else:  # cooling mode
            # In cooling mode, cooling_capacity is negative (removing heat)
            net_heat = heat_gain - cooling_capacity

        return net_heat

    def calculate_temp_change_rate(self, temp: float) -> float:
        """Calculate rate of temperature change (°C/s) at a specific temperature."""
        net_heat = self.calculate_net_heat_at_temp(temp)

        # Temperature change rate = net heat / (mass * specific heat)
        rate = net_heat / (self.room_air_mass * self.specific_heat_air * 1000)

        return rate

    def calculate_temperature_change(self) -> float:
        """Calculate new room temperature after one time step."""
        # If we're very close to target temperature, set to exact target
        if abs(self.room.current_temp - self.room.target_temp) < 0.05:
            return self.room.target_temp

        # Get temperature change rate
        rate = self.calculate_temp_change_rate(self.room.current_temp)

        # Apply time interval to get temperature change
        temp_change = rate * self.hvac.time_interval

        # Calculate new temperature
        new_temp = self.room.current_temp + temp_change

        # Check if we're approaching or moving away from target
        approaching_target = (
            (self.room.mode.lower() == "cooling" and new_temp < self.room.current_temp) or
            (self.room.mode.lower() == "heating" and new_temp > self.room.current_temp)
        )

        # Handle approaching target to prevent oscillation
        if approaching_target:
            # Prevent overshooting target temperature
            if self.room.mode.lower() == "cooling":
                return max(self.room.target_temp, new_temp)
            else:  # heating mode
                return min(self.room.target_temp, new_temp)
        else:
            # We're moving away from target (system can't reach it)
            # Return new temp to accurately model behavior
            return new_temp

    def can_reach_target(self) -> bool:
        """Determine if the system can reach the target temperature."""
        # Check if we're already at target
        if abs(self.room.current_temp - self.room.target_temp) < 0.1:
            return True

        # Direction we need to move
        cooling_needed = self.room.current_temp > self.room.target_temp
        heating_needed = self.room.current_temp < self.room.target_temp

        # Calculate the rate of temperature change
        current_rate = self.calculate_temp_change_rate(self.room.current_temp)

        # Check if we're moving in the right direction with significant speed
        if (cooling_needed and current_rate < -1e-6) or (heating_needed and current_rate > 1e-6):
            return True

        return False

    def calculate_time_to_target(self) -> float:
        """Calculate time to reach target temperature using numerical integration."""
        # If already at target, return 0
        if abs(self.room.current_temp - self.room.target_temp) < 0.1:
            return 0.0

        # Check if target can be reached
        if not self.can_reach_target():
            return float('inf')

        # Setup for numerical integration
        start_temp = self.room.current_temp
        target_temp = self.room.target_temp
        cooling_mode = self.room.mode.lower() == "cooling"

        # Determine direction of approach
        proper_direction = (cooling_mode and start_temp > target_temp) or (
            not cooling_mode and start_temp < target_temp)
        if not proper_direction:
            # Can't reach target if going in wrong direction
            return float('inf')

        # Clear debug info
        self.debug_info = []

        # Track current progress
        current_temp = start_temp
        total_time = 0.0

        # Use adaptive time step for more accurate integration
        # Start with a small time step and increase as we get closer to equilibrium
        min_time_step = 15.0  # 15 seconds minimum
        max_time_step = 300.0  # 5 minutes maximum
        max_time = 24 * 3600  # 24 hours max simulation

        while total_time < max_time:
            # Calculate rate at current temperature
            rate = self.calculate_temp_change_rate(current_temp)

            # Check if we can still make progress
            if (cooling_mode and rate >= 0) or (not cooling_mode and rate <= 0):
                return float('inf')  # Can't reach target

            # Adaptive time step based on rate of change
            # Faster changes = smaller time step
            time_step = min(
                max(min_time_step, 60.0 / abs(rate)), max_time_step)

            # Calculate temperature change for this time step
            temp_change = rate * time_step
            new_temp = current_temp + temp_change

            # Check if we've reached or passed the target
            if (cooling_mode and new_temp <= target_temp) or (not cooling_mode and new_temp >= target_temp):
                # Interpolate exact time to reach target
                if abs(rate) < 1e-6:
                    return float('inf')  # Prevent division by zero

                remaining_temp_diff = abs(target_temp - current_temp)
                remaining_time = remaining_temp_diff / abs(rate)

                return round(total_time + remaining_time, 2)

            # Update for next iteration
            current_temp = new_temp
            total_time += time_step

            # Store debug info periodically
            if int(total_time) % 300 == 0:  # Every 5 minutes
                self.debug_info.append({
                    "temp": round(current_temp, 2),
                    "rate": round(rate * 3600, 4),  # °C/hour
                    "time_elapsed": round(total_time / 60, 1)  # minutes
                })

        # If we get here, we couldn't reach target within max simulation time
        return float('inf')

    def calculate_water_heat_capacity(self) -> float:
        """Calculate heat transfer capacity of the chilled water in Watts."""
        # Convert L/s to kg/s
        mass_flow_rate = self.hvac.chilled_water_flow_rate * \
            (self.water_density / 1000)

        # Calculate capacity using Q = ṁ × cp × ΔT
        delta_t = abs(self.hvac.chilled_water_return_temp -
                      self.hvac.chilled_water_supply_temp)

        # Adjust for mode (heating vs cooling)
        if self.room.mode.lower() == "heating":
            # For heating, supply temp is higher than return temp
            if self.hvac.chilled_water_supply_temp <= self.hvac.chilled_water_return_temp:
                # Correct the temperatures for heating mode
                delta_t = 30  # Typical hot water delta T

        capacity = mass_flow_rate * self.specific_heat_water * \
            delta_t * 1000  # Convert to Watts

        return capacity

    def calculate_pump_energy(self) -> float:
        """Calculate energy consumption of the chilled water pump in Watts with more realistic scaling."""
        # Base consumption on rated pump power
        base_consumption = self.hvac.pump_power * 1000  # Convert to Watts

        # Flow Rate vs Flow Rate Reference (usually design flow)
        relative_flow = self.hvac.chilled_water_flow_rate / \
            0.5  # Normalized to default flow

        # Advanced pump curve modeling:
        # In real VFD pumps, the power curve is flatter than cubic law suggests
        # We use a more realistic formula: P = P_design * (a * (Q/Q_design) + b * (Q/Q_design)² + c * (Q/Q_design)³)
        # Where a + b + c = 1 and the values are chosen to match real pump curves

        # Fixed power component (even at zero flow, there's some power consumption)
        a = 0.15
        b = 0.35  # Linear component
        c = 0.50  # Cubic component

        # More realistic pump curve
        pump_energy = base_consumption * (
            a +
            b * relative_flow +
            # Changed from cubic to quadratic for more realistic behavior
            c * (relative_flow**2)
        )

        # Apply practical minimum power draw
        min_power = base_consumption * 0.2
        pump_energy = max(min_power, pump_energy)

        # Apply a hard cap based on reasonable size limits
        # No commercial pump would exceed about 2x its rated power
        max_power = base_consumption * 2.0
        pump_energy = min(max_power, pump_energy)

        # Additional efficiency losses from flow restrictions at very high flow rates
        if relative_flow > 3.0:
            # At very high flow rates, efficiency drops dramatically
            # Efficiency drops by 10% per unit above 3x flow
            efficiency_factor = 1.0 - min(0.5, (relative_flow - 3.0) * 0.1)
            pump_energy = pump_energy / efficiency_factor  # Higher number = more energy used

        # Adjust for primary/secondary loop configuration
        if self.hvac.primary_secondary_loop:
            # Primary/secondary loops require additional pumping power
            pump_energy *= 1.2  # 20% more energy for additional pump

        return pump_energy

    def check_for_failures(self) -> dict:
        """Check for potential HVAC system failures based on current conditions."""
        failures = {}

        # Environmental/capacity issues
        room_volume = self.room_volume
        # rough estimate: 120W per cubic meter for chilled water systems
        required_capacity = room_volume * 0.12
        actual_capacity = abs(
            self.calculate_cooling_capacity()) / 1000  # in kW

        # Undersized system
        if actual_capacity < required_capacity * 0.7:
            failures['undersized'] = {
                'probability': 1.0,
                'message': f'System capacity ({actual_capacity:.1f}kW) insufficient for room size ({room_volume:.1f}m³).',
                'severity': 'high',
                'solution': f'Increase system power from {self.hvac.power}kW to at least {required_capacity:.1f}kW to match room size.'
            }

        # Extreme temperature operation
        if self.room.external_temp > 43:
            failures['extreme_temp'] = {
                # scales from 0 to 1 as temp goes from 43 to 50
                'probability': (self.room.external_temp - 43) / 7,
                'message': f'System operating above rated conditions ({self.room.external_temp}°C).',
                'severity': 'medium',
                'solution': 'Lower external temperature setting or increase system power to compensate for extreme conditions.'
            }

        # Calculate net heat to detect if system is struggling
        net_heat = self.calculate_net_heat_at_temp(self.room.current_temp)
        if net_heat > 0 and self.room.mode.lower() == "cooling":
            failures['capacity_exceeded'] = {
                'probability': min(1.0, net_heat / 1000),
                'message': 'Cooling demand exceeds system capacity. Room will not reach target temperature.',
                'severity': 'medium',
                'solution': 'Increase system power, reduce number of people in room, improve wall insulation, or add more fan coil units.'
            }

        # Fan speed too low
        if self.hvac.fan_speed < 30 and self.hvac.power > 2:
            failures['fan_speed_low'] = {
                'probability': 0.9,
                'message': 'Fan speed too low for selected power rating, reducing efficiency.',
                'severity': 'medium',
                'solution': 'Increase fan speed to at least 50% for optimal airflow.'
            }

        # Poor insulation with high external temperature differential
        temp_diff = abs(self.room.external_temp - self.room.current_temp)
        if self.room.wall_insulation.lower() == "low" and temp_diff > 15:
            failures['poor_insulation'] = {
                'probability': 0.85,
                'message': 'Poor insulation causing significant heat transfer with high temperature differential.',
                'severity': 'medium',
                'solution': 'Upgrade wall insulation from low to medium or high to improve efficiency.'
            }

        # Too many people for room size
        people_density = self.room.num_people / \
            (self.room.length * self.room.breadth)
        if people_density > 0.5:  # More than 1 person per 2 square meters
            failures['overcrowding'] = {
                'probability': min(0.9, people_density - 0.3),
                'message': f'High occupant density ({self.room.num_people} people) for room size.',
                'severity': 'medium',
                'solution': f'Reduce number of people, add more fan coil units, or increase system power to handle additional heat load.'
            }

        # Incorrect mode for temperature differential
        if (self.room.mode.lower() == "cooling" and self.room.current_temp < self.room.target_temp) or \
           (self.room.mode.lower() == "heating" and self.room.current_temp > self.room.target_temp):
            failures['incorrect_mode'] = {
                'probability': 1.0,
                'message': f'System mode ({self.room.mode}) opposite to required direction for target temperature.',
                'severity': 'high',
                'solution': f'Change mode from {self.room.mode} to {"heating" if self.room.mode.lower() == "cooling" else "cooling"}.'
            }

        # Chilled water specific failures

        # Water temperature differential too small
        water_delta_t = abs(self.hvac.chilled_water_return_temp -
                            self.hvac.chilled_water_supply_temp)
        if water_delta_t < 3:
            failures['small_water_delta_t'] = {
                'probability': 0.9,
                'message': f'Chilled water temperature differential too small ({water_delta_t}°C).',
                'severity': 'medium',
                'solution': 'Increase temperature differential between supply and return water to 5-7°C for better efficiency.'
            }

        # Water flow rate too low
        if self.hvac.chilled_water_flow_rate < 0.1 and self.hvac.power > 3:
            failures['low_water_flow'] = {
                'probability': 0.95,
                'message': 'Chilled water flow rate too low for system capacity.',
                'severity': 'high',
                'solution': 'Increase water flow rate or check for blockages in the system.'
            }

        # Water flow rate too high
        max_recommended_flow = 0.2 * self.hvac.power  # rough estimate: 0.2 L/s per kW
        if self.hvac.chilled_water_flow_rate > max_recommended_flow * 2:
            failures['high_water_flow'] = {
                'probability': 0.8,
                'message': 'Chilled water flow rate excessively high, causing inefficiency and possible noise.',
                'severity': 'medium',
                'solution': f'Reduce water flow rate to around {max_recommended_flow:.2f} L/s to match system capacity.'
            }

        # Pump power too high for flow rate
        expected_pump_power = 0.1 + 0.2 * \
            self.hvac.chilled_water_flow_rate  # kW, simple relationship
        if self.hvac.pump_power > expected_pump_power * 2:
            failures['oversized_pump'] = {
                'probability': 0.7,
                'message': 'Pump power too high for current water flow rate.',
                'severity': 'low',
                'solution': f'Consider using a smaller pump or reducing pump speed for better energy efficiency.'
            }

        # Glycol percentage too high
        if self.hvac.glycol_percentage > 40:
            failures['high_glycol'] = {
                'probability': 0.8,
                'message': f'Glycol percentage ({self.hvac.glycol_percentage}%) higher than necessary, reducing heat transfer efficiency.',
                'severity': 'low',
                'solution': 'Reduce glycol percentage to 25-30% if freeze protection is needed, or to 0% if not required.'
            }

        # Heat exchanger efficiency low
        if self.hvac.heat_exchanger_efficiency < 0.7:
            failures['inefficient_heat_exchanger'] = {
                'probability': 0.9,
                'message': f'Heat exchanger efficiency ({self.hvac.heat_exchanger_efficiency*100:.1f}%) is below optimal range.',
                'severity': 'medium',
                'solution': 'Check heat exchanger for fouling or air buildup and perform maintenance if necessary.'
            }

        # Too few fan coil units for room size
        area_per_fcu = (self.room.length * self.room.breadth) / \
            self.room.fan_coil_units
        if area_per_fcu > 50:  # More than 50m² per fan coil unit
            failures['insufficient_fcu'] = {
                'probability': min(0.9, area_per_fcu / 100),
                'message': f'Too few fan coil units ({self.room.fan_coil_units}) for room size ({self.room.length * self.room.breadth:.1f}m²).',
                'severity': 'medium',
                'solution': f'Add more fan coil units to improve air distribution and system effectiveness.'
            }

        # Chilled water supply temperature too high for cooling mode
        if self.room.mode.lower() == "cooling" and self.hvac.chilled_water_supply_temp > 10:
            failures['high_chw_temp'] = {
                'probability': min(1.0, (self.hvac.chilled_water_supply_temp - 10) / 5),
                'message': f'Chilled water supply temperature ({self.hvac.chilled_water_supply_temp}°C) too high for effective cooling.',
                'severity': 'medium',
                'solution': 'Lower chilled water supply temperature to 5-7°C for better dehumidification and cooling performance.'
            }

        # Primary/secondary loop mismatch with system size
        if not self.hvac.primary_secondary_loop and self.hvac.power > 15:
            failures['missing_secondary_loop'] = {
                'probability': 0.8,
                'message': 'Large system without primary/secondary loop configuration may experience flow distribution issues.',
                'severity': 'medium',
                'solution': 'Enable primary/secondary loop for better flow control in large systems.'
            }

        return failures

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and calculations for chilled water system."""
        cooling_capacity = self.calculate_cooling_capacity()
        energy_consumption = self.calculate_energy_consumption(
            cooling_capacity)
        water_capacity = self.calculate_water_heat_capacity()
        pump_energy = self.calculate_pump_energy()
        fan_energy = self.calculate_fan_energy()
        time_to_target = self.calculate_time_to_target()
        refrigerant_flow = self.calculate_refrigerant_flow(cooling_capacity)
        failures = self.check_for_failures()
        active_failures = {k: v for k,
                           v in failures.items() if v['probability'] > 0.5}

        chiller_energy = energy_consumption - pump_energy - fan_energy

        energy_components = {
            "chiller": abs(cooling_capacity) / self.hvac.cop,
            "pump": pump_energy,
            "fan": fan_energy,
        }

        energy_consumption = sum(energy_components.values())

        absolute_maximum = 75000  # 75kW is reasonable max for small commercial systems
        if energy_consumption > absolute_maximum:
            scaling_factor = absolute_maximum / energy_consumption
            energy_components = {
                k: v * scaling_factor for k, v in energy_components.items()}
            energy_consumption = absolute_maximum

        # Calculate actual water flow rate based on current load
        # In a real system, water flow varies with load to maintain delta T
        load_factor = min(1.0, abs(cooling_capacity) /
                          (self.hvac.power * 1000))

        # Modulate water flow based on load with a minimum flow rate
        # This simulates the variable flow in the secondary loop
        min_flow_rate = 0.1 if self.hvac.chilled_water_flow_rate > 0.1 else 0.05
        actual_flow_rate = max(
            min_flow_rate,
            self.hvac.chilled_water_flow_rate * (0.3 + 0.7 * load_factor)
        )

        # If the system is running, use calculated flow; otherwise use the parameter value
        # This ensures flow displays as 0 when system is off
        display_flow_rate = actual_flow_rate if cooling_capacity != 0 else 0

        # Calculate actual COP of the system
        system_cop = abs(cooling_capacity) / \
            energy_consumption if energy_consumption > 0 else 0

        status = {
            "room_temperature": round(self.room.current_temp, 2),
            "target_temperature": self.room.target_temp,
            "cooling_capacity_kw": round(cooling_capacity / 1000, 2),
            "cooling_capacity_btu": round(cooling_capacity * 3.412, 2),
            "energy_consumption_w": round(energy_consumption, 2),
            "chiller_consumption_w": round(energy_components["chiller"], 2),
            "pump_consumption_w": round(energy_components["pump"], 2),
            "fan_consumption_w": round(energy_components["fan"], 2),
            "water_flow_rate_ls": round(display_flow_rate, 2),
            "water_supply_temp": self.hvac.chilled_water_supply_temp,
            "water_return_temp": self.hvac.chilled_water_return_temp,
            "water_heat_capacity_kw": round(water_capacity / 1000, 2),
            "heat_gain_w": round(self.calculate_heat_gain(), 2),
            "rated_cop": self.hvac.cop,
            "system_cop": round(system_cop, 2),
            "mode": self.room.mode,
            "fan_speed": self.hvac.fan_speed,
            "humidity": self.room.humidity,
            "num_people": self.room.num_people,
            "external_heat_gain": self.room.heat_gain_external,
            "insulation_level": self.room.wall_insulation,
            "glycol_percentage": self.hvac.glycol_percentage,
            "fan_coil_units": self.room.fan_coil_units,
            "time_interval": self.hvac.time_interval,
            "room_volume": round(self.room_volume, 2),
            "room_floor_area": round(self.room.length * self.room.breadth, 2),
            "external_temperature": self.room.external_temp,
            "time_to_target": time_to_target if time_to_target != float('inf') else "Cannot reach target",
            "can_reach_target": self.can_reach_target(),
            # °C/hour
            "temp_change_rate_per_hour": round(self.calculate_temp_change_rate(self.room.current_temp) * 3600, 4),
            "rated_power_kw": self.hvac.power,
            "primary_secondary_loop": self.hvac.primary_secondary_loop,
            "heat_exchanger_efficiency": self.hvac.heat_exchanger_efficiency,
            "refrigerant_flow_gs": round(refrigerant_flow, 2),
            "failures": failures,
            "active_failures": active_failures,
            "has_critical_failure": any(f['severity'] == 'high' and f['probability'] > 0.7 for f in failures.values()),
            "warnings": [f['message'] for f in failures.values() if 0.3 < f['probability'] <= 0.7],
            "critical_alerts": [f['message'] for f in failures.values() if f['probability'] > 0.7]

        }

        return status

    def simulate_until_target(self, max_time_seconds=7200) -> Dict[str, Any]:
        """Run simulation until target temperature is reached or max time elapsed."""
        # Save original state to restore later
        original_temp = self.room.current_temp
        original_time_interval = self.hvac.time_interval

        # Use a reasonable time interval for simulation
        self.hvac.time_interval = 60  # 1 minute per step

        # Initialize simulation data
        current_time = 0
        temperature_history = [(0, self.room.current_temp)]
        energy_consumption = 0

        while current_time < max_time_seconds:
            # Calculate current status
            status = self.get_system_status()

            # Check if we've reached target
            if abs(self.room.current_temp - self.room.target_temp) < 0.1:
                break

            # Update room temperature
            self.room.current_temp = self.calculate_temperature_change()

            # Track energy consumption
            energy_consumption += status["energy_consumption_w"] * \
                self.hvac.time_interval / 3600  # Wh

            # Update time and record history
            current_time += self.hvac.time_interval
            temperature_history.append((current_time, self.room.current_temp))

            # Check if we can reach target
            if not self.can_reach_target():
                break

        # Prepare simulation results
        result = {
            "final_temperature": round(self.room.current_temp, 2),
            "target_reached": abs(self.room.current_temp - self.room.target_temp) < 0.1,
            "time_elapsed_seconds": current_time,
            "time_elapsed_minutes": round(current_time / 60, 1),
            "energy_consumed_wh": round(energy_consumption, 2),
            "energy_consumed_kwh": round(energy_consumption / 1000, 3),
            "temperature_history": temperature_history
        }

        # Restore original state
        self.room.current_temp = original_temp
        self.hvac.time_interval = original_time_interval

        return result
