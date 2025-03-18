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
    heat_gain_external: float = 0.0  # External heat gain in Watts (sunlight, appliances)
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
    primary_secondary_loop: bool = True  # True if using primary/secondary loop configuration
    glycol_percentage: float = 0  # Percentage of glycol in water (0-100)
    heat_exchanger_efficiency: float = 0.85  # Water-to-air heat exchanger efficiency

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
            effective_capacity = min(coil_capacity, water_capacity * heat_exchanger_efficiency)
        else:
            # With direct loop, water capacity directly limits the coil capacity
            effective_capacity = min(rated_capacity, airflow_capacity, water_capacity * heat_exchanger_efficiency)
        
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
            effective_capacity = effective_capacity * 1.2  # Heating is typically 20% more efficient
        
        return effective_capacity

    def calculate_heat_gain(self, at_temp=None) -> float:
        """Calculate total heat gain in Watts with improved thermal calculations."""
        temp = at_temp if at_temp is not None else self.room.current_temp
        
        # Calculate wall area and floor/ceiling area separately
        wall_area = 2 * (self.room.length * self.room.height + self.room.breadth * self.room.height)
        floor_ceiling_area = 2 * (self.room.length * self.room.breadth)
        
        # Get insulation factor based on level - more accurate values
        insulation_factors = {
            "low": 1.2,     # U-value of ~1.2 W/m²K
            "medium": 0.6,  # U-value of ~0.6 W/m²K
            "high": 0.3     # U-value of ~0.3 W/m²K
        }
        
        if isinstance(self.room.wall_insulation, str):
            insulation_factor = insulation_factors.get(self.room.wall_insulation.lower(), 0.6)
        else:
            insulation_factor = 0.6  # Default insulation factor if invalid
        
        # Different insulation properties for floor/ceiling
        floor_ceiling_factor = insulation_factor * 0.8  # Typically better insulated
        
        # Calculate environmental heat gain
        temperature_difference = self.room.external_temp - temp
        wall_heat_gain = wall_area * insulation_factor * temperature_difference
        floor_ceiling_heat_gain = floor_ceiling_area * floor_ceiling_factor * temperature_difference
        environmental_gain = wall_heat_gain + floor_ceiling_heat_gain
        
        # Add heat from people (approximately 115W per person at rest)
        people_heat = self.room.num_people * 115
        
        # Add external heat gain (from sunlight, appliances, etc)
        total_heat_gain = environmental_gain + people_heat + self.room.heat_gain_external
        
        return total_heat_gain

    def calculate_energy_consumption(self, cooling_capacity: float) -> float:
        """Calculate energy consumption in Watts with improved modeling."""
        # Account for part-load efficiency
        part_load_factor = min(1.0, abs(cooling_capacity) / (self.hvac.power * 1000))
        
        # At part load, COP is often better than at full load
        # This is a simple model - in reality, this curve would be more complex
        adjusted_cop = self.hvac.cop * (1 + 0.1 * (1 - part_load_factor))
        
        # Chiller energy consumption (based on cooling capacity and COP)
        chiller_energy = abs(cooling_capacity) / adjusted_cop
        
        # Add pump energy consumption - quadratic relationship with flow rate
        pump_energy = self.calculate_pump_energy()
        
        # Fan energy consumption - using fan affinity laws
        fan_energy = self.calculate_fan_energy()
        
        total_energy = chiller_energy + pump_energy + fan_energy
        
        # Store components for debugging
        self.debug_info.append({
            "chiller_energy": chiller_energy,
            "pump_energy": pump_energy,
            "fan_energy": fan_energy,
            "adjusted_cop": adjusted_cop,
            "part_load_factor": part_load_factor
        })
        
        return total_energy

    def calculate_fan_energy(self) -> float:
        """Calculate fan energy consumption based on fan laws."""
        # Fan power is proportional to the cube of speed
        baseline_fan_power = self.fan_power_factor * self.hvac.air_flow_rate * 1000  # Convert to Watts
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
        return (cooling_capacity / 1000) / enthalpy_difference * 1000  # Convert to g/s

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
            
        # Direction we need to move (cooling = negative rate, heating = positive rate)
        desired_direction = -1 if self.room.mode.lower() == "cooling" else 1
            
        # Calculate the rate of temperature change
        current_rate = self.calculate_temp_change_rate(self.room.current_temp)
        
        # If rate sign matches desired direction, we're making progress
        # Also check if rate is significant enough (not practically zero)
        return (current_rate * desired_direction) < -1e-6

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
        proper_direction = (cooling_mode and start_temp > target_temp) or (not cooling_mode and start_temp < target_temp)
        if not proper_direction:
            return float('inf')  # Can't reach target if going in wrong direction
        
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
            time_step = min(max(min_time_step, 60.0 / abs(rate)), max_time_step)
                
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
        mass_flow_rate = self.hvac.chilled_water_flow_rate * (self.water_density / 1000)
        
        # Calculate capacity using Q = ṁ × cp × ΔT
        delta_t = abs(self.hvac.chilled_water_return_temp - self.hvac.chilled_water_supply_temp)
        
        # Adjust for mode (heating vs cooling)
        if self.room.mode.lower() == "heating":
            # For heating, supply temp is higher than return temp
            if self.hvac.chilled_water_supply_temp <= self.hvac.chilled_water_return_temp:
                # Correct the temperatures for heating mode
                delta_t = 30  # Typical hot water delta T
        
        capacity = mass_flow_rate * self.specific_heat_water * delta_t * 1000  # Convert to Watts
        
        return capacity
    
    def calculate_pump_energy(self) -> float:
        """Calculate energy consumption of the chilled water pump in Watts using affinity laws."""
        # Base consumption on rated pump power
        base_consumption = self.hvac.pump_power * 1000  # Convert to Watts
        
        # Scale based on flow rate using pump affinity laws
        # Power is proportional to the cube of flow rate
        relative_flow = self.hvac.chilled_water_flow_rate / 0.5  # Normalized to default flow
        
        # Add system curve effect (system pressure increases with flow squared)
        # This creates a more realistic power curve
        pump_energy = base_consumption * (0.5 * relative_flow**2 + 0.5 * relative_flow**3)
        
        # Adjust for primary/secondary loop configuration
        if self.hvac.primary_secondary_loop:
            # Primary/secondary loops require additional pumping power
            pump_energy *= 1.2  # 20% more energy for additional pump
        
        return pump_energy

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and calculations for chilled water system."""
        cooling_capacity = self.calculate_cooling_capacity()
        energy_consumption = self.calculate_energy_consumption(cooling_capacity)
        water_capacity = self.calculate_water_heat_capacity()
        pump_energy = self.calculate_pump_energy()
        fan_energy = self.calculate_fan_energy()
        time_to_target = self.calculate_time_to_target()
        
        # Calculate actual COP of the system
        system_cop = abs(cooling_capacity) / energy_consumption if energy_consumption > 0 else 0
        
        status = {
            "room_temperature": round(self.room.current_temp, 2),
            "target_temperature": self.room.target_temp,
            "cooling_capacity_kw": round(cooling_capacity / 1000, 2),
            "cooling_capacity_btu": round(cooling_capacity * 3.412, 2),
            "energy_consumption_w": round(energy_consumption, 2),
            "chiller_consumption_w": round(abs(cooling_capacity) / self.hvac.cop, 2),
            "pump_consumption_w": round(pump_energy, 2),
            "fan_consumption_w": round(fan_energy, 2),
            "water_flow_rate_ls": self.hvac.chilled_water_flow_rate,
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
            "time_to_target_minutes": round(time_to_target / 60, 1) if time_to_target != float('inf') else "Cannot reach target",
            "can_reach_target": self.can_reach_target(),
            "temp_change_rate_per_hour": round(self.calculate_temp_change_rate(self.room.current_temp) * 3600, 4),  # °C/hour
            "rated_power_kw": self.hvac.power,
            "primary_secondary_loop": self.hvac.primary_secondary_loop,
            "heat_exchanger_efficiency": self.hvac.heat_exchanger_efficiency
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
            energy_consumption += status["energy_consumption_w"] * self.hvac.time_interval / 3600  # Wh
            
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