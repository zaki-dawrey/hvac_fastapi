import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

@dataclass
class HeatPumpRoomParameters:
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

@dataclass
class HeatPumpHVACParameters:
    power: float  # kW
    cop: float = 3.0  # Coefficient of Performance
    air_flow_rate: float = 0.5  # m³/s
    supply_temp: float = 12.0  # °C
    fan_speed: float = 100.0  # Fan speed percentage
    time_interval: float = 1.0  # Simulation update interval in seconds

class HeatPumpSystemSimulator:
    def __init__(self, room: HeatPumpRoomParameters, hvac: HeatPumpHVACParameters):
        self.room = room
        self.hvac = hvac
        self.specific_heat_air = 1.005  # kJ/kg·K
        self.air_density = 1.225  # kg/m³
        self.debug_info = []  # To store debug information during calculations

    @property
    def room_volume(self) -> float:
        """Calculate room volume in cubic meters."""
        return self.room.length * self.room.breadth * self.room.height

    @property
    def room_air_mass(self) -> float:
        """Calculate mass of air in the room."""
        return self.room_volume * self.air_density

    def calculate_cooling_capacity(self, at_temp=None) -> float:
        """Calculate cooling capacity in Watts with improved logic."""
        temp = at_temp if at_temp is not None else self.room.current_temp
        
        # Calculate theoretical maximum capacity based on rated power
        rated_capacity = self.hvac.power * 1000  # Convert kW to Watts
        
        # Calculate capacity based on air flow and temperature differential
        airflow_capacity = (
            self.hvac.air_flow_rate
            * self.specific_heat_air
            * 1000  # Convert to Watts
            * abs(temp - self.hvac.supply_temp)
        )
        
        # Take the lesser of the two capacities (limiting factor)
        # Adjust for fan speed and apply COP for actual cooling/heating effect
        effective_capacity = min(rated_capacity, airflow_capacity) * (self.hvac.fan_speed / 100.0)
        
        return effective_capacity

    def calculate_heat_gain(self, at_temp=None) -> float:
        """Calculate total heat gain in Watts."""
        temp = at_temp if at_temp is not None else self.room.current_temp
        
        # Calculate surface area
        surface_area = 2 * (
            self.room.length * self.room.breadth
            + self.room.length * self.room.height
            + self.room.breadth * self.room.height
        )
        
        # Get insulation factor based on level
        insulation_factors = {
            "low": 0.8,
            "medium": 0.5,
            "high": 0.3
        }
        if isinstance(self.room.wall_insulation, str):
            insulation_factor = insulation_factors.get(self.room.wall_insulation.lower(), 0.5)
        else:
            insulation_factor = 0.5  # Default insulation factor if invalid
        
        # Calculate environmental heat gain
        temperature_difference = self.room.external_temp - temp
        environmental_gain = surface_area * insulation_factor * temperature_difference
        
        # Add heat from people (approximately 100W per person)
        people_heat = self.room.num_people * 100
        
        # Add external heat gain (from sunlight, appliances, etc)
        total_heat_gain = environmental_gain + people_heat + self.room.heat_gain_external
        
        return total_heat_gain

    def calculate_energy_consumption(self, cooling_capacity: float) -> float:
        """Calculate energy consumption in Watts."""
        return cooling_capacity / self.hvac.cop

    def calculate_refrigerant_flow(self, cooling_capacity: float) -> float:
        """Calculate refrigerant flow rate in g/s."""
        # Using typical enthalpy values for R-410A
        enthalpy_difference = 20  # kJ/kg (typical value)
        return (cooling_capacity / 1000) / enthalpy_difference * 1000  # Convert to g/s

    def calculate_net_heat_at_temp(self, temp: float) -> float:
        """Calculate net heat transfer (in Watts) at a specific temperature."""
        # Calculate components at specified temperature
        cooling_capacity = self.calculate_cooling_capacity(at_temp=temp)
        heat_gain = self.calculate_heat_gain(at_temp=temp)
        
        # Calculate temperature difference from target
        temp_diff = abs(temp - self.room.target_temp)
        
        # Modified slowdown factor - less aggressive reduction
        slowdown_factor = 1.0  # Removed artificial slowdown
        
        # Calculate net heat based on mode
        if self.room.mode.lower() == "heating":
            # In heating mode, cooling_capacity is actually heating capacity
            net_heat = heat_gain + cooling_capacity
        else:  # cooling mode
            net_heat = heat_gain - cooling_capacity
        
        return net_heat
    
    def calculate_temp_change_rate(self, temp: float) -> float:
        """Calculate rate of temperature change (°C/s) at a specific temperature."""
        net_heat = self.calculate_net_heat_at_temp(temp)
        rate = net_heat / (self.room_air_mass * self.specific_heat_air * 1000)
        return rate

    def calculate_temperature_change(self) -> float:
        """Calculate new room temperature after one time step (in seconds)."""
        # If we're within a very small threshold of target, set to exact target
        if abs(self.room.current_temp - self.room.target_temp) < 0.1:
            return self.room.target_temp

        # Get temperature change rate
        rate = self.calculate_temp_change_rate(self.room.current_temp)
        
        # Apply time interval
        temp_change = rate * self.hvac.time_interval
        
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
        desired_rate_sign = -1 if self.room.mode.lower() == "cooling" else 1
            
        # Check if the system can move in the right direction at current temp
        current_rate = self.calculate_temp_change_rate(self.room.current_temp)
        
        # If rate sign matches desired direction, we're making progress
        # Also check if rate is significant enough (not practically zero)
        return (current_rate * desired_rate_sign) > 1e-6

    def calculate_time_to_target(self) -> float:
        """Calculate time to reach target temperature using an improved numerical solution."""
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
        
        # Use small, fixed time step for accurate integration
        time_step = 60.0  # 60 seconds per step
        max_time = 24 * 3600  # 24 hours max simulation
        max_steps = int(max_time / time_step)
        
        for step in range(max_steps):
            # Calculate rate at current temperature
            rate = self.calculate_temp_change_rate(current_temp)
            
            # Check if we can still make progress
            if (cooling_mode and rate >= 0) or (not cooling_mode and rate <= 0):
                return float('inf')  # Can't reach target
                
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
            if step % 10 == 0:
                self.debug_info.append({
                    "temp": current_temp, 
                    "rate": rate, 
                    "time_elapsed": total_time
                })
        
        # If we get here, we couldn't reach target within max simulation time
        return float('inf')

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and calculations."""
        cooling_capacity = self.calculate_cooling_capacity()
        energy_consumption = self.calculate_energy_consumption(cooling_capacity)
        refrigerant_flow = self.calculate_refrigerant_flow(cooling_capacity)
        time_to_target = self.calculate_time_to_target()
        
        return {
            "room_temperature": round(self.room.current_temp, 2),
            "target_temperature": self.room.target_temp,
            "cooling_capacity_kw": round(cooling_capacity / 1000, 2),
            "cooling_capacity_btu": round(cooling_capacity * 3.412, 2),
            "energy_consumption_w": round(energy_consumption, 2),
            "refrigerant_flow_gs": round(refrigerant_flow, 2),
            "heat_gain_w": round(self.calculate_heat_gain(), 2),
            "cop": self.hvac.cop,
            "mode": self.room.mode,
            "fan_speed": self.hvac.fan_speed,
            "humidity": self.room.humidity,
            "num_people": self.room.num_people,
            "external_heat_gain": self.room.heat_gain_external,
            "insulation_level": self.room.wall_insulation,
            "time_interval": self.hvac.time_interval,
            "room_size": round(self.room.length * self.room.breadth, 2),
            "external_temperature": self.room.external_temp,
            "time_to_target": time_to_target if time_to_target != float('inf') else "Cannot reach target",
            "can_reach_target": self.can_reach_target(),
            "temp_change_rate": round(self.calculate_temp_change_rate(self.room.current_temp) * 3600, 4),  # °C/hour
            "rated_power_kw": self.hvac.power
        }
