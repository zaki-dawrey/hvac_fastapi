import math
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RoomParameters:
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
class HVACParameters:
    power: float  # kW
    cop: float = 3.0  # Coefficient of Performance
    air_flow_rate: float = 0.5  # m³/s
    supply_temp: float = 12.0  # °C
    fan_speed: float = 100.0  # Fan speed percentage
    time_interval: float = 1.0  # Simulation update interval in seconds

class HVACSimulator:
    def __init__(self, room: RoomParameters, hvac: HVACParameters):
        self.room = room
        self.hvac = hvac
        self.specific_heat_air = 1.005  # kJ/kg·K
        self.air_density = 1.225  # kg/m³

    @property
    def room_volume(self) -> float:
        """Calculate room volume in cubic meters."""
        return self.room.length * self.room.breadth * self.room.height

    @property
    def room_air_mass(self) -> float:
        """Calculate mass of air in the room."""
        return self.room_volume * self.air_density

    def calculate_cooling_capacity(self) -> float:
        """Calculate cooling capacity in Watts."""
        return (
            self.hvac.air_flow_rate
            * self.specific_heat_air
            * 1000  # Convert to Watts
            * (self.room.current_temp - self.hvac.supply_temp)
        )

    def calculate_heat_gain(self) -> float:
        """Calculate total heat gain in Watts."""
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
        insulation_factor = insulation_factors.get(self.room.wall_insulation.lower(), 0.5)
        
        # Calculate environmental heat gain
        temperature_difference = self.room.external_temp - self.room.current_temp
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

    def calculate_temperature_change(self) -> float:
        """Calculate new room temperature after one time step (in seconds)."""
        # If we're within 0.1°C of target, maintain current temperature
        if abs(self.room.current_temp - self.room.target_temp) < 0.1:
            return self.room.current_temp

        cooling_capacity = self.calculate_cooling_capacity()
        heat_gain = self.calculate_heat_gain()
        
        # Calculate temperature difference from target
        temp_diff = abs(self.room.current_temp - self.room.target_temp)
        
        # Apply gradual slowdown factor as we approach target temperature
        slowdown_factor = min(1.0, temp_diff / 2.0)  # Gradually reduce effect as we get closer
        
        # Adjust cooling/heating based on mode
        if self.room.mode.lower() == "heating":
            net_heat = heat_gain + (cooling_capacity * slowdown_factor)
        else:  # cooling mode
            net_heat = heat_gain - (cooling_capacity * slowdown_factor)
            
        # Adjust for fan speed
        net_heat *= (self.hvac.fan_speed / 100.0)
        
        # Temperature change using heat balance equation
        temp_change = (
            net_heat * self.hvac.time_interval / (self.room_air_mass * self.specific_heat_air * 1000)
        )
        
        new_temp = self.room.current_temp + temp_change
        
        # Prevent overshooting target temperature
        if self.room.current_temp > self.room.target_temp:
            return max(self.room.target_temp, new_temp)
        else:
            return min(self.room.target_temp, new_temp)

    def calculate_time_to_target(self) -> float:
        """Calculate estimated time to reach target temperature in seconds."""
        if abs(self.room.current_temp - self.room.target_temp) < 0.1:
            return 0

        cooling_capacity = self.calculate_cooling_capacity()
        heat_gain = self.calculate_heat_gain()
        
        # Calculate net heat transfer rate
        if self.room.mode.lower() == "heating":
            net_heat = heat_gain + (cooling_capacity * (self.hvac.fan_speed / 100.0))
        else:  # cooling mode
            net_heat = heat_gain - (cooling_capacity * (self.hvac.fan_speed / 100.0))
        
        # Calculate temperature change rate (°C/s)
        temp_change_rate = abs(
            net_heat / (self.room_air_mass * self.specific_heat_air * 1000)
        )
        
        # Calculate time needed
        temp_difference = abs(self.room.current_temp - self.room.target_temp)
        estimated_time = temp_difference / temp_change_rate if temp_change_rate > 0 else float('inf')
        
        return round(estimated_time, 2)

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and calculations."""
        cooling_capacity = self.calculate_cooling_capacity()
        energy_consumption = self.calculate_energy_consumption(cooling_capacity)
        refrigerant_flow = self.calculate_refrigerant_flow(cooling_capacity)
        
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
            "external_temperature": self.room.external_temp
        }