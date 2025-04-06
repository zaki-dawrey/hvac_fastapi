import math
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import time
import json


@dataclass
class VRFRoomParameters:
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


@dataclass
class ZoneParameters:
    name: str  # Zone name
    demand_kw: float  # Zone demand in kW
    current_temp: float  # Current temperature in °C
    target_temp: float  # Target temperature in °C
    area_percentage: float  # Percentage of total room area
    humidity: float = 50.0  # Current humidity (%)
    num_people: int = 0  # Number of people in zone
    heat_gain_external: float = 0.0  # External heat gain specific to this zone


@dataclass
class VRFHVACParameters:
    power: float = 3.5  # Rated power consumption in kW
    max_capacity_kw: float = 14.0  # Maximum system capacity in kW
    min_capacity_kw: float = 3.0  # Minimum system capacity in kW
    cop: float = 3.0  # Coefficient of Performance
    # Dictionary of zone names and their loads in kW
    zones: Dict[str, float] = None
    heat_recovery: bool = False  # Whether system has heat recovery capability
    air_flow_rate: float = 0.5  # m³/s
    supply_temp: float = 12.0  # °C
    fan_speed: float = 100.0  # Fan speed percentage
    time_interval: float = 1.0  # Simulation update interval in seconds


class VRFSystemSimulator:
    def __init__(self, room: VRFRoomParameters, hvac: VRFHVACParameters):
        self.room = room
        self.hvac = hvac
        self.specific_heat_air = 1.005  # kJ/kg·K
        self.air_density = 1.225  # kg/m³
        self.debug_info = []  # To store debug information during calculations
        self.hvac.zones = self.hvac.zones or {"main": 5.0}
        self.total_demand = sum(hvac.zones.values()) if hvac.zones else 0
        self.simulation_history = []  # Store history of simulation data
        self.simulation_time = 0.0  # Track total simulation time

        # Initialize zone parameters
        self.zone_parameters = self._initialize_zone_parameters()

        self.validate_inputs()

    def _initialize_zone_parameters(self) -> Dict[str, ZoneParameters]:
        """Initialize zone parameters based on hvac.zones."""
        zone_params = {}
        total_demand = sum(self.hvac.zones.values())

        for zone_name, demand in self.hvac.zones.items():
            # Calculate area percentage based on demand proportion
            area_percentage = (demand / total_demand) * \
                100 if total_demand > 0 else 0

            # Create ZoneParameters object for each zone
            zone_params[zone_name] = ZoneParameters(
                name=zone_name,
                demand_kw=demand,
                current_temp=self.room.current_temp,  # Initialize with room temp
                target_temp=self.room.target_temp,    # Initialize with room target
                area_percentage=area_percentage,
                humidity=self.room.humidity,          # Initialize with room humidity
                num_people=int(self.room.num_people * area_percentage /
                               100) if self.room.num_people > 0 else 0,
                heat_gain_external=self.room.heat_gain_external * area_percentage / 100
            )

        return zone_params

    def validate_inputs(self):
        """Validate input parameters and set defaults if necessary."""
        # Validate room dimensions
        if any(dim <= 0 for dim in [self.room.length, self.room.breadth, self.room.height]):
            raise ValueError("Room dimensions must be positive values")

        # Validate temperature settings
        if not -50 <= self.room.current_temp <= 50 or not -50 <= self.room.target_temp <= 50:
            raise ValueError(
                "Temperature values should be between -50°C and 50°C")

        # Validate mode
        if self.room.mode.lower() not in ["cooling", "heating"]:
            raise ValueError("Mode must be either 'cooling' or 'heating'")

        # Validate insulation
        if isinstance(self.room.wall_insulation, str) and self.room.wall_insulation.lower() not in ["low", "medium", "high"]:
            self.room.wall_insulation = "medium"
            print("Warning: Invalid insulation value. Set to 'medium'")

        # Validate fan speed
        if not 0 <= self.hvac.fan_speed <= 100:
            self.hvac.fan_speed = max(0, min(100, self.hvac.fan_speed))
            print(f"Warning: Fan speed adjusted to {self.hvac.fan_speed}%")

        # Calculate power if not provided
        if self.hvac.power is None:
            self.hvac.power = self.hvac.max_capacity_kw / self.hvac.cop

    @property
    def room_volume(self) -> float:
        """Calculate room volume in cubic meters."""
        return self.room.length * self.room.breadth * self.room.height

    @property
    def room_air_mass(self) -> float:
        """Calculate mass of air in the room."""
        return self.room_volume * self.air_density

    def calculate_cooling_capacity(self, at_temp=None) -> float:
        """Calculate cooling capacity in Watts with VRF system logic."""
        temp = at_temp if at_temp is not None else self.room.current_temp

        # Calculate maximum available capacity based on temperature
        max_capacity = self.hvac.max_capacity_kw * 1000  # Convert to Watts
        min_capacity = self.hvac.min_capacity_kw * 1000  # Convert to Watts

        # Calculate required capacity based on load
        required_capacity = self.total_demand * 1000  # Convert kW to Watts

        # Consider heat recovery if enabled
        if self.hvac.heat_recovery and self.room.mode.lower() == "cooling":
            recovered_heat = self.total_demand * 0.3 * 1000  # Assume 30% heat recovery
            required_capacity -= recovered_heat

        # Ensure capacity is within system limits
        effective_capacity = max(min_capacity, min(
            required_capacity, max_capacity))

        # Adjust for fan speed
        return effective_capacity * (self.hvac.fan_speed / 100.0)

    def calculate_zone_cooling_capacity(self, zone_name: str, at_temp=None) -> float:
        """Calculate cooling capacity allocated to a specific zone in Watts."""
        total_capacity = self.calculate_cooling_capacity(at_temp)

        # Get zone demand
        zone_demand = self.hvac.zones.get(zone_name, 0)

        # Calculate proportion of capacity allocated to this zone
        if self.total_demand > 0:
            zone_proportion = zone_demand / self.total_demand
        else:
            zone_proportion = 0

        return total_capacity * zone_proportion

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
            insulation_factor = insulation_factors.get(
                self.room.wall_insulation.lower(), 0.5)
        else:
            insulation_factor = 0.5  # Default insulation factor if invalid

        # Calculate environmental heat gain
        temperature_difference = self.room.external_temp - temp
        environmental_gain = surface_area * insulation_factor * temperature_difference

        # Add heat from people (approximately 100W per person)
        people_heat = self.room.num_people * 100

        # Add external heat gain (from sunlight, appliances, etc)
        total_heat_gain = environmental_gain + \
            people_heat + self.room.heat_gain_external

        return total_heat_gain

    def calculate_zone_heat_gain(self, zone_name: str, at_temp=None) -> float:
        """Calculate heat gain for a specific zone in Watts."""
        if zone_name not in self.zone_parameters:
            return 0.0

        zone = self.zone_parameters[zone_name]
        temp = at_temp if at_temp is not None else zone.current_temp

        # Calculate surface area proportion for this zone
        zone_area_proportion = zone.area_percentage / 100
        surface_area = 2 * (
            self.room.length * self.room.breadth
            + self.room.length * self.room.height
            + self.room.breadth * self.room.height
        ) * zone_area_proportion

        # Get insulation factor based on level
        insulation_factors = {
            "low": 0.8,
            "medium": 0.5,
            "high": 0.3
        }
        insulation_factor = insulation_factors.get(
            self.room.wall_insulation.lower(), 0.5)

        # Calculate environmental heat gain for zone
        temperature_difference = self.room.external_temp - temp
        environmental_gain = surface_area * insulation_factor * temperature_difference

        # Add heat from people in this zone
        people_heat = zone.num_people * 100

        # Add external heat gain specific to this zone
        total_heat_gain = environmental_gain + people_heat + zone.heat_gain_external

        return total_heat_gain

    def calculate_energy_consumption(self, cooling_capacity: float) -> float:
        """Calculate energy consumption in Watts."""
        return cooling_capacity / self.hvac.cop

    def calculate_refrigerant_flow(self, cooling_capacity: float) -> float:
        """Calculate refrigerant flow rate in g/s."""
        # Using typical enthalpy values for R-410A
        enthalpy_difference = 20  # kJ/kg (typical value)
        # Convert to g/s
        return (cooling_capacity / 1000) / enthalpy_difference * 1000

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

    def calculate_zone_net_heat_at_temp(self, zone_name: str, temp: float) -> float:
        """Calculate net heat transfer for a specific zone (in Watts) at a specific temperature."""
        # Calculate components at specified temperature for this zone
        cooling_capacity = self.calculate_zone_cooling_capacity(
            zone_name, at_temp=temp)
        heat_gain = self.calculate_zone_heat_gain(zone_name, at_temp=temp)

        zone = self.zone_parameters.get(zone_name)
        if not zone:
            return 0.0

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

    def calculate_zone_temp_change_rate(self, zone_name: str, temp: float) -> float:
        """Calculate rate of temperature change for a specific zone (°C/s)."""
        if zone_name not in self.zone_parameters:
            return 0.0

        zone = self.zone_parameters[zone_name]

        # Calculate zone's air mass based on its proportion of the room
        zone_air_mass = self.room_air_mass * (zone.area_percentage / 100)

        # Calculate net heat for this zone
        net_heat = self.calculate_zone_net_heat_at_temp(zone_name, temp)

        # Calculate rate of temperature change
        if zone_air_mass > 0:
            rate = net_heat / (zone_air_mass * self.specific_heat_air * 1000)
        else:
            rate = 0

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

    def calculate_zone_temperature_change(self, zone_name: str) -> float:
        """Calculate new zone temperature after one time step."""
        if zone_name not in self.zone_parameters:
            return self.room.current_temp

        zone = self.zone_parameters[zone_name]

        # If we're within a very small threshold of target, set to exact target
        if abs(zone.current_temp - zone.target_temp) < 0.1:
            return zone.target_temp

        # Get temperature change rate for this zone
        rate = self.calculate_zone_temp_change_rate(
            zone_name, zone.current_temp)

        # Apply time interval
        temp_change = rate * self.hvac.time_interval

        new_temp = zone.current_temp + temp_change

        # Check if we're approaching or moving away from target
        approaching_target = (
            (self.room.mode.lower() == "cooling" and new_temp < zone.current_temp) or
            (self.room.mode.lower() == "heating" and new_temp > zone.current_temp)
        )

        # Handle approaching target to prevent oscillation
        if approaching_target:
            # Prevent overshooting target temperature
            if self.room.mode.lower() == "cooling":
                return max(zone.target_temp, new_temp)
            else:  # heating mode
                return min(zone.target_temp, new_temp)
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

    def can_zone_reach_target(self, zone_name: str) -> bool:
        """Determine if a specific zone can reach its target temperature."""
        if zone_name not in self.zone_parameters:
            return False

        zone = self.zone_parameters[zone_name]

        # Check if we're already at target
        if abs(zone.current_temp - zone.target_temp) < 0.1:
            return True

        # Direction we need to move (cooling = negative rate, heating = positive rate)
        desired_rate_sign = -1 if self.room.mode.lower() == "cooling" else 1

        # Check if the system can move in the right direction at current temp
        current_rate = self.calculate_zone_temp_change_rate(
            zone_name, zone.current_temp)

        # If rate sign matches desired direction, we're making progress
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

    def calculate_zone_time_to_target(self, zone_name: str) -> float:
        """Calculate time for a specific zone to reach its target temperature."""
        if zone_name not in self.zone_parameters:
            return float('inf')

        zone = self.zone_parameters[zone_name]

        # If already at target, return 0
        if abs(zone.current_temp - zone.target_temp) < 0.1:
            return 0.0

        # Check if target can be reached
        if not self.can_zone_reach_target(zone_name):
            return float('inf')

        # Setup for numerical integration
        start_temp = zone.current_temp
        target_temp = zone.target_temp
        cooling_mode = self.room.mode.lower() == "cooling"

        # Determine direction of approach
        proper_direction = (cooling_mode and start_temp > target_temp) or (
            not cooling_mode and start_temp < target_temp)
        if not proper_direction:
            # Can't reach target if going in wrong direction
            return float('inf')

        # Track current progress
        current_temp = start_temp
        total_time = 0.0

        # Use small, fixed time step for accurate integration
        time_step = 60.0  # 60 seconds per step
        max_time = 24 * 3600  # 24 hours max simulation
        max_steps = int(max_time / time_step)

        for step in range(max_steps):
            # Calculate rate at current temperature
            rate = self.calculate_zone_temp_change_rate(
                zone_name, current_temp)

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

        # If we get here, we couldn't reach target within max simulation time
        return float('inf')

    def check_for_failures(self) -> dict:
        """Check for potential VRF system failures based on current conditions."""
        failures = {}

        # Environmental/capacity issues
        room_volume = self.room_volume
        required_capacity = room_volume * 0.1  # rough estimate: 100W per cubic meter
        actual_capacity = self.hvac.max_capacity_kw  # in kW

        # Undersized system - compare to total demand
        if actual_capacity < self.total_demand * 0.9:
            failures['undersized'] = {
                'probability': min(1.0, (self.total_demand - actual_capacity) / (actual_capacity * 0.2 + 0.1)),
                'message': f'System capacity ({actual_capacity:.1f}kW) insufficient for total zone demand ({self.total_demand:.1f}kW).',
                'severity': 'high',
                'solution': f'Increase system capacity from {actual_capacity}kW to at least {self.total_demand * 1.1:.1f}kW to handle all zones.'
            }

        # Extreme temperature operation
        if self.room.external_temp > 43:
            failures['extreme_temp'] = {
                # scales from 0 to 1 as temp goes from 43 to 50
                'probability': (self.room.external_temp - 43) / 7,
                'message': f'System operating above rated conditions ({self.room.external_temp}°C).',
                'severity': 'medium',
                'solution': 'Lower external temperature setting or increase system capacity to compensate for extreme conditions.'
            }

        # VRF-specific: Too many zones for system capacity
        if len(self.zone_parameters) > 0:
            avg_zone_capacity = actual_capacity / len(self.zone_parameters)
            if avg_zone_capacity < 1.0 and len(self.zone_parameters) > 4:
                failures['too_many_zones'] = {
                    'probability': min(1.0, 1.0 - avg_zone_capacity/1.5),
                    'message': f'Too many zones ({len(self.zone_parameters)}) for system capacity ({actual_capacity:.1f}kW).',
                    'severity': 'medium',
                    'solution': f'Reduce number of zones or increase system capacity to at least {len(self.zone_parameters) * 1.5:.1f}kW.'
                }

        # VRF-specific: Zone balancing issues
        zone_demands = [z.demand_kw for z in self.zone_parameters.values()]
        if zone_demands:
            max_demand = max(zone_demands)
            min_demand = min(zone_demands)
            if max_demand > min_demand * 5 and len(zone_demands) > 1:
                failures['zone_imbalance'] = {
                    'probability': min(0.9, max_demand / (min_demand * 5)),
                    'message': f'Significant zone demand imbalance (max: {max_demand:.1f}kW, min: {min_demand:.1f}kW).',
                    'severity': 'medium',
                    'solution': 'Redistribute zone demands more evenly or consider a system with better modulation capability.'
                }

        # Fan speed too low
        if self.hvac.fan_speed < 30 and self.hvac.max_capacity_kw > 5:
            failures['fan_speed_low'] = {
                'probability': 0.9,
                'message': 'Fan speed too low for selected capacity rating, reducing efficiency.',
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

        # System oversized for total demand
        if actual_capacity > self.total_demand * 2.5:
            failures['oversized'] = {
                'probability': 0.7,
                'message': f'System capacity ({actual_capacity:.1f}kW) significantly exceeds total demand ({self.total_demand:.1f}kW).',
                'severity': 'low',
                'solution': 'Consider reducing system capacity or adding more zones for more efficient operation.'
            }

        # VRF-specific: Heat recovery could be beneficial but isn't enabled
        if not self.hvac.heat_recovery:
            # Check if we have mixed mode operation potential
            has_cooling_zones = False
            has_heating_zones = False

            for zone in self.zone_parameters.values():
                if zone.current_temp > zone.target_temp:
                    has_cooling_zones = True
                if zone.current_temp < zone.target_temp:
                    has_heating_zones = True

            if has_cooling_zones and has_heating_zones:
                failures['missing_heat_recovery'] = {
                    'probability': 0.95,
                    'message': 'System has mixed heating/cooling demands but heat recovery is not enabled.',
                    'severity': 'medium',
                    'solution': 'Enable heat recovery to improve efficiency with mixed mode operation.'
                }

        # Too many people for room size
        people_density = self.room.num_people / \
            (self.room.length * self.room.breadth)
        if people_density > 0.5:  # More than 1 person per 2 square meters
            failures['overcrowding'] = {
                'probability': min(0.9, people_density - 0.3),
                'message': f'High occupant density ({self.room.num_people} people) for room size.',
                'severity': 'medium',
                'solution': f'Reduce number of people or increase system capacity to handle additional heat load.'
            }

        # Inefficient mode selection
        if (self.room.mode.lower() == "cooling" and self.room.current_temp < self.room.external_temp - 5) or \
           (self.room.mode.lower() == "heating" and self.room.current_temp > self.room.external_temp + 5):
            failures['inefficient_mode'] = {
                'probability': 0.6,
                'message': f'Current mode ({self.room.mode}) may be inefficient given temperature conditions.',
                'severity': 'low',
                'solution': f'Consider changing mode or adjusting target temperature for more efficient operation.'
            }

        # Unrealistic target temperatures in zones
        extreme_targets = []
        for zone_name, zone in self.zone_parameters.items():
            if (self.room.mode.lower() == "cooling" and zone.target_temp < 18) or \
               (self.room.mode.lower() == "heating" and zone.target_temp > 30):
                extreme_targets.append(f"{zone_name} ({zone.target_temp}°C)")

        if extreme_targets:
            failures['unrealistic_zone_targets'] = {
                'probability': 0.75,
                'message': f'Extreme target temperatures in zones: {", ".join(extreme_targets)}.',
                'severity': 'medium',
                'solution': 'Set more moderate target temperatures (20-26°C) for optimal efficiency.'
            }

        # Airflow rate mismatch with system capacity
        expected_airflow = self.hvac.max_capacity_kw * \
            0.18  # rough estimate: 0.18 m³/s per kW for VRF
        if self.hvac.air_flow_rate < expected_airflow * 0.6:
            failures['low_airflow'] = {
                'probability': 0.8,
                'message': 'Airflow rate too low for system capacity rating, reducing efficiency.',
                'severity': 'medium',
                'solution': f'Increase airflow rate to at least {expected_airflow:.2f} m³/s to match system capacity.'
            }

        # VRF-specific: Min capacity too high for low load conditions
        min_zone_demand = min(
            z.demand_kw for z in self.zone_parameters.values()) if self.zone_parameters else 0
        if min_zone_demand < self.hvac.min_capacity_kw * 0.5 and min_zone_demand > 0:
            failures['high_min_capacity'] = {
                'probability': min(0.9, 1.0 - (min_zone_demand / (self.hvac.min_capacity_kw * 0.5))),
                'message': f'Minimum capacity ({self.hvac.min_capacity_kw:.1f}kW) too high for smallest zone ({min_zone_demand:.1f}kW).',
                'severity': 'medium',
                'solution': 'Lower system minimum capacity or combine smaller zones for better modulation.'
            }

        # VRF-specific: Supply temperature too extreme
        if self.room.mode.lower() == "cooling" and self.hvac.supply_temp < 8:
            failures['supply_temp_too_low'] = {
                'probability': min(0.85, (8 - self.hvac.supply_temp) / 4),
                'message': f'Supply temperature ({self.hvac.supply_temp}°C) is too low for efficient operation.',
                'severity': 'medium',
                'solution': 'Increase supply temperature to 10-12°C for better efficiency and comfort.'
            }
        elif self.room.mode.lower() == "heating" and self.hvac.supply_temp > 45:
            failures['supply_temp_too_high'] = {
                'probability': min(0.85, (self.hvac.supply_temp - 45) / 10),
                'message': f'Supply temperature ({self.hvac.supply_temp}°C) is too high for efficient operation.',
                'severity': 'medium',
                'solution': 'Decrease supply temperature to 35-40°C for better efficiency and comfort.'
            }

        # Check for zones that cannot reach target
        unreachable_zones = []
        for zone_name, zone in self.zone_parameters.items():
            if not self.can_zone_reach_target(zone_name):
                unreachable_zones.append(zone_name)

        if unreachable_zones:
            failures['unreachable_zones'] = {
                'probability': 0.95,
                'message': f'The following zones cannot reach target temperature: {", ".join(unreachable_zones)}.',
                'severity': 'high',
                'solution': 'Adjust zone demand allocation, increase system capacity, or set more realistic target temperatures.'
            }

        return failures

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and calculations."""
        cooling_capacity = self.calculate_cooling_capacity()
        energy_consumption = self.calculate_energy_consumption(
            cooling_capacity)
        refrigerant_flow = self.calculate_refrigerant_flow(cooling_capacity)
        time_to_target = self.calculate_time_to_target()
        failures = self.check_for_failures()
        active_failures = {k: v for k,
                           v in failures.items() if v['probability'] > 0.5}

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
            # °C/hour
            "temp_change_rate": round(self.calculate_temp_change_rate(self.room.current_temp) * 3600, 4),
            # Changed from power to max_capacity_kw
            "rated_power_kw": self.hvac.max_capacity_kw,
            "zones": self.hvac.zones,  # Add zones to response
            "heat_recovery": self.hvac.heat_recovery,  # Add heat recovery status
            # Add total demand
            "total_zone_demand_kw": round(self.total_demand, 2),
            "supply_temp": self.hvac.supply_temp,  # Add supply temperature
            # Add current simulation time
            "simulation_time": round(self.simulation_time, 2),
            # Add zone-specific data
            "zone_data": self.get_all_zone_status(),
            "failures": failures,
            "active_failures": active_failures,
            "has_critical_failure": any(f['severity'] == 'high' and f['probability'] > 0.7 for f in failures.values()),
            "warnings": [f['message'] for f in failures.values() if 0.3 < f['probability'] <= 0.7],
            "critical_alerts": [f['message'] for f in failures.values() if f['probability'] > 0.7]

        }

    def get_zone_status(self, zone_name: str) -> Dict[str, Any]:
        """Get status for a specific zone."""
        if zone_name not in self.zone_parameters:
            return {}

        zone = self.zone_parameters[zone_name]

        # Calculate zone-specific metrics
        cooling_capacity = self.calculate_zone_cooling_capacity(zone_name)
        energy_consumption = self.calculate_energy_consumption(
            cooling_capacity)
        heat_gain = self.calculate_zone_heat_gain(zone_name)
        time_to_target = self.calculate_zone_time_to_target(zone_name)

        return {
            "name": zone_name,
            "current_temperature": round(zone.current_temp, 2),
            "target_temperature": zone.target_temp,
            "humidity": zone.humidity,
            "cooling_capacity_kw": round(cooling_capacity / 1000, 2),
            "energy_consumption_w": round(energy_consumption, 2),
            "heat_gain_w": round(heat_gain, 2),
            "demand_kw": zone.demand_kw,
            "area_percentage": zone.area_percentage,
            "num_people": zone.num_people,
            "external_heat_gain": zone.heat_gain_external,
            "time_to_target": time_to_target if time_to_target != float('inf') else "Cannot reach target",
            "can_reach_target": self.can_zone_reach_target(zone_name),
            # °C/hour
            "temp_change_rate": round(self.calculate_zone_temp_change_rate(zone_name, zone.current_temp) * 3600, 4)
        }

    def get_all_zone_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all zones."""
        zone_statuses = {}
        for zone_name in self.zone_parameters:
            zone_statuses[zone_name] = self.get_zone_status(zone_name)
        return zone_statuses

    def calculate_humidity_change(self) -> float:
        """Calculate changes in humidity based on cooling/heating operation."""
        # This is a simplified model for humidity changes
        if self.room.mode.lower() == "cooling":
            # Cooling typically reduces humidity
            humidity_change = -0.1 * \
                abs(self.room.current_temp - self.hvac.supply_temp) / 10
        else:
            # Heating typically reduces humidity slightly
            humidity_change = -0.05 * \
                abs(self.room.current_temp - self.hvac.supply_temp) / 10

        # Apply changes based on current humidity
        # Don't let humidity go below 20% or above 90%
        new_humidity = max(20, min(90, self.room.humidity + humidity_change))
        return new_humidity

    def calculate_zone_humidity_change(self, zone_name: str) -> float:
        """Calculate humidity change for a specific zone."""
        if zone_name not in self.zone_parameters:
            return 50.0  # Default humidity

        zone = self.zone_parameters[zone_name]

        # This is a simplified model for humidity changes
        if self.room.mode.lower() == "cooling":
            # Cooling typically reduces humidity
            humidity_change = -0.1 * \
                abs(zone.current_temp - self.hvac.supply_temp) / 10
        else:
            # Heating typically reduces humidity slightly
            humidity_change = -0.05 * \
                abs(zone.current_temp - self.hvac.supply_temp) / 10

        # Apply changes based on current humidity
        # Don't let humidity go below 20% or above 90%
        new_humidity = max(20, min(90, zone.humidity + humidity_change))
        return new_humidity

    def update_room_state(self):
        """Update room temperature and humidity based on current calculations."""
        # Update room temperature
        self.room.current_temp = self.calculate_temperature_change()

        # Update room humidity
        self.room.humidity = self.calculate_humidity_change()

        # Update all zone states
        for zone_name in self.zone_parameters:
            zone = self.zone_parameters[zone_name]
            zone.current_temp = self.calculate_zone_temperature_change(
                zone_name)
            zone.humidity = self.calculate_zone_humidity_change(zone_name)

        # Update simulation time
        self.simulation_time += self.hvac.time_interval

    def run_simulation_step(self) -> Dict[str, Any]:
        """Run a single step of the simulation and return current status."""
        self.update_room_state()
        status = self.get_system_status()

        # Store status in history
        self.simulation_history.append(status)

        return status

    def run_simulation(self, duration: float) -> List[Dict[str, Any]]:
        """Run simulation for specified duration (in seconds) and return history."""
        # Calculate number of steps
        steps = math.ceil(duration / self.hvac.time_interval)

        # Clear previous history
        self.simulation_history = []

        # Run simulation steps
        for _ in range(steps):
            self.run_simulation_step()

        return self.simulation_history

    def run_to_target(self, max_duration: float = 3600) -> Dict[str, Any]:
        """
        Run simulation until target temperature is reached or max_duration is exceeded.
        Returns final system status.
        """
        # Calculate number of steps
        steps = math.ceil(max_duration / self.hvac.time_interval)

        # Clear previous history
        self.simulation_history = []

        # Tolerance for considering target reached
        temp_tolerance = 0.2

        for _ in range(steps):
            status = self.run_simulation_step()

            # Check if we've reached target temperature (including zones)
            main_target_reached = abs(
                self.room.current_temp - self.room.target_temp) <= temp_tolerance

            # Check if all zones have reached their targets
            all_zones_reached = True
            for zone_name, zone in self.zone_parameters.items():
                if abs(zone.current_temp - zone.target_temp) > temp_tolerance:
                    all_zones_reached = False
                    break

            # If both main room and all zones have reached targets, we're done
            if main_target_reached and all_zones_reached:
                break

            # Also break if we can't make progress
            if not self.can_reach_target():
                break

        return self.get_system_status()

    def get_simulation_history(self) -> List[Dict[str, Any]]:
        """Get the full simulation history."""
        return self.simulation_history

    def calculate_zone_distribution(self) -> Dict[str, Dict[str, float]]:
        """Calculate how resources are distributed among zones."""
        zone_distribution = {}
        total_demand = sum(self.hvac.zones.values())

        for zone_name, demand in self.hvac.zones.items():
            if total_demand > 0:
                proportion = demand / total_demand
            else:
                proportion = 0

            # Calculate allocated capacity
            total_capacity = self.calculate_cooling_capacity()
            allocated_capacity = total_capacity * proportion

            zone_distribution[zone_name] = {
                "demand_kw": demand,
                "proportion": proportion,
                "allocated_capacity_w": allocated_capacity,
                "energy_consumption_w": self.calculate_energy_consumption(allocated_capacity)
            }

        return zone_distribution

    def adjust_zone_demand(self, zone_name: str, new_demand: float) -> bool:
        """
        Adjust the demand for a specific zone.
        Returns True if successful, False if zone doesn't exist.
        """
        if zone_name not in self.hvac.zones:
            return False

        # Update zone demand
        self.hvac.zones[zone_name] = new_demand

        # Recalculate total demand
        self.total_demand = sum(self.hvac.zones.values())

        # Update zone parameters
        self._initialize_zone_parameters()

        return True

    def adjust_zone_target_temp(self, zone_name: str, target_temp: float) -> bool:
        """
        Adjust the target temperature for a specific zone.
        Returns True if successful, False if zone doesn't exist.
        """
        if zone_name not in self.zone_parameters:
            return False

        # Update zone target temperature
        self.zone_parameters[zone_name].target_temp = target_temp

        return True

    def add_zone(self, zone_name: str, demand_kw: float, target_temp: float = None) -> bool:
        """
        Add a new zone to the system.
        Returns True if successful, False if zone already exists.
        """
        if zone_name in self.hvac.zones:
            return False

        # Set default target temperature if not provided
        if target_temp is None:
            target_temp = self.room.target_temp

        # Add zone to HVAC zones
        self.hvac.zones[zone_name] = demand_kw

        # Update total demand
        self.total_demand = sum(self.hvac.zones.values())

        # Reinitialize zone parameters to include the new zone
        self.zone_parameters = self._initialize_zone_parameters()

        # Update specific target temperature for this zone
        self.zone_parameters[zone_name].target_temp = target_temp

        return True

    def remove_zone(self, zone_name: str) -> bool:
        """
        Remove a zone from the system.
        Returns True if successful, False if zone doesn't exist.
        """
        if zone_name not in self.hvac.zones:
            return False

        # Remove zone from HVAC zones
        del self.hvac.zones[zone_name]

        # Remove zone from zone parameters
        if zone_name in self.zone_parameters:
            del self.zone_parameters[zone_name]

        # Update total demand
        self.total_demand = sum(self.hvac.zones.values())

        # Reinitialize zone parameters with updated proportions
        self.zone_parameters = self._initialize_zone_parameters()

        return True

    def adjust_system_parameters(self, **kwargs):
        """Update system parameters with provided keyword arguments."""
        # Update room parameters
        for key, value in kwargs.items():
            if hasattr(self.room, key):
                setattr(self.room, key, value)
            elif hasattr(self.hvac, key):
                setattr(self.hvac, key, value)

        # Recalculate dependent parameters
        self.total_demand = sum(self.hvac.zones.values()
                                ) if self.hvac.zones else 0

        # Validate inputs after changes
        self.validate_inputs()

    def save_simulation_to_json(self, filepath: str):
        """Save simulation history to a JSON file."""
        data = {
            "room_parameters": self.room.__dict__,
            "hvac_parameters": {k: v for k, v in self.hvac.__dict__.items() if k != 'zones'},
            "zones": self.hvac.zones,
            "simulation_history": self.simulation_history,
            "simulation_time": self.simulation_time
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str):
        """Load simulation from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Create room parameters
        room = VRFRoomParameters(
            **{k: v for k, v in data["room_parameters"].items()})

        # Create HVAC parameters
        hvac_params = {k: v for k, v in data["hvac_parameters"].items()}
        hvac_params["zones"] = data["zones"]
        hvac = VRFHVACParameters(**hvac_params)

        # Create simulator
        simulator = cls(room, hvac)

        # Load simulation history if available
        if "simulation_history" in data:
            simulator.simulation_history = data["simulation_history"]

        # Load simulation time if available
        if "simulation_time" in data:
            simulator.simulation_time = data["simulation_time"]

        return simulator

    def generate_efficiency_report(self) -> Dict[str, Any]:
        """Generate an efficiency report for the current system configuration."""
        # Calculate base metrics
        cooling_capacity = self.calculate_cooling_capacity()
        energy_consumption = self.calculate_energy_consumption(
            cooling_capacity)
        heat_gain = self.calculate_heat_gain()

        # Calculate efficiency metrics
        energy_efficiency_ratio = (
            cooling_capacity * 3.412) / energy_consumption if energy_consumption > 0 else 0
        watts_per_square_meter = energy_consumption / \
            (self.room.length * self.room.breadth)

        # Zone-specific efficiency
        zone_efficiency = {}
        for zone_name in self.zone_parameters:
            zone = self.zone_parameters[zone_name]
            zone_capacity = self.calculate_zone_cooling_capacity(zone_name)
            zone_energy = self.calculate_energy_consumption(zone_capacity)
            zone_heat_gain = self.calculate_zone_heat_gain(zone_name)

            # Calculate zone efficiency metrics
            zone_eer = (zone_capacity * 3.412) / \
                zone_energy if zone_energy > 0 else 0
            zone_watts_per_sqm = zone_energy / \
                ((self.room.length * self.room.breadth)
                 * (zone.area_percentage / 100))

            zone_efficiency[zone_name] = {
                "cooling_capacity_btu": round(zone_capacity * 3.412, 2),
                "energy_consumption_w": round(zone_energy, 2),
                "heat_gain_w": round(zone_heat_gain, 2),
                "energy_efficiency_ratio": round(zone_eer, 2),
                "watts_per_square_meter": round(zone_watts_per_sqm, 2),
                "can_reach_target": self.can_zone_reach_target(zone_name),
                "time_to_target_s": self.calculate_zone_time_to_target(zone_name)
            }

        return {
            "system_cop": self.hvac.cop,
            "total_cooling_capacity_btu": round(cooling_capacity * 3.412, 2),
            "total_energy_consumption_w": round(energy_consumption, 2),
            "total_heat_gain_w": round(heat_gain, 2),
            "energy_efficiency_ratio": round(energy_efficiency_ratio, 2),
            "watts_per_square_meter": round(watts_per_square_meter, 2),
            "can_reach_target": self.can_reach_target(),
            "time_to_target_s": self.calculate_time_to_target(),
            "zone_efficiency": zone_efficiency,
            "recommended_improvements": self._generate_improvement_recommendations()
        }

    def _generate_improvement_recommendations(self) -> List[str]:
        """Generate recommendations for system improvements based on current state."""
        recommendations = []

        # Check if system is properly sized
        total_demand = self.total_demand * 1000  # Convert to W
        max_capacity = self.hvac.max_capacity_kw * 1000

        if total_demand > max_capacity * 0.9:
            recommendations.append(
                "System is undersized for the total demand. Consider upgrading to a higher capacity unit.")
        elif total_demand < max_capacity * 0.5:
            recommendations.append(
                "System may be oversized for the current demand, which can reduce efficiency. Consider a smaller unit or adding more zones.")

        # Check insulation level
        if self.room.wall_insulation.lower() == "low":
            recommendations.append(
                "Upgrading wall insulation would significantly reduce heat gain/loss and improve efficiency.")

        # Check for heat recovery
        if not self.hvac.heat_recovery and len(self.hvac.zones) > 1:
            recommendations.append(
                "Adding heat recovery capability would improve efficiency in a multi-zone setup.")

        # Check target temperature vs external temperature
        temp_diff = abs(self.room.target_temp - self.room.external_temp)
        if temp_diff > 15 and self.room.mode.lower() == "cooling":
            recommendations.append(
                "Large temperature difference between target and external temperature. Consider a more moderate target temperature to improve efficiency.")

        # Check zones that cannot reach target
        unreachable_zones = []
        for zone_name in self.zone_parameters:
            if not self.can_zone_reach_target(zone_name):
                unreachable_zones.append(zone_name)

        if unreachable_zones:
            zone_list = ", ".join(unreachable_zones)
            recommendations.append(
                f"The following zones cannot reach their target temperatures: {zone_list}. Consider adjusting zone demands or target temperatures.")

        return recommendations


# def example_usage():
#     """Example usage of the VRF simulator with zones."""
#     # Create room parameters
#     room = VRFRoomParameters(
#         length=10.0,
#         breadth=8.0,
#         height=3.0,
#         current_temp=30.0,
#         target_temp=24.0,
#         external_temp=35.0,
#         wall_insulation="medium",
#         humidity=60.0,
#         num_people=5,
#         heat_gain_external=500.0,
#         mode="cooling"
#     )

#     # Create HVAC parameters with zones
#     hvac = VRFHVACParameters(
#         max_capacity_kw=14.0,
#         min_capacity_kw=3.0,
#         cop=3.5,
#         zones={
#             "office": 5.0,
#             "meeting_room": 3.0,
#             "reception": 2.0
#         },
#         heat_recovery=True,
#         air_flow_rate=0.6,
#         supply_temp=12.0,
#         fan_speed=90.0,
#         time_interval=60.0  # 1 minute simulation steps
#     )

#     # Create simulator
#     simulator = VRFSystemSimulator(room, hvac)

#     # Set different target temperatures for each zone
#     simulator.adjust_zone_target_temp("office", 23.0)
#     simulator.adjust_zone_target_temp("meeting_room", 24.0)
#     simulator.adjust_zone_target_temp("reception", 25.0)

#     # Run simulation for 2 hours
#     print("Running simulation for 2 hours...")
#     simulation_results = simulator.run_simulation(7200)  # 2 hours in seconds

#     # Print final status
#     final_status = simulation_results[-1]
#     print(f"Final room temperature: {final_status['room_temperature']}°C")
#     print(f"Energy consumed: {final_status['energy_consumption_w']} W")

#     # Print zone temperatures
#     print("\nZone temperatures:")
#     for zone_name, zone_data in final_status['zone_data'].items():
#         print(f"  {zone_name}: {zone_data['current_temperature']}°C (target: {zone_data['target_temperature']}°C)")

#     # Generate efficiency report
#     print("\nGenerating efficiency report...")
#     efficiency_report = simulator.generate_efficiency_report()

#     # Print recommendations
#     print("\nRecommended improvements:")
#     for recommendation in efficiency_report['recommended_improvements']:3
#         print(f"- {recommendation}")

#     # Save simulation to file
#     simulator.save_simulation_to_json("vrf_simulation_results.json")
#     print("\nSimulation results saved to vrf_simulation_results.json")


# if __name__ == "__main__":
#     example_usage()
