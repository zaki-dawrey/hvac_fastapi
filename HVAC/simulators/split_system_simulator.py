"""Simulator for a split HVAC system."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SplitRoomParameters:
    """Data class to hold room parameters for split system."""
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
class SplitHVACParameters:
    """Data class to hold HVAC parameters for split system."""
    power: float  # kW
    cop: float = 3.0  # Coefficient of Performance
    air_flow_rate: float = 0.5  # m³/s
    supply_temp: float = 12.0  # °C
    fan_speed: float = 100.0  # Fan speed percentage
    time_interval: float = 1.0  # Simulation update interval in seconds


class SplitSystemSimulator:
    """Simulator for a split HVAC system."""

    def __init__(self, room: SplitRoomParameters, hvac: SplitHVACParameters):
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
        effective_capacity = min(
            rated_capacity, airflow_capacity) * (self.hvac.fan_speed / 100.0)

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

    def check_for_failures(self) -> dict:
        """Check for potential HVAC system failures based on current conditions."""
        failures = {}

        # Environmental/capacity issues
        room_volume = self.room_volume
        required_capacity = room_volume * 0.1  # rough estimate: 100W per cubic meter
        actual_capacity = self.calculate_cooling_capacity() / 1000  # in kW

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
                'solution': 'Increase system power, reduce number of people in room, or improve wall insulation.'
            }

        # Fan speed too low
        if self.hvac.fan_speed < 30 and self.hvac.power > 2:
            failures['fan_speed_low'] = {
                'probability': 0.9,
                'message': 'Fan speed too low for selected power rating, reducing efficiency.',
                'severity': 'medium',
                'solution': 'Increase fan speed to at least 50% for optimal airflow.'
            }

        # Fan speed too high for small room
        if self.hvac.fan_speed > 90 and room_volume < 30:
            failures['fan_speed_high'] = {
                'probability': 0.8,
                'message': 'Fan speed too high for room size, causing energy waste.',
                'severity': 'low',
                'solution': 'Reduce fan speed to 60-70% for this room size to optimize efficiency.'
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

        # System oversized for room
        if actual_capacity > required_capacity * 2:
            failures['oversized'] = {
                'probability': 0.7,
                'message': f'System capacity ({actual_capacity:.1f}kW) significantly exceeds room requirements ({required_capacity:.1f}kW).',
                'severity': 'low',
                'solution': 'Consider reducing system power or decreasing airflow rate for more efficient operation.'
            }

        # Too many people for room size
        people_density = self.room.num_people / \
            (self.room.length * self.room.breadth)
        if people_density > 0.5:  # More than 1 person per 2 square meters
            failures['overcrowding'] = {
                'probability': min(0.9, people_density - 0.3),
                'message': f'High occupant density ({self.room.num_people} people) for room size.',
                'severity': 'medium',
                'solution': f'Reduce number of people or increase system power to handle additional heat load.'
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

        # Target temperature unrealistic
        if (self.room.mode.lower() == "cooling" and self.room.target_temp < 18) or \
           (self.room.mode.lower() == "heating" and self.room.target_temp > 30):
            failures['unrealistic_target'] = {
                'probability': 0.75,
                'message': f'Target temperature ({self.room.target_temp}°C) requires excessive energy consumption.',
                'severity': 'medium',
                'solution': 'Set a more moderate target temperature (20-26°C) for optimal efficiency.'
            }

        # Airflow rate mismatch with system power
        expected_airflow = self.hvac.power * 0.2  # rough estimate: 0.2 m³/s per kW
        if self.hvac.air_flow_rate < expected_airflow * 0.6:
            failures['low_airflow'] = {
                'probability': 0.8,
                'message': 'Airflow rate too low for system power rating, reducing efficiency.',
                'severity': 'medium',
                'solution': f'Increase airflow rate to at least {expected_airflow:.2f} m³/s to match system power.'
            }

        # Large temperature differential between current and target
        temp_diff_target = abs(self.room.current_temp - self.room.target_temp)
        if temp_diff_target > 15:
            failures['large_temp_differential'] = {
                'probability': min(0.9, temp_diff_target / 20),
                'message': f'Large temperature differential ({temp_diff_target:.1f}°C) between current and target.',
                'severity': 'medium',
                'solution': 'Set a more moderate target or increase system power for faster temperature change.'
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
            "rated_power_kw": self.hvac.power,
            "failures": failures,
            "active_failures": active_failures,
            "has_critical_failure": any(f['severity'] == 'high' and f['probability'] > 0.7 for f in failures.values()),
            "warnings": [f['message'] for f in failures.values() if 0.3 < f['probability'] <= 0.7],
            "critical_alerts": [f['message'] for f in failures.values() if f['probability'] > 0.7]
        }
