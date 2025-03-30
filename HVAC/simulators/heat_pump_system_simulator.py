"""Simulator for a heat pump system."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HeatPumpRoomParameters:
    """Data class to hold room parameters for heat pump system."""
    length: float  # meters
    breadth: float  # meters
    height: float  # meters
    current_temp: float  # °C
    target_temp: float  # °C
    external_temp: float = 35.0  # °C
    wall_insulation: str = "medium"  # Insulation level (low/medium/high)
    num_people: int = 0  # Number of people in room
    mode: str = "cooling"  # cooling/heating mode
    humidity: float = 50.0  # Target humidity (%)
    # External heat gain in Watts (sunlight, appliances)
    heat_gain_external: float = 0.0


@dataclass
class HeatPumpHVACParameters:
    """Data class to hold HVAC parameters for heat pump system."""
    power: float = 3.5  # kW - rated power
    cop_rated: float = 3.5  # Rated Coefficient of Performance at standard conditions
    cop_min: float = 1.5  # Minimum COP at extreme temperatures
    air_flow_rate: float = 0.5  # m³/s
    supply_temp_cooling: float = 12.0  # °C - Supply air temperature in cooling mode
    supply_temp_heating: float = 45.0  # °C - Supply air temperature in heating mode
    fan_speed: float = 50.0  # Fan speed percentage
    time_interval: float = 1.0  # Simulation update interval in seconds
    # Temperature threshold for defrost cycle (°C)
    defrost_temp_threshold: float = 5.0
    # Defrost cycle time in seconds (10 minutes)
    defrost_cycle_time: float = 10 * 60
    # Time between defrost cycles in seconds (60 minutes)
    defrost_interval: float = 60 * 60
    refrigerant_type: str = "R410A"  # Type of refrigerant used


class HeatPumpSystemSimulator:
    """Simulator class for a heat pump system in a room."""

    def __init__(self, room: HeatPumpRoomParameters, hvac: HeatPumpHVACParameters):
        self.room = room
        self.hvac = hvac
        self.specific_heat_air = 1.005  # kJ/kg·K
        self.air_density = 1.225  # kg/m³
        self.debug_info = []  # To store debug information during calculations
        self.time_since_defrost = 0  # Time since last defrost cycle in seconds
        self.in_defrost_mode = False  # Indicates if system is currently in defrost mode
        self.defrost_remaining_time = 0  # Remaining time in defrost cycle

    @property
    def room_volume(self) -> float:
        """Calculate room volume in cubic meters."""
        return self.room.length * self.room.breadth * self.room.height

    @property
    def room_air_mass(self) -> float:
        """Calculate mass of air in the room."""
        return self.room_volume * self.air_density

    @property
    def supply_temp(self) -> float:
        """Get appropriate supply temperature based on operation mode."""
        if self.room.mode.lower() == "cooling":
            return self.hvac.supply_temp_cooling
        else:  # heating mode
            return self.hvac.supply_temp_heating

    def calculate_cop(self) -> float:
        """Calculate actual COP based on external temperature and mode."""
        # Different COP calculation based on mode
        if self.room.mode.lower() == "cooling":
            # COP decreases as external temperature increases
            # Typical reference is 35°C for cooling
            reference_temp = 35.0
            # Decrease COP by approximately 2-3% per degree above reference
            temp_factor = max(
                0, 1.0 - 0.025 * max(0, self.room.external_temp - reference_temp))
        else:  # heating mode
            # COP decreases as external temperature decreases
            # Typical reference is 7°C for heating
            reference_temp = 7.0
            # Decrease COP by approximately 3-4% per degree below reference
            temp_factor = max(
                0, 1.0 - 0.035 * max(0, reference_temp - self.room.external_temp))

        # Calculate actual COP, ensuring it doesn't go below minimum
        actual_cop = max(self.hvac.cop_min, self.hvac.cop_rated * temp_factor)

        # If in defrost mode, COP is effectively reduced
        if self.in_defrost_mode:
            actual_cop *= 0.5  # 50% efficiency during defrost

        return actual_cop

    def calculate_defrost_state(self, time_step: float) -> None:
        """Update defrost cycle state."""
        # Only consider defrost in heating mode
        if self.room.mode.lower() != "heating":
            self.in_defrost_mode = False
            self.defrost_remaining_time = 0
            return

        # Update defrost timers
        if self.in_defrost_mode:
            self.defrost_remaining_time -= time_step
            if self.defrost_remaining_time <= 0:
                self.in_defrost_mode = False
                self.time_since_defrost = 0
        else:
            self.time_since_defrost += time_step

        # Check if defrost should be activated
        if (not self.in_defrost_mode and
            self.time_since_defrost >= self.hvac.defrost_interval and
                self.room.external_temp <= self.hvac.defrost_temp_threshold):
            self.in_defrost_mode = True
            self.defrost_remaining_time = self.hvac.defrost_cycle_time

    def calculate_heating_cooling_capacity(self, at_temp=None) -> float:
        """Calculate capacity in Watts with improved heat pump logic."""
        temp = at_temp if at_temp is not None else self.room.current_temp

        # Get actual COP based on conditions
        actual_cop = self.calculate_cop()

        # Calculate theoretical maximum capacity based on rated power
        rated_capacity = self.hvac.power * 1000  # Convert kW to Watts

        # Calculate capacity based on air flow and temperature differential
        airflow_capacity = (
            self.hvac.air_flow_rate
            * self.specific_heat_air
            * 1000  # Convert to Watts
            * abs(temp - self.supply_temp)
        )

        # Take the lesser of the two capacities (limiting factor)
        # Adjust for fan speed
        nominal_capacity = min(
            rated_capacity, airflow_capacity) * (self.hvac.fan_speed / 100.0)

        # Apply temperature-dependent derating for heating mode
        if self.room.mode.lower() == "heating":
            # Heat pump capacity typically decreases in very cold weather
            if self.room.external_temp < -5:
                capacity_factor = 0.6  # Severe capacity reduction below -5°C
            elif self.room.external_temp < 0:
                capacity_factor = 0.75  # Moderate reduction between 0 and -5°C
            elif self.room.external_temp < 5:
                capacity_factor = 0.9  # Small reduction between 0 and 5°C
            else:
                capacity_factor = 1.0  # Normal capacity above 5°C

            nominal_capacity *= capacity_factor

        # If in defrost mode during heating, we're actually cooling the indoor space
        if self.in_defrost_mode and self.room.mode.lower() == "heating":
            # Reverse capacity direction (negative value = heat extraction)
            # Typical defrost uses about 30% of normal capacity to reverse flow
            return -nominal_capacity * 0.3

        return nominal_capacity

    def calculate_heat_gain(self, at_temp=None) -> float:
        """Calculate total heat gain/loss in Watts."""
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

        # Calculate environmental heat transfer
        temperature_difference = self.room.external_temp - temp
        environmental_gain = surface_area * insulation_factor * temperature_difference

        # Add heat from people (approximately 100W per person)
        people_heat = self.room.num_people * 100

        # Add external heat gain (from sunlight, appliances, etc)
        total_heat_gain = environmental_gain + \
            people_heat + self.room.heat_gain_external

        return total_heat_gain

    def calculate_energy_consumption(self, capacity: float) -> float:
        """Calculate energy consumption in Watts."""
        # Get the actual COP for current conditions
        actual_cop = self.calculate_cop()

        # In defrost mode, add extra energy for defrosting
        if self.in_defrost_mode:
            # ~20% extra energy for defrost cycle
            defrost_energy = abs(capacity) * 0.2
            return abs(capacity) / actual_cop + defrost_energy

        return abs(capacity) / actual_cop

    def calculate_refrigerant_flow(self, capacity: float) -> float:
        """Calculate refrigerant flow rate in g/s."""
        # Different enthalpy values based on refrigerant and mode
        refrigerant_enthalpies = {
            "R410A": {"cooling": 20, "heating": 18},
            "R32": {"cooling": 22, "heating": 20},
            "R290": {"cooling": 25, "heating": 23}
        }

        # Get enthalpy difference for current refrigerant and mode
        refrigerant = self.hvac.refrigerant_type
        mode = self.room.mode.lower()

        # Default values if refrigerant not recognized
        if refrigerant not in refrigerant_enthalpies:
            refrigerant = "R410A"

        enthalpy_difference = refrigerant_enthalpies[refrigerant][mode]

        # Convert to g/s
        return (abs(capacity) / 1000) / enthalpy_difference * 1000

    def calculate_net_heat_at_temp(self, temp: float) -> float:
        """Calculate net heat transfer (in Watts) at a specific temperature."""
        # Calculate components at specified temperature
        capacity = self.calculate_heating_cooling_capacity(at_temp=temp)
        heat_gain = self.calculate_heat_gain(at_temp=temp)

        # Calculate net heat based on mode
        if self.room.mode.lower() == "heating":
            # In heating mode, capacity adds heat to room (positive)
            net_heat = heat_gain + capacity
        else:  # cooling mode
            # In cooling mode, capacity removes heat from room (negative)
            net_heat = heat_gain - capacity

        return net_heat

    def calculate_temp_change_rate(self, temp: float) -> float:
        """Calculate rate of temperature change (°C/s) at a specific temperature."""
        net_heat = self.calculate_net_heat_at_temp(temp)
        rate = net_heat / (self.room_air_mass * self.specific_heat_air * 1000)
        return rate

    def calculate_temperature_change(self) -> float:
        """Calculate new room temperature after one time step (in seconds)."""
        # Update defrost cycle state
        self.calculate_defrost_state(self.hvac.time_interval)

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
        current_defrost_time = self.time_since_defrost
        in_defrost = self.in_defrost_mode
        defrost_remaining = self.defrost_remaining_time

        # Use small, fixed time step for accurate integration
        time_step = 60.0  # 60 seconds per step
        max_time = 24 * 3600  # 24 hours max simulation
        max_steps = int(max_time / time_step)

        for step in range(max_steps):
            # Update defrost state for simulation
            if in_defrost:
                defrost_remaining -= time_step
                if defrost_remaining <= 0:
                    in_defrost = False
                    current_defrost_time = 0
            else:
                current_defrost_time += time_step
                if (self.room.mode.lower() == "heating" and
                    current_defrost_time >= self.hvac.defrost_interval and
                        self.room.external_temp <= self.hvac.defrost_temp_threshold):
                    in_defrost = True
                    defrost_remaining = self.hvac.defrost_cycle_time

            # Calculate rate at current temperature considering defrost state
            # We need to temporarily set the defrost state for calculation
            original_defrost = self.in_defrost_mode
            self.in_defrost_mode = in_defrost
            rate = self.calculate_temp_change_rate(current_temp)
            self.in_defrost_mode = original_defrost

            # Check if we can still make progress
            if (cooling_mode and rate >= 0) or (not cooling_mode and rate <= 0):
                if in_defrost and self.room.mode.lower() == "heating":
                    # If we're in defrost, just continue - this is temporary
                    pass
                else:
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
                    "time_elapsed": total_time,
                    "in_defrost": in_defrost
                })

        # If we get here, we couldn't reach target within max simulation time
        return float('inf')

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and calculations."""
        capacity = self.calculate_heating_cooling_capacity()
        actual_cop = self.calculate_cop()
        energy_consumption = self.calculate_energy_consumption(capacity)
        refrigerant_flow = self.calculate_refrigerant_flow(capacity)
        time_to_target = self.calculate_time_to_target()

        return {
            "room_temperature": round(self.room.current_temp, 2),
            "target_temperature": self.room.target_temp,
            "external_temperature": self.room.external_temp,
            f"{self.room.mode}_capacity_kw": round(abs(capacity) / 1000, 2),
            f"{self.room.mode}_capacity_btu": round(abs(capacity) * 3.412, 2),
            "energy_consumption_w": round(energy_consumption, 2),
            "refrigerant_flow_gs": round(refrigerant_flow, 2),
            "heat_gain_w": round(self.calculate_heat_gain(), 2),
            "rated_cop": self.hvac.cop_rated,
            "actual_cop": round(actual_cop, 2),
            "cop_reduction_factor": round(actual_cop / self.hvac.cop_rated, 2),
            "mode": self.room.mode,
            "defrost_active": self.in_defrost_mode,
            "defrost_remaining_time": round(self.defrost_remaining_time / 60, 1) if self.in_defrost_mode else 0,
            "time_since_defrost": round(self.time_since_defrost / 60, 1),
            "fan_speed": self.hvac.fan_speed,
            "humidity": self.room.humidity,
            "num_people": self.room.num_people,
            "external_heat_gain": self.room.heat_gain_external,
            "insulation_level": self.room.wall_insulation,
            "time_interval": self.hvac.time_interval,
            "room_size": round(self.room.length * self.room.breadth, 2),
            "time_to_target": time_to_target if time_to_target != float('inf') else "Cannot reach target",
            "can_reach_target": self.can_reach_target(),
            # °C/hour
            "temp_change_rate": round(self.calculate_temp_change_rate(self.room.current_temp) * 3600, 4),
            "rated_power_kw": self.hvac.power,
            "refrigerant_type": self.hvac.refrigerant_type,
            "supply_temperature": round(self.supply_temp, 1)
        }
