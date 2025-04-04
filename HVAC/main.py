"""FastAPI application for HVAC simulation control and monitoring."""

import asyncio
import os
import math
import json
import datetime
from dotenv import load_dotenv
from typing import Set, Dict, Optional
import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from simulators.simulator_factory import SimulatorFactory


class MonitoredDict(dict):
    """A dictionary that tracks changes and simulation state for each client."""

    def __init__(self, *args, **kwargs):
        self.change_log = []
        self.simulation_states = {}  # Track simulation state per client
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        action = "updated" if key in self else "added"
        self.change_log.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "key": key,
            "value_summary": str(value)[:100]  # Truncated summary
        })
        super().__setitem__(key, value)

        # Initialize simulation state for new clients
        if key not in self.simulation_states:
            self.simulation_states[key] = {
                "is_running": False,
                "is_paused": False
            }

        print(
            f"[MONITOR] {action.upper()} key '{key}' at {self.change_log[-1]['timestamp']}")

    def __delitem__(self, key):
        self.change_log.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "deleted",
            "key": key
        })
        if key in self.simulation_states:
            del self.simulation_states[key]
        super().__delitem__(key)
        print(
            f"[MONITOR] DELETED key '{key}' at {self.change_log[-1]['timestamp']}")

    def set_simulation_state(self, client_id, is_running, is_paused):
        """Set simulation state for a specific client"""
        if client_id in self:
            if client_id not in self.simulation_states:
                self.simulation_states[client_id] = {}
            self.simulation_states[client_id]["is_running"] = is_running
            self.simulation_states[client_id]["is_paused"] = is_paused
            print(
                f"[MONITOR] Simulation state for '{client_id}': running={is_running}, paused={is_paused}")

    def get_simulation_state(self, client_id):
        """Get simulation state for a specific client"""
        if client_id in self.simulation_states:
            return self.simulation_states[client_id]
        return {"is_running": False, "is_paused": False}

    def get_active_clients(self):
        """Get all clients with running simulations"""
        return [
            client_id for client_id, state in self.simulation_states.items()
            if state.get("is_running", False) and not state.get("is_paused", False)
        ]

    def get_changes(self):
        """Get all changes to the dictionary"""
        return self.change_log


MQTT_BROKER = "localhost"
MQTT_TOPIC = "sensor/temperature"

app = FastAPI()
websockets: Set[WebSocket] = set()
mqtt_client = mqtt.Client()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

simulators = MonitoredDict()


def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code: {rc}")


mqtt_client.on_connect = on_connect
mqtt_client.connect(MQTT_BROKER, 1883, 60)


def generate_temperature(client_id=None):
    """Simulates temperature sensor data using HVAC simulator."""
    if client_id and client_id in simulators:
        simulator = simulators[client_id]
        new_temp = simulator.calculate_temperature_change()
        simulator.room.current_temp = new_temp
        return round(new_temp, 2)
    else:
        print(f"Warning: No simulator found for client_id: {client_id}")
        return 25.0  # Default temperature

# # Global variables to track simulation state
# is_simulation_running = False
# is_simulation_paused = False


async def publish_temperature():
    """Publishes temperature and system status data to MQTT and WebSocket clients."""
    while True:
        # Get all actively running client simulations
        active_clients = simulators.get_active_clients()

        for client_id in active_clients:
            if client_id in simulators:
                simulator = simulators[client_id]
                temp = generate_temperature(client_id)
                system_status = simulator.get_system_status()

                # Prepare message with client id, temperature and system status
                message = {
                    "client_id": client_id,
                    "temperature": temp,
                    "system_status": system_status
                }

                mqtt_client.publish(
                    f"{MQTT_TOPIC}/{client_id}", json.dumps(message))
                print(f"Published Temperature for {client_id}: {temp}°C")

                # Broadcast to WebSocket clients
                disconnected = set()
                for ws in websockets:
                    try:
                        # Check if this websocket is associated with this client_id
                        if hasattr(ws, 'client_id') and ws.client_id == client_id:
                            await ws.send_text(json.dumps(message))
                    except:
                        disconnected.add(ws)

                # Remove disconnected clients
                websockets.difference_update(disconnected)

        await asyncio.sleep(2)


@app.get("/")
async def root():
    """Serve the static index.html file."""
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.websocket("/ws/{user_id}/{system_type}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, system_type: str,):
    """WebSocket endpoint for HVAC simulation control and data."""
    await websocket.accept()

    client_id = f"{user_id}_{system_type}"

    # Store client_id with the websocket
    websocket.client_id = client_id

    # Create simulator for this client if it doesn't exist
    if client_id not in simulators:
        # Default parameters - these will be updated by client messages
        room_params = {
            'length': 5.0,
            'breadth': 4.0,
            'height': 3.0,
            'current_temp': 25.0,
            'target_temp': 22.0,
            'external_temp': 35.0,
            'wall_insulation': 'medium',
            'num_people': 0,
            'mode': 'cooling',
        }

        hvac_params = {
            'power': 3.5,
            'air_flow_rate': 0.5,
            'fan_speed': 50.0,
        }

        simulators[client_id] = SimulatorFactory.create_simulator(
            system_type, room_params, hvac_params)

    websockets.add(websocket)

    try:
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                print(f"Received message from {client_id}: {data}")

                # Get the simulator for this client
                simulator = simulators.get(client_id)

                if data.get('type') == 'simulation_control':
                    action = data.get('data', {}).get('action')
                    # Get current simulation state
                    sim_state = simulators.get_simulation_state(client_id)
                    is_running = sim_state["is_running"]
                    is_paused = sim_state["is_paused"]

                    if action == 'start':
                        is_running = True
                        is_paused = False
                        print(f"Simulation started for {client_id}")
                    elif action == 'stop':
                        is_running = False
                        is_paused = False
                        print(f"Simulation stopped for {client_id}")
                    elif action == 'pause':
                        is_paused = True
                        print(f"Simulation paused for {client_id}")
                    elif action == 'resume':
                        is_paused = False
                        print(f"Simulation resumed for {client_id}")

                    # Update simulation state
                    simulators.set_simulation_state(
                        client_id, is_running, is_paused)

                    # Send simulation status to client
                    await websocket.send_text(json.dumps({
                        'type': 'simulation_status',
                        'data': {
                            'isRunning': is_running,
                            'isPaused': is_paused,
                            'estimatedTimeToTarget': simulator.calculate_time_to_target()
                        }
                    }))

                elif data.get('type') == 'room_parameters':
                    params = data.get('data', {})
                    # Update all room parameters
                    if 'length' in params:
                        simulator.room.length = float(params['length'])
                    if 'breadth' in params:
                        simulator.room.breadth = float(params['breadth'])
                    if 'height' in params:
                        simulator.room.height = float(params['height'])
                    if 'currentTemp' in params:
                        simulator.room.current_temp = float(
                            params['currentTemp'])
                    if 'targetTemp' in params:
                        simulator.room.target_temp = float(
                            params['targetTemp'])
                    if 'externalTemp' in params:
                        simulator.room.external_temp = float(
                            params['externalTemp'])
                    if 'wallInsulation' in params:
                        simulator.room.wall_insulation = params['wallInsulation']
                    if 'numPeople' in params:
                        simulator.room.num_people = int(params['numPeople'])
                    if 'mode' in params:
                        simulator.room.mode = params['mode']
                    if 'humidity' in params:
                        simulator.room.humidity = float(params['humidity'])
                    if 'heatGainExternal' in params:
                        simulator.room.heat_gain_external = float(
                            params['heatGainExternal'])

                    # Handle chilled water specific parameters
                    if system_type == "chilled-water-system":
                        if 'fanCoilUnits' in params:
                            simulator.room.fan_coil_units = int(
                                params['fanCoilUnits'])

                    print(f"Updated room parameters for {client_id}")

                elif data.get('type') == 'hvac_parameters':
                    params = data.get('data', {})
                    # Update common HVAC parameters
                    if 'power' in params:
                        simulator.hvac.power = float(params['power'])
                    if 'cop' in params:
                        simulator.hvac.cop = float(params['cop'])
                    if 'airFlowRate' in params:
                        simulator.hvac.air_flow_rate = float(
                            params['airFlowRate'])
                    if 'supplyTemp' in params:
                        simulator.hvac.supply_temp = float(
                            params['supplyTemp'])
                    if 'fanSpeed' in params:
                        simulator.hvac.fan_speed = float(params['fanSpeed'])

                    # Handle chilled water specific parameters
                    if system_type == "chilled-water-system":
                        if 'chilledWaterFlowRate' in params:
                            simulator.hvac.chilled_water_flow_rate = float(
                                params['chilledWaterFlowRate'])
                        if 'chilledWaterSupplyTemp' in params:
                            simulator.hvac.chilled_water_supply_temp = float(
                                params['chilledWaterSupplyTemp'])
                        if 'chilledWaterReturnTemp' in params:
                            simulator.hvac.chilled_water_return_temp = float(
                                params['chilledWaterReturnTemp'])
                        if 'pumpPower' in params:
                            simulator.hvac.pump_power = float(
                                params['pumpPower'])
                        if 'primarySecondaryLoop' in params:
                            simulator.hvac.primary_secondary_loop = bool(
                                params['primarySecondaryLoop'])
                        if 'glycolPercentage' in params:
                            simulator.hvac.glycol_percentage = float(
                                params['glycolPercentage'])
                        if 'heatExchangerEfficiency' in params:
                            simulator.hvac.heat_exchanger_efficiency = float(
                                params['heatExchangerEfficiency'])

                    if system_type == "variable-refrigerant-flow-system":
                        if 'maxCapacityKw' in params:
                            simulator.hvac.max_capacity_kw = float(
                                params['maxCapacityKw'])
                        if 'minCapacityKw' in params:
                            simulator.hvac.min_capacity_kw = float(
                                params['minCapacityKw'])
                        if 'heatRecovery' in params:
                            # Boolean conversion with explicit check
                            heat_recovery = params['heatRecovery']
                            if isinstance(heat_recovery, bool):
                                simulator.hvac.heat_recovery = heat_recovery
                            elif isinstance(heat_recovery, str):
                                simulator.hvac.heat_recovery = heat_recovery.lower() == 'true'
                            else:
                                simulator.hvac.heat_recovery = bool(
                                    heat_recovery)
                        if 'heatRecoveryPercentage' in params:
                            simulator.hvac.heat_recovery_percentage = float(
                                params['heatRecoveryPercentage'])
                        if 'timeInterval' in params:
                            simulator.hvac.time_interval = float(
                                params['timeInterval'])

                        # Handle zones update
                        if 'zones' in params:
                            new_zones = params['zones']
                            # Update existing zones
                            simulator.hvac.zones = new_zones
                            simulator.total_demand = sum(new_zones.values())

                            # Re-initialize zone parameters
                            simulator.zone_parameters = simulator._initialize_zone_parameters()

                        # Update total demand
                        simulator.total_demand = sum(
                            simulator.hvac.zones.values())

                    if system_type == "heat-pump-system":
                        if 'copRated' in params:
                            simulator.hvac.cop_rated = float(
                                params['copRated'])
                        if 'copMin' in params:
                            simulator.hvac.cop_min = float(params['copMin'])
                        if 'supplyTempCooling' in params:
                            simulator.hvac.supply_temp_cooling = float(
                                params['supplyTempCooling'])
                        if 'supplyTempHeating' in params:
                            simulator.hvac.supply_temp_heating = float(
                                params['supplyTempHeating'])
                        if 'defrostTempThreshold' in params:
                            simulator.hvac.defrost_temp_threshold = float(
                                params['defrostTempThreshold'])
                        if 'defrostCycleTime' in params:
                            simulator.hvac.defrost_cycle_time = float(
                                params['defrostCycleTime'])
                        if 'defrostInterval' in params:
                            simulator.hvac.defrost_interval = float(
                                params['defrostInterval'])
                        if 'refrigerantType' in params:
                            simulator.hvac.refrigerant_type = params['refrigerantType']

                    print(f"Updated HVAC parameters for {client_id}")

                elif data.get('type') == 'zone_parameters':
                    params = data.get('data', {})
                    zone_name = params.get('zone_name')

                    if zone_name and zone_name in simulator.zone_parameters:
                        # Update zone target temperature
                        if 'target_temp' in params:
                            simulator.adjust_zone_target_temp(
                                zone_name, float(params['target_temp']))

                        # Update zone mode (only if heat recovery is enabled)
                        if 'mode' in params and simulator.hvac.heat_recovery:
                            simulator.set_zone_mode(zone_name, params['mode'])

                        # Send immediate feedback
                        zone_status = simulator.get_zone_status(zone_name)
                        await websocket.send_text(json.dumps({
                            'type': 'zone_update',
                            'data': {
                                'zone_name': zone_name,
                                'zone_status': zone_status
                            }
                        }))
                        print(f"Updated zone parameters for {zone_name}")

                elif data.get('type') == 'add_zone':
                    params = data.get('data', {})
                    zone_name = params.get(
                        'name', f"Zone {len(simulator.hvac.zones) + 1}")
                    demand_kw = float(params.get('demand', 5.0))
                    target_temp = float(params.get(
                        'target_temp', simulator.room.target_temp))

                    # Add new zone
                    success = simulator.add_zone(
                        zone_name, demand_kw, target_temp)

                    # Send feedback
                    await websocket.send_text(json.dumps({
                        'type': 'zone_operation',
                        'data': {
                            'action': 'add',
                            'success': success,
                            'zone_name': zone_name,
                            'zones': simulator.hvac.zones,
                            'zone_data': simulator.get_all_zone_status()
                        }
                    }))
                    print(
                        f"Added zone {zone_name}" if success else f"Failed to add zone {zone_name}")

                elif data.get('type') == 'remove_zone':
                    params = data.get('data', {})
                    zone_name = params.get('name')

                    if zone_name:
                        # Remove zone
                        success = simulator.remove_zone(zone_name)

                        # Send feedback
                        await websocket.send_text(json.dumps({
                            'type': 'zone_operation',
                            'data': {
                                'action': 'remove',
                                'success': success,
                                'zone_name': zone_name,
                                'zones': simulator.hvac.zones,
                                'zone_data': simulator.get_all_zone_status()
                            }
                        }))
                        print(
                            f"Removed zone {zone_name}" if success else f"Failed to remove zone {zone_name}")

                elif data.get('type') == 'get_status':
                    # Send immediate status update
                    system_status = simulator.get_system_status()

                    # Include time estimate if requested
                    time_to_target = None
                    if data.get('include_time_estimate', False):
                        time_to_target = simulator.calculate_time_to_target()

                    response_data = {
                        'client_id': client_id,
                        'system_status': system_status
                    }

                    # Add simulation status with time estimate if available
                    if time_to_target is not None:
                        response_data['type'] = 'simulation_status'
                        response_data['data'] = {
                            'isRunning': simulators.get_simulation_state(client_id)["is_running"],
                            'isPaused': simulators.get_simulation_state(client_id)["is_paused"],
                            'estimatedTimeToTarget': time_to_target
                        }

                    await websocket.send_text(json.dumps(response_data))

                # In the section where you handle get_time_to_target
                elif data.get('type') == 'get_time_to_target':
                    time_to_target = simulator.calculate_time_to_target()
                    can_reach_target = simulator.can_reach_target()

                    # Handle float('inf') by converting to string before JSON encoding
                    if isinstance(time_to_target, float) and math.isinf(time_to_target):
                        estimated_time = "Cannot reach target"
                    else:
                        estimated_time = time_to_target

                    await websocket.send_text(json.dumps({
                        'type': 'simulation_status',
                        'data': {
                            'isRunning': simulators.get_simulation_state(client_id)["is_running"],
                            'isPaused': simulators.get_simulation_state(client_id)["is_paused"],
                            'estimatedTimeToTarget': estimated_time,
                            'canReachTarget': can_reach_target
                        }
                    }))

                # Send immediate feedback
                system_status = simulator.get_system_status()
                await websocket.send_text(json.dumps({
                    'client_id': client_id,
                    'system_status': system_status
                }))

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError as e:
                print(f"Invalid JSON received from {client_id}: {e}")
            except Exception as e:
                print(f"Error processing message from {client_id}: {e}")
    except Exception as e:
        print(f"WebSocket error for {client_id}: {e}")
    finally:
        websockets.remove(websocket)
        # Keep simulator in memory for reconnection
        print(f"Client {client_id} disconnected")


@app.on_event("startup")
async def startup_event():
    """Start the MQTT client and temperature publishing task."""
    mqtt_client.loop_start()
    asyncio.create_task(publish_temperature())


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the MQTT client and temperature publishing task."""
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


print("Simulators: ", simulators)
