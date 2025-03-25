import asyncio
import random
import os
import math
import json
import datetime
import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Query, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Set, Dict, Optional
from simulators.simulator_factory import SimulatorFactory
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
from jose.exceptions import JWTError


class MonitoredDict(dict):
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
        return self.change_log

    def add_last_activity_timestamp(self, client_id):
        """Record when a client was last active"""
        if client_id in self.simulation_states:
            self.simulation_states[client_id]["last_active"] = datetime.datetime.now()


# Load environment variables
load_dotenv()
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

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

def verify_supabase_token(token: str) -> dict:
    """Verify Supabase JWT token and return payload."""
    try:
        # Decode without verification first to get the header
        header = jwt.get_unverified_header(token)
        
        # Decode and verify the token
        # Note: Supabase tokens are signed by Supabase, we only verify they're valid JWT
        payload = jwt.decode(
            token,
            key="",
            options={
                "verify_signature": False,  # Skip signature verification
                "verify_aud": False,
                "verify_iss": False
            }
        )

        if 'exp' in payload:
            exp = payload['exp']
            if datetime.datetime.utcnow().timestamp() > exp:
                raise ValueError("Token has expired")

        return payload

    except Exception as e:
        raise ValueError(f"Invalid token: {str(e)}")

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

def normalize_system_type(system_type: str) -> str:
    """Convert kebab-case to snake_case if needed for internal use."""
    # This is useful if you need a consistent format internally
    return system_type.replace("-", "_")


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


# Add a cleanup function
async def cleanup_inactive_simulators():
    """Remove simulators that haven't been used in a while."""
    while True:
        await asyncio.sleep(3600)  # Check every hour
        now = datetime.datetime.now()
        inactive_threshold = now - datetime.timedelta(hours=24)  # 24 hours of inactivity
        
        inactive_clients = []
        for client_id, state in simulators.simulation_states.items():
            last_active = state.get("last_active")
            if last_active and last_active < inactive_threshold:
                inactive_clients.append(client_id)
        
        for client_id in inactive_clients:
            if client_id in simulators:
                del simulators[client_id]
                print(f"Removed inactive simulator for {client_id}")


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

security = HTTPBearer()


@app.websocket("/ws/{auth_id}/{system_type}")
async def websocket_endpoint(
    websocket: WebSocket,
    auth_id: str,
    system_type: str,
    token: str = Query(...),  # Require token as query parameter
):

    try:
        # Verify token manually since we can't use regular dependency injection
        payload = verify_supabase_token(token)
        user_id = payload.get("sub")

        # Verify that the auth_id in the URL matches the user_id in the token
        if not user_id or user_id != auth_id:
            await websocket.close(code=4001, reason="Invalid authentication token")
            return

        print(f"User {auth_id} authenticated successfully")
    except Exception as e:
        print(f"Authentication error: {e}")
        await websocket.close(code=4001, reason="Invalid authentication token")
        return

    await websocket.accept()

    # Normalize system type for internal use
    normalized_system_type = normalize_system_type(system_type)

    # Use combination of user_id and system_type as client_id
    client_id = f"{user_id}_{system_type}"

    # Store client_id with the websocket
    websocket.client_id = client_id
    websocket.system_type = system_type
    websocket.normalized_system_type = normalized_system_type


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
            'fanCoilUnits': 1,
        }

        hvac_params = {
            'power': 3.5,
            'air_flow_rate': 0.5,
            'fan_speed': 50.0,
            'chilledWaterFlowRate': 0.5,
            'chilledWaterSupplyTemp': 7.0,
            'chilledWaterReturnTemp': 12.0,
            'pumpPower': 0.75,
            'primarySecondaryLoop': True,
            'glycolPercentage': 0,
            'heatExchangerEfficiency': 0.85,
        }

        simulators[client_id] = SimulatorFactory.create_simulator(
            normalized_system_type, room_params, hvac_params)
        print(
            f"Created simulator for user {auth_id} with system type {system_type}")

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

                    # Handle chilled water specific parameters
                    if system_type == "chilled-water-system" and 'fanCoilUnits' in params:
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

                    print(f"Updated HVAC parameters for {client_id}")

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


@app.post("/api/{auth_id}/simulations/{system_type}")
async def calculate_hvac(
        auth_id: str, system_type: str, params: Dict = Body(...), credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Update HVAC parameters and return system status."""
        try:
            # Verify token
            payload = verify_supabase_token(credentials.credentials)
            user_id = payload.get("sub")

            # Verify that the auth_id in the URL matches the user_id in the token
            if not user_id or user_id != auth_id:
                return JSONResponse(
                    status_code=401,
                    content={"message": "Invalid authentication token"}
                    )
        except Exception as e:
            return JSONResponse(
                status_code=401,
                content={"message": f"Authentication error: {str(e)}"}
            )

        # Normalize system type for internal use
        normalized_system_type = normalize_system_type(system_type)
        
        # Create client_id
        client_id = f"{auth_id}_{system_type}"
        
        # Use existing simulator or create a new one
        if client_id in simulators:
            simulator = simulators[client_id]
        else:
            # Create a new simulator if it doesn't exist
            room_params = {
                'length': float(params.get('length', 5.0)),
                'breadth': float(params.get('breadth', 4.0)),
                'height': float(params.get('height', 3.0)),
                'current_temp': float(params.get('currentTemp', 25.0)),
                'target_temp': float(params.get('targetTemp', 22.0)),
                'external_temp': float(params.get('externalTemp', 35.0)),
                'wall_insulation': params.get('wallInsulation', 'medium'),
                'num_people': int(params.get('numPeople', 0)),
                'mode': params.get('mode', "cooling"),
                'fanCoilUnits': int(params.get('fanCoilUnits', 1)),
            }
            
            hvac_params = {
                'power': float(params.get('power', 3.5)),
                'cop': float(params.get('cop', 3.0)),
                'air_flow_rate': float(params.get('airFlowRate', 0.5)),
                'fan_speed': float(params.get('fanSpeed', 50.0)),
                'supply_temp': float(params.get('supplyTemp', 12.0)),
                'time_interval': float(params.get('timeInterval', 1.0)),
            }
            
            simulators[client_id] = SimulatorFactory.create_simulator(normalized_system_type, room_params, hvac_params)
            print(f"Created new simulator for user {auth_id} with system type {system_type}")
    

            # Update room parameters
            simulator.room.length = float(params.get('length', 5.0))
            simulator.room.breadth = float(params.get('breadth', 4.0))
            simulator.room.height = float(params.get('height', 3.0))
            simulator.room.current_temp = float(params.get('currentTemp', 25.0))
            simulator.room.target_temp = float(params.get('targetTemp', 22.0))
            simulator.room.external_temp = float(params.get('externalTemp', 35.0))
            simulator.room.wall_insulation = params.get('wallInsulation', 'medium')
            simulator.room.num_people = int(params.get('numPeople', 0))
            simulator.room.mode = params.get('mode', "cooling")

            # Update HVAC parameters
            simulator.hvac.power = float(params.get('power', 3.5))
            simulator.hvac.cop = float(params.get('cop', 3.0))
            simulator.hvac.air_flow_rate = float(params.get('airFlowRate', 0.5))
            simulator.hvac.fan_speed = float(params.get('fanSpeed', 50.0))
            simulator.hvac.supply_temp = float(params.get('supplyTemp', 12.0))

            # Handle chilled-water-system specific parameters
            if system_type == "chilled-water-system":
                if 'chilledWaterFlowRate' in params:
                    simulator.hvac.chilled_water_flow_rate = float(
                        params['chilledWaterFlowRate'])
                if 'waterFlowRate' in params:  # Add this as an alternative name
                    simulator.hvac.chilled_water_flow_rate = float(
                        params['waterFlowRate'])
                simulator.room.fan_coil_units = int(params.get('fanCoilUnits', 1))
                simulator.hvac.chilled_water_flow_rate = float(
                    params.get('chilledWaterFlowRate', 0.5))
                simulator.hvac.chilled_water_supply_temp = float(
                    params.get('chilledWaterSupplyTemp', 7.0))
                simulator.hvac.chilled_water_return_temp = float(
                    params.get('chilledWaterReturnTemp', 12.0))
                simulator.hvac.pump_power = float(params.get('pumpPower', 0.75))
                simulator.hvac.primary_secondary_loop = bool(
                    params.get('primarySecondaryLoop', True))
                simulator.hvac.glycol_percentage = float(
                    params.get('glycolPercentage', 0))
                simulator.hvac.heat_exchanger_efficiency = float(
                    params.get('heatExchangerEfficiency', 0.85))

        # Return current system status
        return simulator.get_system_status()


@app.on_event("startup")
async def startup_event():
    mqtt_client.loop_start()
    asyncio.create_task(publish_temperature())
    asyncio.create_task(cleanup_inactive_simulators())


@app.on_event("shutdown")
async def shutdown_event():
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


print("Simulators: ", simulators)
