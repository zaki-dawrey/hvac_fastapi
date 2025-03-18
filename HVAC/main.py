import asyncio
import random
import os
import json
import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Set, Dict, Optional
from simulators.simulator_factory import SimulatorFactory

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

simulators = {}

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

# Global variables to track simulation state
is_simulation_running = False
is_simulation_paused = False

async def publish_temperature():
    """Publishes temperature and system status data to MQTT and WebSocket clients."""
    global is_simulation_running
    while True:
        if is_simulation_running and not is_simulation_paused:
            for client_id, simulator in simulators.items():
                temp = generate_temperature(client_id)
                system_status = simulator.get_system_status()
            
                # Prepare message with client id, temperature and system status
                message = {
                    "client_id": client_id,
                    "temperature": temp,
                    "system_status": system_status
                }

                mqtt_client.publish(f"{MQTT_TOPIC}/{client_id}", json.dumps(message))
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
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, system_type: str, client_id: Optional[str] = None):
    await websocket.accept()
    
    # Generate a client ID if none provided
    if not client_id:
        client_id = f"client_{id(websocket)}"
    
    # Store client_id with the websocket
    websocket.client_id = client_id
    
    # Create simulator for this client if it doesn't exist
    if client_id not in simulators:
        # Default parameters - these will be updated by client messages
        room_params = {
            'length': 5.0,
            'breadth': 4.0,
            'height': 2.5,
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
            'fan_speed': 100.0,
        }
        
        simulators[client_id] = SimulatorFactory.create_simulator(system_type, room_params, hvac_params)
    
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
                    global is_simulation_running, is_simulation_paused
                    action = data.get('data', {}).get('action')
                    if action == 'start':
                        is_simulation_running = True
                        is_simulation_paused = False
                        print(f"Simulation started for {client_id}")
                    elif action == 'stop':
                        is_simulation_running = False
                        is_simulation_paused = False
                        print(f"Simulation stopped for {client_id}")
                    elif action == 'pause':
                        is_simulation_paused = True
                        print(f"Simulation paused for {client_id}")
                    elif action == 'resume':
                        is_simulation_paused = False
                        print(f"Simulation resumed for {client_id}")
                    
                    # Send simulation status to client
                    await websocket.send_text(json.dumps({
                        'type': 'simulation_status',
                        'data': {
                            'isRunning': is_simulation_running,
                            'isPaused': is_simulation_paused,
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
                        simulator.room.current_temp = float(params['currentTemp'])
                    if 'targetTemp' in params:
                        simulator.room.target_temp = float(params['targetTemp'])
                    if 'externalTemp' in params:
                        simulator.room.external_temp = float(params['externalTemp'])
                    if 'wallInsulation' in params:
                        simulator.room.wall_insulation = params['wallInsulation']
                    if 'numPeople' in params:
                        simulator.room.num_people = int(params['numPeople'])
                    if 'mode' in params:
                        simulator.room.mode = params['mode']
                    
                    # Handle chilled water specific parameters
                    if system_type == "chilled-water-system" and 'fanCoilUnits' in params:
                        simulator.room.fan_coil_units = int(params['fanCoilUnits'])
                        
                    print(f"Updated room parameters for {client_id}")

                elif data.get('type') == 'hvac_parameters':
                    params = data.get('data', {})
                    # Update common HVAC parameters
                    if 'power' in params:
                        simulator.hvac.power = float(params['power'])
                    if 'cop' in params:
                        simulator.hvac.cop = float(params['cop'])
                    if 'airFlowRate' in params:
                        simulator.hvac.air_flow_rate = float(params['airFlowRate'])
                    if 'supplyTemp' in params:
                        simulator.hvac.supply_temp = float(params['supplyTemp'])
                    if 'fanSpeed' in params:
                        simulator.hvac.fan_speed = float(params['fanSpeed'])
                    
                    # Handle chilled water specific parameters
                    if system_type == "chilled-water-system":
                        if 'chilledWaterFlowRate' in params:
                            simulator.hvac.chilled_water_flow_rate = float(params['chilledWaterFlowRate'])
                        if 'chilledWaterSupplyTemp' in params:
                            simulator.hvac.chilled_water_supply_temp = float(params['chilledWaterSupplyTemp'])
                        if 'chilledWaterReturnTemp' in params:
                            simulator.hvac.chilled_water_return_temp = float(params['chilledWaterReturnTemp'])
                        if 'pumpPower' in params:
                            simulator.hvac.pump_power = float(params['pumpPower'])
                        if 'primarySecondaryLoop' in params:
                            simulator.hvac.primary_secondary_loop = bool(params['primarySecondaryLoop'])
                        if 'glycolPercentage' in params:
                            simulator.hvac.glycol_percentage = float(params['glycolPercentage'])
                        if 'heatExchangerEfficiency' in params:
                            simulator.hvac.heat_exchanger_efficiency = float(params['heatExchangerEfficiency'])
                    
                    print(f"Updated HVAC parameters for {client_id}")

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

@app.post("/api/calculate/{system_type}")
async def calculate_hvac(system_type: str, params: Dict = Body(...), client_id: Optional[str] = Query(None)):
    """Update HVAC parameters and return system status."""
    
    # Use existing simulator or create a new one
    if client_id and client_id in simulators:
        simulator = simulators[client_id]
    
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
    simulator.hvac.supply_temp = float(params.get('supplyTemp', 12.0))
    
    # Handle chilled-water-system specific parameters
    if system_type == "chilled-water-system":
        simulator.room.fan_coil_units = int(params.get('fanCoilUnits', 1))
        simulator.hvac.chilled_water_flow_rate = float(params.get('chilledWaterFlowRate', 0.5))
        simulator.hvac.chilled_water_supply_temp = float(params.get('chilledWaterSupplyTemp', 7.0))
        simulator.hvac.chilled_water_return_temp = float(params.get('chilledWaterReturnTemp', 12.0))
        simulator.hvac.pump_power = float(params.get('pumpPower', 0.75))
        simulator.hvac.primary_secondary_loop = bool(params.get('primarySecondaryLoop', True))
        simulator.hvac.glycol_percentage = float(params.get('glycolPercentage', 0))
        simulator.hvac.heat_exchanger_efficiency = float(params.get('heatExchangerEfficiency', 0.85))
    
    # Return current system status
    return simulator.get_system_status()

@app.on_event("startup")
async def startup_event():
    mqtt_client.loop_start()
    asyncio.create_task(publish_temperature())

@app.on_event("shutdown")
async def shutdown_event():
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)