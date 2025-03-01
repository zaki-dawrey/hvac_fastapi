import asyncio
import random
import os
import json
import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Set, Dict
from hvac_simulator import HVACSimulator, RoomParameters, HVACParameters

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

# Initialize HVAC simulator with default values
room_params = RoomParameters(
    length=5.0,
    breadth=4.0,
    height=2.5,
    current_temp=25.0,
    target_temp=22.0
)

hvac_params = HVACParameters(
    power=5.0
)

hvac_simulator = HVACSimulator(room_params, hvac_params)

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code: {rc}")

mqtt_client.on_connect = on_connect
mqtt_client.connect(MQTT_BROKER, 1883, 60)

def generate_temperature():
    """Simulates temperature sensor data using HVAC simulator."""
    global hvac_simulator
    new_temp = hvac_simulator.calculate_temperature_change()
    hvac_simulator.room.current_temp = new_temp
    return round(new_temp, 2)

# Global variables to track simulation state
is_simulation_running = False
is_simulation_paused = False

async def publish_temperature():
    """Publishes temperature and system status data to MQTT and WebSocket clients."""
    global is_simulation_running
    while True:
        if is_simulation_running and not is_simulation_paused:
            temp = generate_temperature()
            system_status = hvac_simulator.get_system_status()
            
            # Prepare message with temperature and system status
            message = {
                "temperature": temp,
                "system_status": system_status
            }
            
            mqtt_client.publish(MQTT_TOPIC, json.dumps(message))
            print(f"Published Temperature: {temp}°C")

            # Broadcast to WebSocket clients
            disconnected = set()
            for ws in websockets:
                try:
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
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websockets.add(websocket)
    try:
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                print(f"Received message: {data}")

                if data.get('type') == 'simulation_control':
                    global is_simulation_running, is_simulation_paused
                    action = data.get('data', {}).get('action')
                    if action == 'start':
                        is_simulation_running = True
                        is_simulation_paused = False
                        print("Simulation started")
                    elif action == 'stop':
                        is_simulation_running = False
                        is_simulation_paused = False
                        print("Simulation stopped")
                    elif action == 'pause':
                        is_simulation_paused = True
                        print("Simulation paused")
                    elif action == 'resume':
                        is_simulation_paused = False
                        print("Simulation resumed")
                    # Send simulation status to client
                    await websocket.send_text(json.dumps({
                        'type': 'simulation_status',
                        'data': {
                            'isRunning': is_simulation_running,
                            'isPaused': is_simulation_paused,
                            'estimatedTimeToTarget': hvac_simulator.calculate_time_to_target()
                        }
                    }))

                elif data.get('type') == 'room_parameters':
                    params = data.get('data', {})
                    # Update all room parameters
                    if 'length' in params:
                        hvac_simulator.room.length = float(params['length'])
                    if 'breadth' in params:
                        hvac_simulator.room.breadth = float(params['breadth'])
                    if 'height' in params:
                        hvac_simulator.room.height = float(params['height'])
                    if 'currentTemp' in params:
                        hvac_simulator.room.current_temp = float(params['currentTemp'])
                    if 'targetTemp' in params:
                        hvac_simulator.room.target_temp = float(params['targetTemp'])
                    if 'externalTemp' in params:
                        hvac_simulator.room.external_temp = float(params['externalTemp'])
                    if 'wallInsulation' in params:
                        hvac_simulator.room.wall_insulation = params['wallInsulation']
                    if 'numPeople' in params:
                        hvac_simulator.room.num_people = int(params['numPeople'])
                    if 'mode' in params:
                        hvac_simulator.room.mode = params['mode']
                    print(f"Updated room parameters")

                elif data.get('type') == 'hvac_parameters':
                    params = data.get('data', {})
                    # Update all HVAC parameters
                    if 'power' in params:
                        hvac_simulator.hvac.power = float(params['power'])
                    if 'cop' in params:
                        hvac_simulator.hvac.cop = float(params['cop'])
                    if 'airFlowRate' in params:
                        hvac_simulator.hvac.air_flow_rate = float(params['airFlowRate'])
                    if 'supplyTemp' in params:
                        hvac_simulator.hvac.supply_temp = float(params['supplyTemp'])
                    print(f"Updated HVAC parameters")

                # Send immediate feedback
                system_status = hvac_simulator.get_system_status()
                await websocket.send_text(json.dumps({
                    'system_status': system_status
                }))

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError as e:
                print(f"Invalid JSON received: {e}")
            except Exception as e:
                print(f"Error processing message: {e}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        websockets.remove(websocket)
        print("Client disconnected")

@app.post("/api/calculate")
async def calculate_hvac(params: Dict = Body(...)):
    """Update HVAC parameters and return system status."""
    global hvac_simulator
    
    # Update room parameters
    hvac_simulator.room.length = float(params.get('length', 5.0))
    hvac_simulator.room.breadth = float(params.get('breadth', 4.0))
    hvac_simulator.room.height = float(params.get('height', 3.0))
    hvac_simulator.room.current_temp = float(params.get('currentTemp', 25.0))
    hvac_simulator.room.target_temp = float(params.get('targetTemp', 22.0))
    hvac_simulator.room.external_temp = float(params.get('externalTemp', 35.0))
    hvac_simulator.room.wall_insulation = params.get('wallInsulation', 'medium')
    hvac_simulator.room.num_people = int(params.get('numPeople', 0))
    hvac_simulator.room.mode = params.get('mode', 'cooling')
    
    # Update HVAC parameters
    hvac_simulator.hvac.power = float(params.get('power', 3.5))
    hvac_simulator.hvac.cop = float(params.get('cop', 3.0))
    hvac_simulator.hvac.air_flow_rate = float(params.get('airFlowRate', 0.5))
    hvac_simulator.hvac.supply_temp = float(params.get('supplyTemp', 12.0))
    
    # Return current system status
    return hvac_simulator.get_system_status()

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
