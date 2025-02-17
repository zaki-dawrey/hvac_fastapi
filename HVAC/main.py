import asyncio
import random
import os
import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Set

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

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code: {rc}")

mqtt_client.on_connect = on_connect
mqtt_client.connect(MQTT_BROKER, 1883, 60)

def generate_temperature():
    """Simulates temperature sensor data."""
    return round(random.uniform(20.0, 30.0), 2)

async def publish_temperature():
    """Publishes temperature data to MQTT and WebSocket clients."""
    while True:
        temp = generate_temperature()
        mqtt_client.publish(MQTT_TOPIC, str(temp))
        print(f"Published Temperature: {temp}°C")

        # Broadcast to WebSocket clients
        disconnected = set()
        for ws in websockets:
            try:
                await ws.send_text(f"Temperature: {temp}°C")
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
            # Keep the connection alive with ping/pong
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        websockets.remove(websocket)

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
