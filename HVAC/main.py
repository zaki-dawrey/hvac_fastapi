import asyncio
import random
import paho.mqtt.client as mqtt
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

MQTT_BROKER = "localhost"
MQTT_TOPIC = "hvac/environment/current"

app = FastAPI()
websockets: List[WebSocket] = []

# MQTT Client Setup
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code: {rc}")

mqtt_client.on_connect = on_connect
mqtt_client.connect(MQTT_BROKER, 1883, 60)

def generate_sensor_data():
    """Simulates temperature and humidity sensor data."""
    return {
        "temperature": round(random.uniform(20.0, 30.0), 2),
        "humidity": round(random.uniform(30.0, 70.0), 2)
    }

async def publish_sensor_data():
    """Publishes environmental data to MQTT and WebSocket clients."""
    while True:
        sensor_data = generate_sensor_data()
        mqtt_client.publish(MQTT_TOPIC, json.dumps(sensor_data))
        print(f"Published - Temp: {sensor_data['temperature']}°C, Humidity: {sensor_data['humidity']}%")

        # Broadcast to WebSocket clients
        for ws in websockets:
            await ws.send_text(json.dumps(sensor_data))

        await asyncio.sleep(2)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websockets.append(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        websockets.remove(websocket)

@app.on_event("startup")
async def startup_event():
    mqtt_client.loop_start()
    asyncio.create_task(publish_sensor_data())

@app.on_event("shutdown")
async def shutdown_event():
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
