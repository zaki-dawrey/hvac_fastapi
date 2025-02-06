import asyncio
import random
import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

MQTT_BROKER = "localhost"
MQTT_TOPIC = "sensor/temperature"

app = FastAPI()
websockets: List[WebSocket] = []

# MQTT Client Setup
mqtt_client = mqtt.Client()

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
        for ws in websockets:
            await ws.send_text(f"Temperature: {temp}°C")

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
    asyncio.create_task(publish_temperature())

@app.on_event("shutdown")
async def shutdown_event():
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
