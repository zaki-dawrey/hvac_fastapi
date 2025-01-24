from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import paho.mqtt.client as mqtt

app = FastAPI()

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Create MQTT client
mqtt_client = mqtt.Client()

# Pydantic model for temperature payload
class TemperatureData(BaseModel):
    value: float

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code: {rc}")

def on_message(client, userdata, msg):
    print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")

# Connect MQTT client
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.loop_start()

@app.post("/publish/hvac/temperature/set")
async def set_temperature(temp_data: TemperatureData):
    try:
        mqtt_client.publish("hvac/temperature/set", str(temp_data.value))
        return {"message": f"Temperature set to: {temp_data.value}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    print('Shutting down...')
    mqtt_client.loop_stop()
    mqtt_client.disconnect()