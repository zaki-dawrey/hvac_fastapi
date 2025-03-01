# import paho.mqtt.client as mqtt
# import time
# import random
# import json
# from datetime import datetime

# class TemperatureSensor:
#     def __init__(self, broker="localhost", port=1883):
#         self.client = mqtt.Client()
#         self.broker = broker
#         self.port = port
#         self.base_temperature = 23.0  # Starting temperature
#         self.current_temperature = self.base_temperature

#     def connect(self):
#         self.client.connect(self.broker, self.port)
#         self.client.loop_start()

#     def simulate_temperature(self):
#         # Add random fluctuation between -0.5 and +0.5
#         fluctuation = random.uniform(-0.5, 0.5)
#         self.current_temperature += fluctuation
        
#         # Keep temperature within realistic bounds (18-30°C)
#         self.current_temperature = max(18, min(30, self.current_temperature))
        
#         return round(self.current_temperature, 2)

#     def run(self):
#         try:
#             self.connect()
#             while True:
#                 temperature = self.simulate_temperature()
#                 payload = {
#                     "temperature": temperature,
#                     "timestamp": datetime.now().isoformat()
#                 }
#                 self.client.publish("hvac/temperature/current", json.dumps(payload))
#                 time.sleep(5)  # Send data every 5 seconds
#         except KeyboardInterrupt:
#             self.client.loop_stop()
#             self.client.disconnect()

# if __name__ == "__main__":
#     sensor = TemperatureSensor()
#     sensor.run()
