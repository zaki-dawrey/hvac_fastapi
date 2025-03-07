# import paho.mqtt.client as mqtt
# import time
# import random
# import json
# from datetime import datetime

# class EnvironmentSensor:
#     def __init__(self, broker="localhost", port=1883):
#         self.client = mqtt.Client()
#         self.broker = broker
#         self.port = port
#         self.base_temperature = 23.0
#         self.current_temperature = self.base_temperature
#         self.base_humidity = 50.0
#         self.current_humidity = self.base_humidity

#     def connect(self):
#         self.client.connect(self.broker, self.port)
#         self.client.loop_start()

#     def simulate_temperature(self):
#         fluctuation = random.uniform(-0.5, 0.5)
#         self.current_temperature += fluctuation
#         self.current_temperature = max(18, min(30, self.current_temperature))
#         return round(self.current_temperature, 2)

#     def simulate_humidity(self):
#         fluctuation = random.uniform(-1, 1)
#         self.current_humidity += fluctuation
#         self.current_humidity = max(30, min(70, self.current_humidity))
#         return round(self.current_humidity, 2)

#     def run(self):
#         try:
#             self.connect()
#             while True:
#                 temperature = self.simulate_temperature()
#                 humidity = self.simulate_humidity()
#                 payload = {
#                     "temperature": temperature,
#                     "humidity": humidity,
#                     "timestamp": datetime.now().isoformat()
#                 }
#                 self.client.publish("hvac/environment/current", json.dumps(payload))
#                 time.sleep(5)
#         except KeyboardInterrupt:
#             self.client.loop_stop()
#             self.client.disconnect()

# if __name__ == "__main__":
#     sensor = EnvironmentSensor()
#     sensor.run()
