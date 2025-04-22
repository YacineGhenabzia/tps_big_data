from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'machine-tracking',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    data = message.value
    print("Reçu:", data)
    if data['status'] == "PANNE":
        print(f"Alerte: Machine {data['machine_id']} en panne!")
