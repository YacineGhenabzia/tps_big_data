from kafka import KafkaProducer
import json, time, random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

while True:
    machine_data = {
        "machine_id": "MCH_01",
        "status": random.choice(["OK", "PANNE"]),
        "temperature": round(random.uniform(30, 100), 2),
        "timestamp": time.time()
    }
    producer.send('machine-tracking', machine_data)
    print("Envoyé:", machine_data)
    time.sleep(2)
