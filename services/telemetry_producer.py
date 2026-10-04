from kafka import KafkaProducer
import json
from datetime import datetime, timezone


producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)


services = [
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "productcatalogservice",
    "recommendationservice"
]


for i, service in enumerate(services):

    telemetry_event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "application": "Online Boutique",
        "service": service,

        "metrics": {
            "cpu": 82.5 if service == "checkoutservice" else 20.0 + i * 5,
            "memory": 68.3 if service == "checkoutservice" else 30.0 + i * 4,
            "diskio": 45.2 if service == "checkoutservice" else 20.0 + i * 3,
            "socket": 12.0 if service == "checkoutservice" else 5.0 + i,
            "workload": 75.4 if service == "checkoutservice" else 25.0 + i * 5,
            "error": 3.0 if service == "checkoutservice" else 0.5,
            "latency-50": 120.5 if service == "checkoutservice" else 40.0 + i * 5,
            "latency-90": 250.8 if service == "checkoutservice" else 80.0 + i * 10
        },

        "logs": {
            "log_count": 15 if service == "checkoutservice" else 5,
            "error_count": 3 if service == "checkoutservice" else 0,
            "timeout_count": 1 if service == "checkoutservice" else 0,
            "connection_count": 0
        },

        "traces": {
            "trace_count": 20 if service == "checkoutservice" else 8,
            "duration": 180.5 if service == "checkoutservice" else 50.0,
            "failure_count": 2 if service == "checkoutservice" else 0
        }
    }

    producer.send("telemetry", value=telemetry_event)

    print(f"Sent telemetry for {service}")


producer.flush()
producer.close()

print("\nAll 5 telemetry events sent successfully!")