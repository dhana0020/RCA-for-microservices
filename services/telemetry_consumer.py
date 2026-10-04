from kafka import KafkaConsumer
import requests
import json


consumer = KafkaConsumer(
    "telemetry",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="rca-multi-service-consumer",
    value_deserializer=lambda value: json.loads(value.decode("utf-8"))
)

API_URL = "http://localhost:8000/predict"

EXPECTED_SERVICES = {
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "productcatalogservice",
    "recommendationservice"
}

telemetry_buffer = {}


print("Kafka → Multi-Service RCA consumer started...")
print("Waiting for telemetry events...\n")


for message in consumer:

    telemetry = message.value

    service = telemetry["service"]
    application = telemetry["application"]

    telemetry_buffer[service] = telemetry

    print(f"Received telemetry: {service}")

    # Wait until all 5 services have arrived
    if not EXPECTED_SERVICES.issubset(telemetry_buffer.keys()):
        continue

    print("\n" + "=" * 60)
    print("All 5 services received")
    print("=" * 60)

    services_data = []

    for service_name in EXPECTED_SERVICES:

        telemetry = telemetry_buffer[service_name]

        metrics = telemetry["metrics"]
        logs = telemetry["logs"]
        traces = telemetry["traces"]

        service_data = {
            "service": service_name,

            "cpu_change": metrics["cpu"],
            "cpu_ratio": metrics["cpu"] / 100,

            "mem_change": metrics["memory"],
            "mem_ratio": metrics["memory"] / 100,

            "diskio_change": metrics["diskio"],
            "diskio_ratio": metrics["diskio"] / 100,

            "socket_change": metrics["socket"],
            "socket_ratio": metrics["socket"] / 100,

            "workload_change": metrics["workload"],
            "workload_ratio": metrics["workload"] / 100,

            "error_change": metrics["error"],
            "error_ratio": metrics["error"] / 100,

            "latency-50_change": metrics["latency-50"],
            "latency-50_ratio": metrics["latency-50"] / 100,

            "latency-90_change": metrics["latency-90"],
            "latency-90_ratio": metrics["latency-90"] / 100,

            "log_count_change": logs["log_count"],
            "log_count_ratio": logs["log_count"] / 100,

            "log_error_count": logs["error_count"],
            "log_timeout_count": logs["timeout_count"],
            "log_connection_count": logs["connection_count"],

            "trace_count_change": traces["trace_count"],
            "trace_count_ratio": traces["trace_count"] / 100,

            "trace_duration_change": traces["duration"],
            "trace_duration_ratio": traces["duration"] / 100,

            "trace_failure_count": traces["failure_count"]
        }

        services_data.append(service_data)

    payload = {
        "application": application,
        "services": services_data
    }

    print("\nSending all services to RCA API...")

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        print("\nRCA API response:")
        print("HTTP Status:", response.status_code)

        print(json.dumps(
            response.json(),
            indent=2
        ))

    except Exception as e:

        print("\nError calling RCA API:")
        print(e)

    # Clear buffer for the next incident
    telemetry_buffer.clear()

    print("\nWaiting for next incident...\n")