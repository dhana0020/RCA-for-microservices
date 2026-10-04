from fastapi import FastAPI

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


resource = Resource.create({
    "service.name": "ai-rca-telemetry-test"
})

provider = TracerProvider(resource=resource)

otlp_exporter = OTLPSpanExporter(
    endpoint="localhost:4317",
    insecure=True
)

provider.add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

app = FastAPI()


@app.get("/")
def home():
    with tracer.start_as_current_span("rca-test-operation"):
        return {
            "message": "OpenTelemetry → Collector is working"
        }