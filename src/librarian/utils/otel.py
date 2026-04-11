"""OpenTelemetry bootstrap — call setup_otel() once at process startup.

When OTEL_ENABLED=false (default) this is a no-op, so the otel extra does not
need to be installed in the base librarian environment.

Install the extra to enable:
    uv sync --extra otel
"""

from __future__ import annotations

from agents.librarian.utils.logging import get_logger

log = get_logger(__name__)

_configured = False


def setup_otel() -> None:
    """Configure the OTel SDK + LangChain + Anthropic instrumentation.

    Idempotent — safe to call multiple times.
    Soft-fails with a structured warning if the otel extra is not installed.
    Controlled by LibrarySettings.otel_enabled (env: OTEL_ENABLED).
    """
    global _configured
    if _configured:
        return

    from agents.librarian.utils.config import settings

    if not settings.otel_enabled:
        return

    if settings.otel_exporter == "phoenix":
        # Phoenix manages its own TracerProvider internally via register()
        try:
            from phoenix.otel import register

            register(
                project_name=settings.otel_service_name,
                endpoint=settings.otel_endpoint,
            )
        except ImportError:
            log.warning(
                "otel.phoenix_missing",
                msg="arize-phoenix-otel not installed; run: uv sync --extra otel",
            )
            return
    else:
        # Default: OTLP gRPC — compatible with Jaeger, Phoenix, any OTEL collector
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            log.warning(
                "otel.otlp_missing",
                msg="opentelemetry-exporter-otlp-proto-grpc not installed; run: uv sync --extra otel",
            )
            return

        resource = Resource(attributes={SERVICE_NAME: settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

    # Instrument LangChain (covers LangGraph node execution)
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument()
    except ImportError:
        log.warning("otel.langchain_instrumentor_missing")

    # Instrument Anthropic SDK
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor

        AnthropicInstrumentor().instrument()
    except ImportError:
        log.warning("otel.anthropic_instrumentor_missing")

    _configured = True
    log.info(
        "otel.configured",
        exporter=settings.otel_exporter,
        endpoint=settings.otel_endpoint,
        service=settings.otel_service_name,
    )
