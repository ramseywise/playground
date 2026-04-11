"""AWS Lambda handler — adapts the FastAPI app via Mangum.

Entry point for container-image Lambda deployments.  The same FastAPI app
used by Fargate; Mangum wraps the ASGI interface.  ``lifespan="auto"``
invokes the existing FastAPI lifespan which calls ``init_graph()``.
"""

from __future__ import annotations

from mangum import Mangum

from interfaces.api.app import app

handler = Mangum(app, lifespan="auto")
