from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import (
    HTTPException as StarletteHTTPException,
)

from app.api.routes.health import router as health_router
from app.api.routes.workflows import router as workflow_router
from app.core.config import get_settings
from app.core.errors import (
    http_error_handler,
    validation_error_handler,
    unexpected_error_handler,
)
from app.core.logging import configure_logging

from time import perf_counter
from uuid import uuid4


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        debug=settings.debug,
    )
    logger = configure_logging()

    @application.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id

        start = perf_counter()
        status_code = 500

        try:
            response = await call_next(request)

            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id

            return response
        except Exception as exc:
            logger.error(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                },
            )
            raise
        finally:
            duration_ms = round(
                (perf_counter() - start) * 1000,
                2,
            )
            logger.info(
                "request_finished",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                },
            )



    application.include_router(health_router)
    application.include_router(workflow_router)
    
    application.add_exception_handler(
        StarletteHTTPException,
        http_error_handler,
    )

    application.add_exception_handler(
        RequestValidationError,
        validation_error_handler,
    )

    application.add_exception_handler(
        Exception,
        unexpected_error_handler,
    )

    return application


app = create_app()


    