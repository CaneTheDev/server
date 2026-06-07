import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.api.endpoints import router as api_router
from app.core.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpportunityOS API",
    description="Intelligent Multi-Agent Opportunity Intelligence System Backend",
    version="1.0.0"
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom exception handler to log validation errors.
    """
    body = await request.body()
    logger.error(f"422 Validation Error: {exc.errors()}")
    logger.error(f"Request Body: {body.decode()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode()},
    )

# Enable CORS for frontend integration
allowed_origins = settings.ALLOWED_ORIGINS

# If in DEBUG mode and no origins are specified, allow common local development origins
if settings.DEBUG and not allowed_origins:
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ]
    logger.info(f"CORS: DEBUG mode active. Defaulting to local origins: {allowed_origins}")
else:
    logger.info(f"CORS: Allowed origins configured: {allowed_origins}")

if not allowed_origins and not settings.DEBUG:
    logger.warning("CORS: No ALLOWED_ORIGINS found in environment and DEBUG is False. Cross-origin requests will fail.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Completed request: {request.method} {request.url.path} - Status: {response.status_code}")
    return response

# Include API Router
app.include_router(api_router, prefix="/api")

@app.get("/")
def health_check():
    """
    Standard health check endpoint.
    """
    return {"status": "ok", "app": "OpportunityOS Backend"}
