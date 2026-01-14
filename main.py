"""Main entry point for RetailMind Forecasting Service"""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="RetailMind Forecasting Service",
    version="1.0.0",
    description="Prophet-based forecasting microservice for RetailMind POS"
)

# Include API routes
app.include_router(router)
