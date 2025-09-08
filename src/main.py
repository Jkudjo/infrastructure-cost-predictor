#!/usr/bin/env python3
"""
Infrastructure Cost Predictor & Optimizer - Main Application
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.data_collection.aws_collector import AWSCostCollector
from src.ml_models.cost_predictor import CostPredictor
from src.alerts.budget_manager import BudgetManager
from src.roi_calculator.roi_analyzer import ROIAnalyzer
from src.dashboard.api import router as dashboard_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
cost_predictor = None
budget_manager = None
roi_analyzer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global cost_predictor, budget_manager, roi_analyzer
    
    logger.info("Starting Infrastructure Cost Predictor...")
    
    # Initialize components
    try:
        cost_predictor = CostPredictor()
        budget_manager = BudgetManager()
        roi_analyzer = ROIAnalyzer()
        
        # Load existing models if available
        await cost_predictor.load_models()
        
        logger.info("All components initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    finally:
        logger.info("Shutting down Infrastructure Cost Predictor...")


# Create FastAPI app
app = FastAPI(
    title="Infrastructure Cost Predictor",
    description="AI-powered cloud cost prediction and optimization platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dashboard_router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information"""
    return """
    <html>
        <head>
            <title>Infrastructure Cost Predictor</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .header { text-align: center; color: #2c3e50; }
                .feature { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
                .api-link { color: #3498db; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="header">🚀 Infrastructure Cost Predictor & Optimizer</h1>
                <p>AI-powered platform for predicting, monitoring, and optimizing cloud infrastructure costs.</p>
                
                <div class="feature">
                    <h3>📊 Cost Forecasting</h3>
                    <p>ML-based prediction of compute, storage, and network costs with seasonal pattern detection.</p>
                </div>
                
                <div class="feature">
                    <h3>💰 Budget Management</h3>
                    <p>Smart alerts and automated recommendations for cost optimization.</p>
                </div>
                
                <div class="feature">
                    <h3>📈 ROI Analysis</h3>
                    <p>Calculate return on investment for infrastructure changes and optimizations.</p>
                </div>
                
                <p>
                    <a href="/docs" class="api-link">📚 API Documentation</a> |
                    <a href="/dashboard" class="api-link">📊 Dashboard</a> |
                    <a href="https://github.com/jkudjo/infrastructure-cost-predictor" class="api-link">🐙 GitHub</a>
                </p>
            </div>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "cost_predictor": cost_predictor is not None,
            "budget_manager": budget_manager is not None,
            "roi_analyzer": roi_analyzer is not None
        }
    }


@app.get("/api/v1/predict")
async def predict_costs(days: int = 30):
    """Predict future costs"""
    if not cost_predictor:
        raise HTTPException(status_code=503, detail="Cost predictor not initialized")
    
    try:
        prediction = await cost_predictor.predict(days=days)
        return {
            "prediction": prediction,
            "confidence": cost_predictor.get_confidence_score(),
            "model_info": cost_predictor.get_model_info()
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/budget/status")
async def get_budget_status():
    """Get current budget status"""
    if not budget_manager:
        raise HTTPException(status_code=503, detail="Budget manager not initialized")
    
    try:
        status = await budget_manager.get_budget_status()
        return status
    except Exception as e:
        logger.error(f"Budget status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/roi/calculate")
async def calculate_roi(request: Dict[str, Any]):
    """Calculate ROI for infrastructure changes"""
    if not roi_analyzer:
        raise HTTPException(status_code=503, detail="ROI analyzer not initialized")
    
    try:
        roi = await roi_analyzer.calculate_roi(
            current_cost=request.get("current_cost"),
            new_cost=request.get("new_cost"),
            migration_cost=request.get("migration_cost", 0),
            timeframe_months=request.get("timeframe_months", 12)
        )
        return roi
    except Exception as e:
        logger.error(f"ROI calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
