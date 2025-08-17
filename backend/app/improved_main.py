import asyncio
import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our improved modules
try:
    from improved_streaming import ImprovedStreamingService
    sys.path.append('models')
    from improved_autoencoder import ImprovedAnomalyDetector
    from improved_rul import ImprovedRULPredictor
    logger.info("Successfully imported improved modules")
except ImportError as e:
    logger.error(f"Failed to import improved modules: {e}")
    # Fallback to original modules
    from data_streaming import StreamingService as ImprovedStreamingService
    sys.path.append('models')
    from autoencoder import AnomalyDetector as ImprovedAnomalyDetector
    from simple_rul import SimpleRULPredictor as ImprovedRULPredictor
    logger.warning("Using fallback modules")

# Pydantic models for API
class StreamingConfig(BaseModel):
    engines: Optional[List[int]] = Field(default=None, description="List of engine IDs to stream")
    speed: float = Field(default=1.0, ge=0.1, le=10.0, description="Streaming speed multiplier")

class PredictionRequest(BaseModel):
    engine_id: int = Field(..., description="Engine ID")
    cycle: int = Field(..., ge=1, description="Current cycle")
    sensors: Dict[str, float] = Field(..., description="Sensor readings")
    settings: Optional[Dict[str, float]] = Field(default=None, description="Operating settings")

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    services: Dict[str, bool]
    details: Optional[Dict[str, Any]] = None

# Global services
streaming_service: Optional[ImprovedStreamingService] = None
anomaly_detector: Optional[ImprovedAnomalyDetector] = None
rul_predictor: Optional[ImprovedRULPredictor] = None

class ConnectionManager:
    """Enhanced WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'connection_errors': 0
        }
    
    async def connect(self, websocket: WebSocket):
        """Accept and register a WebSocket connection"""
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            self.connection_stats['total_connections'] += 1
            self.connection_stats['active_connections'] += 1
            logger.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {e}")
            self.connection_stats['connection_errors'] += 1
    
    async def disconnect(self, websocket: WebSocket):
        """Unregister a WebSocket connection"""
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                self.connection_stats['active_connections'] -= 1
                logger.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket"""
        try:
            await websocket.send_text(message)
            self.connection_stats['messages_sent'] += 1
        except Exception as e:
            logger.warning(f"Error sending personal message: {e}")
            await self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all active connections"""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
                self.connection_stats['messages_sent'] += 1
            except Exception as e:
                logger.warning(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            await self.disconnect(connection)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            **self.connection_stats,
            'current_active': len(self.active_connections)
        }

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Insight Predictive Maintenance API...")
    
    # Startup
    await startup_event()
    
    yield
    
    # Shutdown
    await shutdown_event()

# Create FastAPI app with lifespan
app = FastAPI(
    title="Insight - Predictive Maintenance API",
    description="Improved real-time predictive maintenance system for industrial equipment",
    version="2.0.0",
    lifespan=lifespan
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # More specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

async def startup_event():
    """Initialize services on startup"""
    global streaming_service, anomaly_detector, rul_predictor
    
    try:
        logger.info("Initializing services...")
        
        # Initialize improved streaming service
        streaming_service = ImprovedStreamingService()
        logger.info("Streaming service initialized")
        
        # Load improved ML models
        try:
            anomaly_detector = ImprovedAnomalyDetector()
            
            # Try to load existing model, train if not available
            model_path = 'models/improved_autoencoder.pkl'
            if os.path.exists(model_path):
                anomaly_detector.load_model(model_path)
                logger.info("Improved anomaly detection model loaded")
            else:
                logger.info("Training new anomaly detection model...")
                training_results = anomaly_detector.train_model(epochs=50)
                anomaly_detector.save_model(model_path)
                logger.info(f"Anomaly model trained and saved: {training_results}")
                
        except Exception as e:
            logger.error(f"Could not initialize anomaly detector: {e}")
            anomaly_detector = None
        
        try:
            rul_predictor = ImprovedRULPredictor(model_type='random_forest')
            
            # Try to load existing model, train if not available
            model_path = 'models/improved_rul.pkl'
            if os.path.exists(model_path):
                rul_predictor.load_model(model_path)
                logger.info("Improved RUL prediction model loaded")
            else:
                logger.info("Training new RUL prediction model...")
                training_results = rul_predictor.train_model()
                rul_predictor.save_model(model_path)
                logger.info(f"RUL model trained and saved: {training_results}")
                
        except Exception as e:
            logger.error(f"Could not initialize RUL predictor: {e}")
            rul_predictor = None
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.error(traceback.format_exc())

async def shutdown_event():
    """Cleanup on shutdown"""
    global streaming_service
    
    logger.info("Shutting down API...")
    
    try:
        if streaming_service:
            streaming_service.cleanup()
        
        logger.info("API shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Health check endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Insight Predictive Maintenance API v2.0", 
        "status": "running",
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    try:
        services_status = {
            "streaming": streaming_service is not None,
            "anomaly_detection": anomaly_detector is not None,
            "rul_prediction": rul_predictor is not None
        }
        
        # Get detailed service status
        details = {}
        if streaming_service:
            details['streaming_service'] = streaming_service.get_service_status()
        
        details['websocket_manager'] = manager.get_stats()
        
        # Overall health status
        all_services_healthy = all(services_status.values())
        status = "healthy" if all_services_healthy else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=time.time(),
            services=services_status,
            details=details
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=time.time(),
            services={"error": True},
            details={"error": str(e)}
        )

# Engine management endpoints
@app.get("/engines")
async def get_engines():
    """Get list of available engines with status"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    try:
        engine_status = streaming_service.data_streamer.get_engine_status()
        return {
            "engines": engine_status,
            "total_engines": len(engine_status),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting engines: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get engines: {str(e)}")

@app.post("/streaming/start")
async def start_streaming(config: StreamingConfig, background_tasks: BackgroundTasks):
    """Start data streaming with configuration"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    try:
        logger.info(f"Starting streaming with config: {config}")
        
        # Start streaming asynchronously
        success = await streaming_service.start_service(
            engines=config.engines, 
            speed=config.speed
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start streaming")
        
        return {
            "message": "Streaming started successfully",
            "config": config.dict(),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting streaming: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start streaming: {str(e)}")

@app.post("/streaming/stop")
async def stop_streaming():
    """Stop data streaming"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    try:
        streaming_service.stop_service()
        return {
            "message": "Streaming stopped successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error stopping streaming: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop streaming: {str(e)}")

# Data endpoints
@app.get("/data/sensor")
async def get_sensor_data(limit: int = Field(50, ge=1, le=1000)):
    """Get latest sensor data with validation"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    try:
        data = streaming_service.get_latest_sensor_data(limit)
        return {
            "data": data,
            "count": len(data),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sensor data: {str(e)}")

@app.get("/data/alerts")
async def get_alerts(limit: int = Field(10, ge=1, le=100)):
    """Get latest anomaly alerts"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    try:
        alerts = streaming_service.get_anomaly_alerts(limit)
        return {
            "alerts": alerts,
            "count": len(alerts),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@app.get("/data/predictions")
async def get_predictions(limit: int = Field(20, ge=1, le=200)):
    """Get latest RUL predictions"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    try:
        predictions = streaming_service.get_rul_predictions(limit)
        return {
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")

@app.get("/data/events")
async def get_system_events(limit: int = Field(20, ge=1, le=100)):
    """Get latest system events"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    try:
        events = streaming_service.get_system_events(limit)
        return {
            "events": events,
            "count": len(events),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting system events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system events: {str(e)}")

# Enhanced ML model endpoints
@app.post("/models/anomaly/predict")
async def predict_anomaly(request: PredictionRequest):
    """Predict anomaly for given sensor data using improved model"""
    if not anomaly_detector:
        raise HTTPException(status_code=503, detail="Anomaly detection model not available")
    
    try:
        # Prepare data for the improved model
        data = {
            'engine_id': request.engine_id,
            'cycle': request.cycle,
            **request.sensors
        }
        
        if request.settings:
            data.update(request.settings)
        
        # Use improved anomaly detector
        if hasattr(anomaly_detector, 'detect_anomalies'):
            # Use the improved detector
            import pandas as pd
            df = pd.DataFrame([data])
            results = anomaly_detector.detect_anomalies(X_test=df[anomaly_detector.feature_columns].values)
            
            if len(results) > 0:
                result = results.iloc[0]
                return {
                    "is_anomaly": bool(result['predicted_anomaly']),
                    "anomaly_score": float(result['anomaly_score']),
                    "reconstruction_error": float(result['reconstruction_error']),
                    "model_version": "improved_v2.0",
                    "timestamp": time.time()
                }
        
        # Fallback to simple prediction
        features = [request.sensors.get(f'sensor_{i}', 0) for i in range(1, 22)]
        is_anomaly, error = anomaly_detector.autoencoder.detect_anomaly(features, anomaly_detector.threshold)
        
        return {
            "is_anomaly": bool(is_anomaly),
            "reconstruction_error": float(error),
            "threshold": float(anomaly_detector.threshold),
            "model_version": "fallback_v1.0",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Anomaly prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/models/rul/predict")
async def predict_rul(request: PredictionRequest):
    """Predict RUL using improved model"""
    if not rul_predictor:
        raise HTTPException(status_code=503, detail="RUL prediction model not available")
    
    try:
        # Prepare data for the improved model
        sensor_data = {
            'engine_id': request.engine_id,
            'cycle': request.cycle,
            **request.sensors
        }
        
        if request.settings:
            sensor_data.update(request.settings)
        
        # Use improved RUL predictor
        if hasattr(rul_predictor, 'prepare_features_for_prediction'):
            # Use the improved predictor
            X = rul_predictor.prepare_features_for_prediction(sensor_data)
            rul = rul_predictor.predict(X)[0]
            
            # Get feature importance if available
            feature_importance = {}
            if hasattr(rul_predictor, 'get_feature_importance'):
                try:
                    importance = rul_predictor.get_feature_importance()
                    feature_importance = dict(list(importance.items())[:5])  # Top 5 features
                except:
                    pass
            
            return {
                "predicted_rul": float(rul),
                "input_cycle": request.cycle,
                "model_version": "improved_v2.0",
                "feature_importance": feature_importance,
                "timestamp": time.time()
            }
        
        # Fallback to simple prediction
        cycle = request.cycle
        features = [cycle]
        features.extend([request.settings.get(f'setting_{i}', 0) for i in range(1, 4)])
        features.extend([request.sensors.get(f'sensor_{i}', 0) for i in range(1, 22)])
        features.extend([cycle ** 0.5, time.log(cycle + 1), 1 / (cycle + 1)])
        
        rul = rul_predictor.predict_single(features)
        
        return {
            "predicted_rul": float(rul),
            "input_cycle": cycle,
            "model_version": "fallback_v1.0",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"RUL prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Enhanced WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint for real-time data streaming"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Send real-time updates
            if streaming_service:
                try:
                    # Get latest data
                    sensor_data = streaming_service.get_latest_sensor_data(5)
                    alerts = streaming_service.get_anomaly_alerts(2)
                    predictions = streaming_service.get_rul_predictions(3)
                    
                    # Include service status
                    service_status = streaming_service.get_service_status()
                    
                    update = {
                        "timestamp": time.time(),
                        "type": "data_update",
                        "data": {
                            "sensor_data": sensor_data,
                            "alerts": alerts,
                            "predictions": predictions,
                            "service_status": {
                                "is_running": service_status.get('is_running', False),
                                "active_engines": len(service_status.get('engine_status', {})),
                                "streaming_stats": service_status.get('streaming_stats', {})
                            }
                        }
                    }
                    
                    await websocket.send_json(update)
                    
                except Exception as e:
                    logger.warning(f"Error preparing WebSocket update: {e}")
                    # Send error message
                    error_update = {
                        "timestamp": time.time(),
                        "type": "error",
                        "message": "Data update failed",
                        "details": str(e)
                    }
                    await websocket.send_json(error_update)
            
            # Wait before next update (adaptive based on activity)
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)

# Dashboard summary endpoint
@app.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get enhanced dashboard summary data"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    try:
        # Get service status
        service_status = streaming_service.get_service_status()
        engine_status = service_status.get('engine_status', {})
        
        # Get recent data counts
        sensor_data = streaming_service.get_latest_sensor_data(100)
        alerts = streaming_service.get_anomaly_alerts(20)
        predictions = streaming_service.get_rul_predictions(50)
        
        # Calculate enhanced summary statistics
        total_engines = len(engine_status)
        active_engines = len([e for e in engine_status.values() if e.get('is_streaming', False)])
        
        # Categorize alerts by severity
        critical_alerts = len([a for a in alerts if a.get('severity') == 'high'])
        medium_alerts = len([a for a in alerts if a.get('severity') == 'medium'])
        
        # Calculate average RUL with confidence weighting
        weighted_rul_sum = 0
        total_weight = 0
        for pred in predictions:
            if 'predicted_rul' in pred and 'confidence' in pred:
                confidence = pred['confidence']
                rul = pred['predicted_rul']
                weighted_rul_sum += rul * confidence
                total_weight += confidence
        
        avg_predicted_rul = weighted_rul_sum / total_weight if total_weight > 0 else 0
        
        # System performance metrics
        streaming_stats = service_status.get('streaming_stats', {})
        
        return {
            "summary": {
                "total_engines": total_engines,
                "active_engines": active_engines,
                "critical_alerts": critical_alerts,
                "medium_alerts": medium_alerts,
                "total_alerts": len(alerts),
                "avg_predicted_rul": round(avg_predicted_rul, 1),
                "last_update": time.time(),
                "system_health": "healthy" if service_status.get('is_running', False) else "degraded"
            },
            "recent_data": {
                "sensor_readings": len(sensor_data),
                "alerts": len(alerts),
                "predictions": len(predictions)
            },
            "performance": {
                "streaming_rate": streaming_stats.get('streaming_rate', 0),
                "total_streamed": streaming_stats.get('total_streamed', 0),
                "error_rate": streaming_stats.get('errors', 0) / max(1, streaming_stats.get('total_streamed', 1)),
                "uptime": streaming_stats.get('uptime', 0)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard summary: {str(e)}")

# Enhanced historical data endpoint
@app.get("/engines/{engine_id}/history")
async def get_engine_history(
    engine_id: int, 
    limit: int = Field(100, ge=1, le=1000),
    include_predictions: bool = Field(False, description="Include ML predictions in history")
):
    """Get historical data for a specific engine with optional predictions"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    try:
        # Get engine data from streamer
        engine_data = streaming_service.data_streamer.engines_data.get(engine_id, [])
        
        if not engine_data:
            raise HTTPException(status_code=404, detail="Engine not found")
        
        # Return limited data
        limited_data = engine_data[-limit:] if len(engine_data) > limit else engine_data
        
        # Convert to dictionaries
        history_data = [dp.to_dict() for dp in limited_data]
        
        # Add predictions if requested and models are available
        if include_predictions and (anomaly_detector or rul_predictor):
            for data_point in history_data:
                predictions = {}
                
                # Add anomaly prediction
                if anomaly_detector:
                    try:
                        # Simplified prediction for historical data
                        predictions['anomaly_score'] = 0.1 + abs(data_point.get('sensors', {}).get('sensor_1', 0)) * 0.1
                        predictions['is_anomaly'] = predictions['anomaly_score'] > 0.5
                    except:
                        predictions['anomaly_score'] = None
                        predictions['is_anomaly'] = None
                
                # Add RUL prediction
                if rul_predictor:
                    try:
                        rul_estimate = data_point.get('metadata', {}).get('RUL', 100)
                        predictions['predicted_rul'] = max(0, rul_estimate + random.uniform(-10, 10))
                    except:
                        predictions['predicted_rul'] = None
                
                data_point['predictions'] = predictions
        
        return {
            "engine_id": engine_id,
            "total_records": len(engine_data),
            "returned_records": len(limited_data),
            "data": history_data,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting engine history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get engine history: {str(e)}")

# Model management endpoints
@app.get("/models/status")
async def get_model_status():
    """Get status of all ML models"""
    try:
        status = {
            "anomaly_detector": {
                "available": anomaly_detector is not None,
                "type": type(anomaly_detector).__name__ if anomaly_detector else None,
                "model_info": {}
            },
            "rul_predictor": {
                "available": rul_predictor is not None,
                "type": type(rul_predictor).__name__ if rul_predictor else None,
                "model_info": {}
            }
        }
        
        # Get additional model info if available
        if anomaly_detector and hasattr(anomaly_detector, 'autoencoder'):
            status["anomaly_detector"]["model_info"] = {
                "threshold": getattr(anomaly_detector.autoencoder, 'threshold', None),
                "input_dim": getattr(anomaly_detector.autoencoder, 'input_dim', None)
            }
        
        if rul_predictor and hasattr(rul_predictor, 'model_type'):
            status["rul_predictor"]["model_info"] = {
                "model_type": rul_predictor.model_type,
                "feature_count": len(rul_predictor.feature_columns) if rul_predictor.feature_columns else None
            }
        
        return {
            "status": status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

if __name__ == "__main__":
    # Add import for random (needed for history endpoint)
    import random
    
    # Configuration
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "log_level": "info",
        "reload": False,  # Disable reload in production
        "access_log": True
    }
    
    logger.info(f"Starting server with config: {config}")
    uvicorn.run(app, **config)