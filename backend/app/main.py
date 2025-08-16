from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import asyncio
import time
import math
from typing import List, Dict, Any
import threading
import sys
import os

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from data_streaming import StreamingService
import sys
sys.path.append('models')
from autoencoder import AnomalyDetector
from simple_rul import SimpleRULPredictor

app = FastAPI(
    title="Insight - Predictive Maintenance API",
    description="Real-time predictive maintenance system for industrial equipment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
streaming_service = None
anomaly_detector = None
rul_predictor = None
websocket_connections: List[WebSocket] = []

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed, remove it
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global streaming_service, anomaly_detector, rul_predictor
    
    print("Starting Insight Predictive Maintenance API...")
    
    # Initialize streaming service
    streaming_service = StreamingService()
    
    # Load ML models
    try:
        anomaly_detector = AnomalyDetector()
        anomaly_detector.load_model('models/autoencoder_anomaly.json')
        print("Anomaly detection model loaded")
    except Exception as e:
        print(f"Could not load anomaly model: {e}")
        anomaly_detector = None
    
    try:
        rul_predictor = SimpleRULPredictor()
        rul_predictor.load_model('models/simple_rul.json')
        print("RUL prediction model loaded")
    except Exception as e:
        print(f"Could not load RUL model: {e}")
        rul_predictor = None
    
    print("API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global streaming_service
    if streaming_service:
        streaming_service.stop_service()
    print("API shutdown completed")

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Insight Predictive Maintenance API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "streaming": streaming_service is not None,
            "anomaly_detection": anomaly_detector is not None,
            "rul_prediction": rul_predictor is not None
        }
    }

# Engine management endpoints
@app.get("/engines")
async def get_engines():
    """Get list of available engines"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    engine_status = streaming_service.data_streamer.get_engine_status()
    return {"engines": engine_status}

@app.post("/streaming/start")
async def start_streaming(engines: List[int] = None, speed: float = 1.0):
    """Start data streaming for specified engines"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    try:
        streaming_service.start_service(engines=engines, speed=speed)
        return {"message": "Streaming started", "engines": engines, "speed": speed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start streaming: {str(e)}")

@app.post("/streaming/stop")
async def stop_streaming():
    """Stop data streaming"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    streaming_service.stop_service()
    return {"message": "Streaming stopped"}

# Data endpoints
@app.get("/data/sensor")
async def get_sensor_data(limit: int = 50):
    """Get latest sensor data"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    data = streaming_service.get_latest_sensor_data(limit)
    return {"data": data, "count": len(data)}

@app.get("/data/alerts")
async def get_alerts(limit: int = 10):
    """Get latest anomaly alerts"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    alerts = streaming_service.get_anomaly_alerts(limit)
    return {"alerts": alerts, "count": len(alerts)}

@app.get("/data/predictions")
async def get_predictions(limit: int = 20):
    """Get latest RUL predictions"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    predictions = streaming_service.get_rul_predictions(limit)
    return {"predictions": predictions, "count": len(predictions)}

# ML model endpoints
@app.post("/models/anomaly/predict")
async def predict_anomaly(data: Dict[str, Any]):
    """Predict anomaly for given sensor data"""
    if not anomaly_detector:
        raise HTTPException(status_code=503, detail="Anomaly detection model not available")
    
    try:
        # Extract features (this would need proper feature engineering)
        features = [data.get(f'sensor_{i}', 0) for i in range(1, 22)]
        
        is_anomaly, error = anomaly_detector.autoencoder.detect_anomaly(features, anomaly_detector.threshold)
        
        return {
            "is_anomaly": bool(is_anomaly),
            "reconstruction_error": float(error),
            "threshold": float(anomaly_detector.threshold)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/models/rul/predict")
async def predict_rul(data: Dict[str, Any]):
    """Predict RUL for given sensor data"""
    if not rul_predictor:
        raise HTTPException(status_code=503, detail="RUL prediction model not available")
    
    try:
        # Extract and engineer features
        cycle = data.get('cycle', 1)
        features = [cycle]
        
        # Add settings
        features.extend([data.get(f'setting_{i}', 0) for i in range(1, 4)])
        
        # Add sensors
        features.extend([data.get(f'sensor_{i}', 0) for i in range(1, 22)])
        
        # Add engineered features
        features.extend([
            cycle ** 0.5,
            math.log(cycle + 1),
            1 / (cycle + 1),
        ])
        
        rul = rul_predictor.predict_single(features)
        
        return {
            "predicted_rul": float(rul),
            "input_cycle": cycle
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Send real-time updates
            if streaming_service:
                # Get latest data
                sensor_data = streaming_service.get_latest_sensor_data(5)
                alerts = streaming_service.get_anomaly_alerts(2)
                predictions = streaming_service.get_rul_predictions(3)
                
                update = {
                    "timestamp": time.time(),
                    "sensor_data": sensor_data,
                    "alerts": alerts,
                    "predictions": predictions
                }
                
                await websocket.send_text(json.dumps(update))
            
            # Wait before next update
            await asyncio.sleep(2)  # Send updates every 2 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Dashboard summary endpoint
@app.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get dashboard summary data"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    # Get engine status
    engine_status = streaming_service.data_streamer.get_engine_status()
    
    # Get recent data counts
    sensor_data = streaming_service.get_latest_sensor_data(100)
    alerts = streaming_service.get_anomaly_alerts(20)
    predictions = streaming_service.get_rul_predictions(50)
    
    # Calculate summary statistics
    total_engines = len(engine_status)
    active_engines = len([e for e in sensor_data if time.time() - e.get('timestamp', 0) < 60])
    critical_alerts = len([a for a in alerts if 'anomaly' in a.get('alert_type', '')])
    
    avg_rul = 0
    if predictions:
        avg_rul = sum(p.get('predicted_rul', 0) for p in predictions) / len(predictions)
    
    return {
        "summary": {
            "total_engines": total_engines,
            "active_engines": active_engines,
            "critical_alerts": critical_alerts,
            "avg_predicted_rul": avg_rul,
            "last_update": time.time()
        },
        "recent_data": {
            "sensor_readings": len(sensor_data),
            "alerts": len(alerts),
            "predictions": len(predictions)
        }
    }

# Historical data endpoint
@app.get("/engines/{engine_id}/history")
async def get_engine_history(engine_id: int, limit: int = 100):
    """Get historical data for a specific engine"""
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    # Get engine data from streamer
    engine_data = streaming_service.data_streamer.engines_data.get(engine_id, [])
    
    if not engine_data:
        raise HTTPException(status_code=404, detail="Engine not found")
    
    # Return limited data
    limited_data = engine_data[-limit:] if len(engine_data) > limit else engine_data
    
    return {
        "engine_id": engine_id,
        "total_records": len(engine_data),
        "returned_records": len(limited_data),
        "data": limited_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)