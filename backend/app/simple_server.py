"""
Simple HTTP server for testing the API without uvicorn
"""
import http.server
import socketserver
import json
import urllib.parse
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))

from data_streaming import StreamingService
from autoencoder import AnomalyDetector  
from simple_rul import SimpleRULPredictor
import time

# Global services
streaming_service = StreamingService()
anomaly_detector = None
rul_predictor = None

# Try to load models
try:
    anomaly_detector = AnomalyDetector()
    anomaly_detector.load_model('models/autoencoder_anomaly.json')
    print("Anomaly detection model loaded")
except Exception as e:
    print(f"Could not load anomaly model: {e}")

try:
    rul_predictor = SimpleRULPredictor()
    rul_predictor.load_model('models/simple_rul.json')
    print("RUL prediction model loaded")
except Exception as e:
    print(f"Could not load RUL model: {e}")

class APIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        path = self.path
        
        if path == "/" or path == "/health":
            self.send_json_response({
                "status": "healthy",
                "timestamp": time.time(),
                "services": {
                    "streaming": streaming_service is not None,
                    "anomaly_detection": anomaly_detector is not None,
                    "rul_prediction": rul_predictor is not None
                }
            })
        
        elif path == "/engines":
            engine_status = streaming_service.data_streamer.get_engine_status()
            self.send_json_response({"engines": engine_status})
        
        elif path.startswith("/data/sensor"):
            data = streaming_service.get_latest_sensor_data(50)
            self.send_json_response({"data": data, "count": len(data)})
        
        elif path.startswith("/data/alerts"):
            alerts = streaming_service.get_anomaly_alerts(10)
            self.send_json_response({"alerts": alerts, "count": len(alerts)})
        
        elif path.startswith("/data/predictions"):
            predictions = streaming_service.get_rul_predictions(20)
            self.send_json_response({"predictions": predictions, "count": len(predictions)})
        
        elif path == "/dashboard/summary":
            engine_status = streaming_service.data_streamer.get_engine_status()
            sensor_data = streaming_service.get_latest_sensor_data(100)
            alerts = streaming_service.get_anomaly_alerts(20)
            predictions = streaming_service.get_rul_predictions(50)
            
            total_engines = len(engine_status)
            active_engines = len([e for e in sensor_data if time.time() - e.get('timestamp', 0) < 60])
            critical_alerts = len(alerts)
            avg_rul = sum(p.get('predicted_rul', 0) for p in predictions) / len(predictions) if predictions else 0
            
            self.send_json_response({
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
            })
        
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests"""
        path = self.path
        
        if path == "/streaming/start":
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data) if post_data else {}
                engines = data.get('engines', [1, 2, 3])
                speed = data.get('speed', 1.0)
                
                streaming_service.start_service(engines=engines, speed=speed)
                self.send_json_response({"message": "Streaming started", "engines": engines, "speed": speed})
            except Exception as e:
                self.send_error(500, f"Failed to start streaming: {str(e)}")
        
        elif path == "/streaming/stop":
            streaming_service.stop_service()
            self.send_json_response({"message": "Streaming stopped"})
        
        else:
            self.send_error(404, "Not Found")
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server(port=8000):
    """Run the simple API server"""
    print(f"Starting Insight API server on port {port}")
    print(f"Access the API at: http://localhost:{port}")
    
    with socketserver.TCPServer(("", port), APIHandler) as httpd:
        print(f"Server running at http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            streaming_service.stop_service()

if __name__ == "__main__":
    run_server()