import csv
import json
import time
import random
import asyncio
import threading
from queue import Queue
from typing import List, Dict, Any
import os

class DataStreamer:
    """Simulates real-time data streaming from engines"""
    
    def __init__(self, data_file='data/processed/train_processed.csv'):
        self.data_file = data_file
        self.engines_data = {}
        self.streaming_queue = Queue()
        self.is_streaming = False
        self.stream_speed = 1.0  # Speed multiplier
        self.subscribers = []
        
        # Load and prepare data
        self.load_engine_data()
    
    def load_engine_data(self):
        """Load and organize data by engine"""
        print(f"Loading streaming data from {self.data_file}")
        
        with open(self.data_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            for row in reader:
                if len(row) < len(header):
                    continue
                
                engine_id = int(float(row[0]))
                cycle = int(float(row[1]))
                
                # Create data point
                data_point = {
                    'engine_id': engine_id,
                    'cycle': cycle,
                    'timestamp': time.time(),  # Will be updated during streaming
                }
                
                # Add all sensor and setting data
                for i, col_name in enumerate(header[2:], start=2):
                    if i < len(row):
                        data_point[col_name] = float(row[i])
                
                # Group by engine
                if engine_id not in self.engines_data:
                    self.engines_data[engine_id] = []
                
                self.engines_data[engine_id].append(data_point)
        
        # Sort each engine's data by cycle
        for engine_id in self.engines_data:
            self.engines_data[engine_id].sort(key=lambda x: x['cycle'])
        
        print(f"Loaded data for {len(self.engines_data)} engines")
    
    def subscribe(self, callback):
        """Subscribe to data stream updates"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback):
        """Unsubscribe from data stream updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def notify_subscribers(self, data):
        """Notify all subscribers of new data"""
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")
    
    def start_streaming(self, selected_engines=None, speed=1.0):
        """Start streaming data for selected engines"""
        self.stream_speed = speed
        self.is_streaming = True
        
        if selected_engines is None:
            # Select a few engines for demonstration
            selected_engines = list(self.engines_data.keys())[:5]
        
        print(f"Starting data stream for engines: {selected_engines}")
        
        # Start streaming in a separate thread
        streaming_thread = threading.Thread(
            target=self._stream_data,
            args=(selected_engines,),
            daemon=True
        )
        streaming_thread.start()
        
        return streaming_thread
    
    def stop_streaming(self):
        """Stop data streaming"""
        self.is_streaming = False
        print("Data streaming stopped")
    
    def _stream_data(self, engine_ids):
        """Internal method to stream data"""
        # Create engine state tracking
        engine_states = {}
        for engine_id in engine_ids:
            if engine_id in self.engines_data:
                engine_states[engine_id] = {
                    'current_index': 0,
                    'data': self.engines_data[engine_id]
                }
        
        start_time = time.time()
        
        while self.is_streaming and engine_states:
            current_time = time.time()
            
            # Stream data for each active engine
            for engine_id in list(engine_states.keys()):
                state = engine_states[engine_id]
                
                if state['current_index'] < len(state['data']):
                    # Get current data point
                    data_point = state['data'][state['current_index']].copy()
                    
                    # Update timestamp to current time
                    data_point['timestamp'] = current_time
                    data_point['stream_time'] = current_time - start_time
                    
                    # Add some noise to simulate real sensor readings
                    for key in data_point:
                        if key.startswith('sensor_') and not key.endswith('_trend'):
                            data_point[key] += random.gauss(0, 0.01)
                    
                    # Put data in queue and notify subscribers
                    self.streaming_queue.put(data_point)
                    self.notify_subscribers(data_point)
                    
                    # Move to next data point
                    state['current_index'] += 1
                else:
                    # Engine has finished its lifecycle, remove from active engines
                    print(f"Engine {engine_id} completed its lifecycle")
                    del engine_states[engine_id]
            
            # Sleep based on stream speed
            time.sleep(1.0 / self.stream_speed)
        
        print("Data streaming completed for all engines")
    
    def get_latest_data(self, max_items=100):
        """Get latest data from the streaming queue"""
        items = []
        count = 0
        
        while not self.streaming_queue.empty() and count < max_items:
            items.append(self.streaming_queue.get())
            count += 1
        
        return items
    
    def get_engine_status(self):
        """Get current status of all engines"""
        status = {}
        
        for engine_id, data in self.engines_data.items():
            if data:
                latest_point = data[-1]
                status[engine_id] = {
                    'total_cycles': len(data),
                    'max_rul': max(point.get('RUL', 0) for point in data),
                    'min_rul': min(point.get('RUL', 0) for point in data),
                    'avg_sensor_1': sum(point.get('sensor_1', 0) for point in data) / len(data)
                }
        
        return status

class MessageBroker:
    """Simple message broker for handling real-time communications"""
    
    def __init__(self):
        self.topics = {}
        self.subscribers = {}
    
    def create_topic(self, topic_name):
        """Create a new topic"""
        if topic_name not in self.topics:
            self.topics[topic_name] = Queue()
            self.subscribers[topic_name] = []
            print(f"Created topic: {topic_name}")
    
    def subscribe(self, topic_name, callback):
        """Subscribe to a topic"""
        if topic_name not in self.topics:
            self.create_topic(topic_name)
        
        self.subscribers[topic_name].append(callback)
        print(f"Subscribed to topic: {topic_name}")
    
    def publish(self, topic_name, message):
        """Publish a message to a topic"""
        if topic_name in self.topics:
            # Add to queue
            self.topics[topic_name].put(message)
            
            # Notify subscribers
            for callback in self.subscribers.get(topic_name, []):
                try:
                    callback(message)
                except Exception as e:
                    print(f"Error notifying subscriber: {e}")
    
    def get_messages(self, topic_name, max_messages=10):
        """Get messages from a topic"""
        messages = []
        if topic_name in self.topics:
            queue = self.topics[topic_name]
            count = 0
            
            while not queue.empty() and count < max_messages:
                messages.append(queue.get())
                count += 1
        
        return messages

class StreamingService:
    """Main service for handling real-time data streaming"""
    
    def __init__(self):
        self.data_streamer = DataStreamer()
        self.message_broker = MessageBroker()
        self.prediction_callbacks = []
        
        # Create topics
        self.message_broker.create_topic('sensor_data')
        self.message_broker.create_topic('anomaly_alerts')
        self.message_broker.create_topic('rul_predictions')
        
        # Subscribe to sensor data for processing
        self.message_broker.subscribe('sensor_data', self.process_sensor_data)
    
    def start_service(self, engines=None, speed=1.0):
        """Start the streaming service"""
        print("Starting streaming service...")
        
        # Subscribe streamer to broker
        self.data_streamer.subscribe(
            lambda data: self.message_broker.publish('sensor_data', data)
        )
        
        # Start streaming
        return self.data_streamer.start_streaming(engines, speed)
    
    def stop_service(self):
        """Stop the streaming service"""
        self.data_streamer.stop_streaming()
        print("Streaming service stopped")
    
    def process_sensor_data(self, data):
        """Process incoming sensor data for predictions"""
        # This would integrate with ML models
        engine_id = data.get('engine_id')
        cycle = data.get('cycle')
        
        # Simulate anomaly detection
        sensor_1 = data.get('sensor_1', 0)
        sensor_2 = data.get('sensor_2', 0)
        
        # Simple threshold-based anomaly detection for demo
        if abs(sensor_1) > 2.0 or abs(sensor_2) > 2.0:
            alert = {
                'engine_id': engine_id,
                'cycle': cycle,
                'timestamp': data.get('timestamp'),
                'alert_type': 'sensor_anomaly',
                'details': f'Sensor values outside normal range: sensor_1={sensor_1:.3f}, sensor_2={sensor_2:.3f}'
            }
            self.message_broker.publish('anomaly_alerts', alert)
        
        # Simulate RUL prediction
        estimated_rul = max(0, data.get('RUL', 100) + random.gauss(0, 10))
        prediction = {
            'engine_id': engine_id,
            'cycle': cycle,
            'timestamp': data.get('timestamp'),
            'predicted_rul': estimated_rul,
            'confidence': random.uniform(0.7, 0.95)
        }
        self.message_broker.publish('rul_predictions', prediction)
    
    def get_latest_sensor_data(self, max_items=50):
        """Get latest sensor data"""
        return self.message_broker.get_messages('sensor_data', max_items)
    
    def get_anomaly_alerts(self, max_items=10):
        """Get latest anomaly alerts"""
        return self.message_broker.get_messages('anomaly_alerts', max_items)
    
    def get_rul_predictions(self, max_items=20):
        """Get latest RUL predictions"""
        return self.message_broker.get_messages('rul_predictions', max_items)

if __name__ == "__main__":
    # Demo the streaming service
    service = StreamingService()
    
    print("Starting streaming demo...")
    
    # Start streaming with first 3 engines at 2x speed
    thread = service.start_service(engines=[1, 2, 3], speed=2.0)
    
    try:
        # Let it run for a few seconds
        time.sleep(10)
        
        # Get some data
        sensor_data = service.get_latest_sensor_data(5)
        alerts = service.get_anomaly_alerts(3)
        predictions = service.get_rul_predictions(5)
        
        print(f"\nCollected {len(sensor_data)} sensor readings")
        print(f"Generated {len(alerts)} anomaly alerts")
        print(f"Made {len(predictions)} RUL predictions")
        
        if sensor_data:
            print(f"\nSample sensor data: {sensor_data[0]}")
        
        if alerts:
            print(f"Sample alert: {alerts[0]}")
        
        if predictions:
            print(f"Sample prediction: {predictions[0]}")
    
    finally:
        service.stop_service()
        print("Demo completed")