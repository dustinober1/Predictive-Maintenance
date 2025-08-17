import asyncio
import csv
import json
import logging
import time
import random
import threading
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue, Empty, Full
from typing import List, Dict, Any, Optional, Callable, Set
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamState(Enum):
    """Enumeration for streaming states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class DataPoint:
    """Structured data point for streaming"""
    engine_id: int
    cycle: int
    timestamp: float
    sensors: Dict[str, float]
    settings: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class AlertData:
    """Structured alert data"""
    engine_id: int
    cycle: int
    timestamp: float
    alert_type: str
    severity: str
    message: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PredictionData:
    """Structured prediction data"""
    engine_id: int
    cycle: int
    timestamp: float
    predicted_rul: float
    confidence: float
    model_version: str
    features_used: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CircularBuffer:
    """Thread-safe circular buffer with size limits"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        
    def append(self, item: Any):
        """Add item to buffer"""
        with self.lock:
            self.buffer.append(item)
    
    def get_latest(self, count: int = None) -> List[Any]:
        """Get latest items from buffer"""
        with self.lock:
            if count is None:
                return list(self.buffer)
            return list(self.buffer)[-count:] if count <= len(self.buffer) else list(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)

class DataValidator:
    """Validates streaming data for quality and consistency"""
    
    def __init__(self):
        self.sensor_ranges = {}
        self.setting_ranges = {}
        self.validation_stats = {
            'total_validated': 0,
            'validation_errors': 0,
            'corrections_made': 0
        }
    
    def validate_data_point(self, data: Dict[str, Any]) -> tuple[bool, Dict[str, Any], List[str]]:
        """Validate a single data point"""
        is_valid = True
        corrected_data = data.copy()
        issues = []
        
        self.validation_stats['total_validated'] += 1
        
        try:
            # Validate required fields
            required_fields = ['engine_id', 'cycle', 'timestamp']
            for field in required_fields:
                if field not in data:
                    issues.append(f"Missing required field: {field}")
                    is_valid = False
                    continue
                
                # Type validation
                if field in ['engine_id', 'cycle'] and not isinstance(data[field], (int, float)):
                    try:
                        corrected_data[field] = int(float(data[field]))
                        self.validation_stats['corrections_made'] += 1
                    except (ValueError, TypeError):
                        issues.append(f"Invalid type for {field}: {type(data[field])}")
                        is_valid = False
                
                elif field == 'timestamp' and not isinstance(data[field], (int, float)):
                    try:
                        corrected_data[field] = float(data[field])
                        self.validation_stats['corrections_made'] += 1
                    except (ValueError, TypeError):
                        issues.append(f"Invalid timestamp: {data[field]}")
                        is_valid = False
            
            # Validate sensor data
            for key, value in data.items():
                if key.startswith('sensor_'):
                    if not isinstance(value, (int, float)):
                        try:
                            corrected_data[key] = float(value)
                            self.validation_stats['corrections_made'] += 1
                        except (ValueError, TypeError):
                            issues.append(f"Invalid sensor value for {key}: {value}")
                            corrected_data[key] = 0.0
                            is_valid = False
                    
                    # Range validation (if we have learned ranges)
                    if key in self.sensor_ranges:
                        min_val, max_val = self.sensor_ranges[key]
                        if not (min_val <= corrected_data[key] <= max_val):
                            issues.append(f"Sensor {key} value {corrected_data[key]} outside expected range [{min_val}, {max_val}]")
                            # Clamp to range
                            corrected_data[key] = max(min_val, min(max_val, corrected_data[key]))
                            self.validation_stats['corrections_made'] += 1
            
            # Validate settings
            for key, value in data.items():
                if key.startswith('setting_'):
                    if not isinstance(value, (int, float)):
                        try:
                            corrected_data[key] = float(value)
                            self.validation_stats['corrections_made'] += 1
                        except (ValueError, TypeError):
                            issues.append(f"Invalid setting value for {key}: {value}")
                            corrected_data[key] = 0.0
                            is_valid = False
            
            if not is_valid:
                self.validation_stats['validation_errors'] += 1
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            issues.append(f"Validation exception: {str(e)}")
            is_valid = False
            self.validation_stats['validation_errors'] += 1
        
        return is_valid, corrected_data, issues
    
    def learn_ranges(self, data_file: str):
        """Learn sensor and setting ranges from historical data"""
        try:
            logger.info(f"Learning data ranges from {data_file}")
            
            sensor_values = {}
            setting_values = {}
            
            with open(data_file, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    for key, value in row.items():
                        try:
                            numeric_value = float(value)
                            
                            if key.startswith('sensor_'):
                                if key not in sensor_values:
                                    sensor_values[key] = []
                                sensor_values[key].append(numeric_value)
                            
                            elif key.startswith('setting_'):
                                if key not in setting_values:
                                    setting_values[key] = []
                                setting_values[key].append(numeric_value)
                        
                        except (ValueError, TypeError):
                            continue
            
            # Calculate ranges (with some margin)
            for key, values in sensor_values.items():
                if values:
                    min_val, max_val = min(values), max(values)
                    range_margin = (max_val - min_val) * 0.1  # 10% margin
                    self.sensor_ranges[key] = (min_val - range_margin, max_val + range_margin)
            
            for key, values in setting_values.items():
                if values:
                    min_val, max_val = min(values), max(values)
                    range_margin = (max_val - min_val) * 0.1
                    self.setting_ranges[key] = (min_val - range_margin, max_val + range_margin)
            
            logger.info(f"Learned ranges for {len(self.sensor_ranges)} sensors and {len(self.setting_ranges)} settings")
            
        except Exception as e:
            logger.error(f"Error learning ranges: {e}")

class ImprovedDataStreamer:
    """Improved data streamer with robust error handling and memory management"""
    
    def __init__(self, data_file: str = 'data/processed/train_processed.csv', max_memory_mb: int = 100):
        self.data_file = data_file
        self.max_memory_mb = max_memory_mb
        self.engines_data = {}
        self.streaming_state = StreamState.STOPPED
        self.stream_speed = 1.0
        self.selected_engines = []
        
        # Thread management
        self.streaming_thread = None
        self.should_stop = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Subscribers with weak references to prevent memory leaks
        self.subscribers = weakref.WeakSet()
        
        # Data validation
        self.validator = DataValidator()
        
        # Streaming statistics
        self.stats = {
            'total_streamed': 0,
            'errors': 0,
            'start_time': None,
            'last_activity': None
        }
        
        # Engine state tracking
        self.engine_states = {}
        
        # Load data with error handling
        self.load_engine_data()
    
    def load_engine_data(self):
        """Load and organize data by engine with memory management"""
        try:
            logger.info(f"Loading streaming data from {self.data_file}")
            
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
            # Learn data ranges for validation
            self.validator.learn_ranges(self.data_file)
            
            # Load data in chunks to manage memory
            chunk_size = 1000
            total_rows = 0
            
            with open(self.data_file, 'r') as f:
                reader = csv.DictReader(f)
                
                current_chunk = []
                for row in reader:
                    try:
                        # Validate and clean data
                        is_valid, cleaned_row, issues = self.validator.validate_data_point(row)
                        
                        if not is_valid:
                            logger.warning(f"Data validation issues: {issues}")
                        
                        engine_id = int(float(cleaned_row['engine_id']))
                        cycle = int(float(cleaned_row['cycle']))
                        
                        # Create structured data point
                        sensors = {k: v for k, v in cleaned_row.items() if k.startswith('sensor_')}
                        settings = {k: v for k, v in cleaned_row.items() if k.startswith('setting_')}
                        metadata = {k: v for k, v in cleaned_row.items() 
                                  if not k.startswith(('sensor_', 'setting_')) and k not in ['engine_id', 'cycle']}
                        
                        data_point = DataPoint(
                            engine_id=engine_id,
                            cycle=cycle,
                            timestamp=time.time(),  # Will be updated during streaming
                            sensors=sensors,
                            settings=settings,
                            metadata=metadata
                        )
                        
                        # Group by engine
                        if engine_id not in self.engines_data:
                            self.engines_data[engine_id] = []
                        
                        self.engines_data[engine_id].append(data_point)
                        total_rows += 1
                        
                        # Memory management - limit data per engine
                        if len(self.engines_data[engine_id]) > 500:  # Max 500 cycles per engine
                            self.engines_data[engine_id].pop(0)  # Remove oldest
                        
                    except Exception as e:
                        logger.warning(f"Error processing row: {e}")
                        self.stats['errors'] += 1
                        continue
            
            # Sort each engine's data by cycle
            for engine_id in self.engines_data:
                self.engines_data[engine_id].sort(key=lambda x: x.cycle)
            
            logger.info(f"Loaded {total_rows} data points for {len(self.engines_data)} engines")
            logger.info(f"Validation stats: {self.validator.validation_stats}")
            
        except Exception as e:
            logger.error(f"Error loading engine data: {e}")
            raise
    
    def subscribe(self, callback: Callable):
        """Subscribe to data stream updates with weak reference"""
        self.subscribers.add(callback)
        logger.debug(f"Added subscriber: {callback}")
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from data stream updates"""
        self.subscribers.discard(callback)
        logger.debug(f"Removed subscriber: {callback}")
    
    def notify_subscribers(self, data: DataPoint):
        """Safely notify all subscribers of new data"""
        # Convert to weak set to list to avoid "set changed size during iteration"
        subscribers_list = list(self.subscribers)
        
        for callback in subscribers_list:
            try:
                callback(data.to_dict())
            except Exception as e:
                logger.warning(f"Error notifying subscriber {callback}: {e}")
                # Remove broken subscriber
                self.subscribers.discard(callback)
    
    async def start_streaming(self, selected_engines: List[int] = None, speed: float = 1.0) -> bool:
        """Start streaming data asynchronously"""
        try:
            if self.streaming_state in [StreamState.RUNNING, StreamState.STARTING]:
                logger.warning("Streaming already active")
                return False
            
            self.streaming_state = StreamState.STARTING
            self.stream_speed = speed
            self.should_stop.clear()
            
            if selected_engines is None:
                selected_engines = list(self.engines_data.keys())[:5]
            
            # Validate selected engines
            valid_engines = [e for e in selected_engines if e in self.engines_data]
            if not valid_engines:
                raise ValueError("No valid engines found in selected engines")
            
            self.selected_engines = valid_engines
            
            logger.info(f"Starting data stream for engines: {self.selected_engines} at {speed}x speed")
            
            # Initialize engine states
            self.engine_states = {}
            for engine_id in self.selected_engines:
                self.engine_states[engine_id] = {
                    'current_index': 0,
                    'data': self.engines_data[engine_id],
                    'last_streamed': None,
                    'error_count': 0
                }
            
            # Start streaming in executor
            self.streaming_thread = self.executor.submit(self._stream_data_sync)
            
            self.streaming_state = StreamState.RUNNING
            self.stats['start_time'] = time.time()
            self.stats['total_streamed'] = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            self.streaming_state = StreamState.ERROR
            return False
    
    def stop_streaming(self):
        """Stop data streaming gracefully"""
        if self.streaming_state == StreamState.STOPPED:
            return
        
        logger.info("Stopping data streaming...")
        self.streaming_state = StreamState.STOPPING
        self.should_stop.set()
        
        # Wait for streaming thread to finish
        if self.streaming_thread:
            try:
                self.streaming_thread.result(timeout=5.0)
            except Exception as e:
                logger.warning(f"Error stopping streaming thread: {e}")
        
        self.streaming_state = StreamState.STOPPED
        logger.info("Data streaming stopped")
    
    def _stream_data_sync(self):
        """Internal synchronous method to stream data"""
        start_time = time.time()
        last_cycle_time = start_time
        
        try:
            while not self.should_stop.is_set() and self.engine_states:
                current_time = time.time()
                
                # Check if it's time for next cycle based on speed
                cycle_interval = 1.0 / self.stream_speed
                if current_time - last_cycle_time < cycle_interval:
                    time.sleep(0.1)  # Small sleep to avoid busy waiting
                    continue
                
                last_cycle_time = current_time
                
                # Stream data for each active engine
                engines_to_remove = []
                
                for engine_id in list(self.engine_states.keys()):
                    try:
                        state = self.engine_states[engine_id]
                        
                        if state['current_index'] >= len(state['data']):
                            logger.info(f"Engine {engine_id} completed its lifecycle")
                            engines_to_remove.append(engine_id)
                            continue
                        
                        # Get current data point
                        data_point = state['data'][state['current_index']]
                        
                        # Create a copy with updated timestamp
                        streamed_point = DataPoint(
                            engine_id=data_point.engine_id,
                            cycle=data_point.cycle,
                            timestamp=current_time,
                            sensors=data_point.sensors.copy(),
                            settings=data_point.settings.copy(),
                            metadata={
                                **data_point.metadata,
                                'stream_time': current_time - start_time,
                                'stream_speed': self.stream_speed
                            }
                        )
                        
                        # Add realistic sensor noise
                        for sensor_name in streamed_point.sensors:
                            if not sensor_name.endswith('_trend'):
                                noise_level = abs(streamed_point.sensors[sensor_name]) * 0.01
                                streamed_point.sensors[sensor_name] += random.gauss(0, noise_level)
                        
                        # Notify subscribers
                        self.notify_subscribers(streamed_point)
                        
                        # Update state
                        state['current_index'] += 1
                        state['last_streamed'] = current_time
                        state['error_count'] = 0  # Reset error count on success
                        
                        self.stats['total_streamed'] += 1
                        self.stats['last_activity'] = current_time
                        
                    except Exception as e:
                        logger.error(f"Error streaming data for engine {engine_id}: {e}")
                        self.engine_states[engine_id]['error_count'] += 1
                        
                        # Remove engine if too many errors
                        if self.engine_states[engine_id]['error_count'] > 5:
                            logger.error(f"Too many errors for engine {engine_id}, removing from stream")
                            engines_to_remove.append(engine_id)
                        
                        self.stats['errors'] += 1
                
                # Remove completed or failed engines
                for engine_id in engines_to_remove:
                    if engine_id in self.engine_states:
                        del self.engine_states[engine_id]
                
                # Check if all engines are done
                if not self.engine_states:
                    logger.info("All engines completed streaming")
                    break
            
        except Exception as e:
            logger.error(f"Critical error in streaming: {e}")
            logger.error(traceback.format_exc())
            self.streaming_state = StreamState.ERROR
            self.stats['errors'] += 1
        
        finally:
            logger.info("Data streaming loop completed")
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get current streaming statistics"""
        stats = self.stats.copy()
        stats['state'] = self.streaming_state.value
        stats['active_engines'] = len(self.engine_states)
        stats['validation_stats'] = self.validator.validation_stats
        
        if stats['start_time']:
            stats['uptime'] = time.time() - stats['start_time']
            if stats['uptime'] > 0:
                stats['streaming_rate'] = stats['total_streamed'] / stats['uptime']
        
        return stats
    
    def get_engine_status(self) -> Dict[int, Dict[str, Any]]:
        """Get current status of all engines"""
        status = {}
        
        for engine_id, data_points in self.engines_data.items():
            if data_points:
                # Calculate statistics
                cycles = [dp.cycle for dp in data_points]
                
                status[engine_id] = {
                    'total_cycles': len(data_points),
                    'max_cycle': max(cycles),
                    'min_cycle': min(cycles),
                    'is_streaming': engine_id in self.engine_states,
                    'current_cycle': None,
                    'completion_percent': 0.0
                }
                
                # Add streaming status if active
                if engine_id in self.engine_states:
                    state = self.engine_states[engine_id]
                    current_idx = state['current_index']
                    
                    if current_idx < len(data_points):
                        status[engine_id]['current_cycle'] = data_points[current_idx].cycle
                        status[engine_id]['completion_percent'] = (current_idx / len(data_points)) * 100
                    
                    status[engine_id]['errors'] = state.get('error_count', 0)
                    status[engine_id]['last_streamed'] = state.get('last_streamed')
        
        return status
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_streaming()
        self.executor.shutdown(wait=True)
        self.engines_data.clear()
        self.engine_states.clear()
        logger.info("DataStreamer cleanup completed")

class ImprovedMessageBroker:
    """Improved message broker with better error handling and memory management"""
    
    def __init__(self, max_buffer_size: int = 1000):
        self.topics = {}
        self.subscribers = {}
        self.max_buffer_size = max_buffer_size
        self.message_stats = {}
        self.lock = threading.RLock()
    
    def create_topic(self, topic_name: str):
        """Create a new topic with circular buffer"""
        with self.lock:
            if topic_name not in self.topics:
                self.topics[topic_name] = CircularBuffer(self.max_buffer_size)
                self.subscribers[topic_name] = weakref.WeakSet()
                self.message_stats[topic_name] = {
                    'total_published': 0,
                    'total_consumed': 0,
                    'last_activity': None
                }
                logger.info(f"Created topic: {topic_name}")
    
    def subscribe(self, topic_name: str, callback: Callable):
        """Subscribe to a topic with weak reference"""
        if topic_name not in self.topics:
            self.create_topic(topic_name)
        
        with self.lock:
            self.subscribers[topic_name].add(callback)
            logger.debug(f"Subscribed to topic: {topic_name}")
    
    def unsubscribe(self, topic_name: str, callback: Callable):
        """Unsubscribe from a topic"""
        if topic_name in self.subscribers:
            with self.lock:
                self.subscribers[topic_name].discard(callback)
                logger.debug(f"Unsubscribed from topic: {topic_name}")
    
    def publish(self, topic_name: str, message: Any):
        """Publish a message to a topic"""
        if topic_name not in self.topics:
            self.create_topic(topic_name)
        
        try:
            with self.lock:
                # Add to buffer
                self.topics[topic_name].append(message)
                
                # Update stats
                self.message_stats[topic_name]['total_published'] += 1
                self.message_stats[topic_name]['last_activity'] = time.time()
                
                # Notify subscribers
                subscribers_list = list(self.subscribers[topic_name])
            
            # Notify outside of lock to avoid deadlock
            for callback in subscribers_list:
                try:
                    callback(message)
                except Exception as e:
                    logger.warning(f"Error notifying subscriber for topic {topic_name}: {e}")
                    # Remove broken subscriber
                    with self.lock:
                        self.subscribers[topic_name].discard(callback)
        
        except Exception as e:
            logger.error(f"Error publishing to topic {topic_name}: {e}")
    
    def get_messages(self, topic_name: str, max_messages: int = 10) -> List[Any]:
        """Get latest messages from a topic"""
        if topic_name not in self.topics:
            return []
        
        try:
            with self.lock:
                messages = self.topics[topic_name].get_latest(max_messages)
                self.message_stats[topic_name]['total_consumed'] += len(messages)
                return messages
        
        except Exception as e:
            logger.error(f"Error getting messages from topic {topic_name}: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics"""
        with self.lock:
            stats = {
                'topics': list(self.topics.keys()),
                'total_topics': len(self.topics),
                'topic_stats': {}
            }
            
            for topic_name in self.topics:
                stats['topic_stats'][topic_name] = {
                    **self.message_stats[topic_name],
                    'buffer_size': self.topics[topic_name].size(),
                    'subscriber_count': len(self.subscribers[topic_name])
                }
            
            return stats

class ImprovedStreamingService:
    """Main service with comprehensive error handling and monitoring"""
    
    def __init__(self, data_file: str = 'data/processed/train_processed.csv'):
        self.data_streamer = ImprovedDataStreamer(data_file)
        self.message_broker = ImprovedMessageBroker()
        self.is_running = False
        self.start_time = None
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.health_monitor_thread = None
        self.last_health_check = None
        
        # Create topics
        self._setup_topics()
        
        # Subscribe to sensor data for processing
        self.message_broker.subscribe('sensor_data', self._process_sensor_data)
    
    def _setup_topics(self):
        """Setup required message broker topics"""
        topics = ['sensor_data', 'anomaly_alerts', 'rul_predictions', 'system_events']
        for topic in topics:
            self.message_broker.create_topic(topic)
    
    async def start_service(self, engines: List[int] = None, speed: float = 1.0) -> bool:
        """Start the streaming service"""
        try:
            if self.is_running:
                logger.warning("Service already running")
                return False
            
            logger.info("Starting improved streaming service...")
            
            # Subscribe streamer to broker
            self.data_streamer.subscribe(
                lambda data: self.message_broker.publish('sensor_data', data)
            )
            
            # Start streaming
            success = await self.data_streamer.start_streaming(engines, speed)
            
            if success:
                self.is_running = True
                self.start_time = time.time()
                
                # Start health monitoring
                self._start_health_monitoring()
                
                # Publish start event
                self.message_broker.publish('system_events', {
                    'event_type': 'service_started',
                    'timestamp': time.time(),
                    'engines': engines,
                    'speed': speed
                })
                
                logger.info("Streaming service started successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting service: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def stop_service(self):
        """Stop the streaming service"""
        if not self.is_running:
            logger.warning("Service not running")
            return
        
        logger.info("Stopping streaming service...")
        
        try:
            # Stop health monitoring
            self._stop_health_monitoring()
            
            # Stop streaming
            self.data_streamer.stop_streaming()
            
            self.is_running = False
            
            # Publish stop event
            self.message_broker.publish('system_events', {
                'event_type': 'service_stopped',
                'timestamp': time.time(),
                'uptime': time.time() - self.start_time if self.start_time else 0
            })
            
            logger.info("Streaming service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping service: {e}")
    
    def _process_sensor_data(self, data: Dict[str, Any]):
        """Process incoming sensor data for ML predictions"""
        try:
            engine_id = data.get('engine_id')
            cycle = data.get('cycle')
            timestamp = data.get('timestamp', time.time())
            
            # Enhanced anomaly detection logic
            sensors = data.get('sensors', {})
            
            # Check for multiple anomaly types
            anomalies_detected = []
            
            # Sensor threshold anomalies
            for sensor_name, value in sensors.items():
                if isinstance(value, (int, float)):
                    if abs(value) > 3.0:  # Example threshold
                        anomalies_detected.append({
                            'type': 'threshold_exceeded',
                            'sensor': sensor_name,
                            'value': value,
                            'threshold': 3.0
                        })
            
            # Sensor correlation anomalies (example)
            if 'sensor_1' in sensors and 'sensor_2' in sensors:
                correlation_score = abs(sensors['sensor_1'] - sensors['sensor_2'])
                if correlation_score > 2.5:
                    anomalies_detected.append({
                        'type': 'correlation_anomaly',
                        'sensors': ['sensor_1', 'sensor_2'],
                        'correlation_score': correlation_score
                    })
            
            # Publish anomaly alerts
            for anomaly in anomalies_detected:
                alert = AlertData(
                    engine_id=engine_id,
                    cycle=cycle,
                    timestamp=timestamp,
                    alert_type=anomaly['type'],
                    severity='medium',
                    message=f"Anomaly detected: {anomaly['type']}",
                    details=anomaly
                )
                self.message_broker.publish('anomaly_alerts', alert.to_dict())
            
            # Enhanced RUL prediction
            # This would integrate with the improved ML models
            base_rul = data.get('metadata', {}).get('RUL', 100)
            if base_rul is not None:
                # Add some realistic prediction logic
                degradation_factor = max(0, (cycle - 1) / 100.0)  # Simplified degradation
                noise = random.gauss(0, 5)
                predicted_rul = max(0, base_rul * (1 - degradation_factor) + noise)
                
                prediction = PredictionData(
                    engine_id=engine_id,
                    cycle=cycle,
                    timestamp=timestamp,
                    predicted_rul=predicted_rul,
                    confidence=max(0.5, 1.0 - degradation_factor),
                    model_version="improved_v1.0",
                    features_used=list(sensors.keys())
                )
                self.message_broker.publish('rul_predictions', prediction.to_dict())
        
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
    
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        def health_monitor():
            while self.is_running:
                try:
                    self._perform_health_check()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health monitor error: {e}")
                    time.sleep(self.health_check_interval)
        
        self.health_monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        self.health_monitor_thread.start()
        logger.info("Health monitoring started")
    
    def _stop_health_monitoring(self):
        """Stop health monitoring"""
        # Health monitor will stop when self.is_running becomes False
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=5.0)
    
    def _perform_health_check(self):
        """Perform system health check"""
        try:
            health_status = {
                'timestamp': time.time(),
                'service_running': self.is_running,
                'streamer_stats': self.data_streamer.get_streaming_stats(),
                'broker_stats': self.message_broker.get_stats(),
                'memory_usage': self._get_memory_usage()
            }
            
            self.last_health_check = health_status
            
            # Check for issues and publish alerts if needed
            issues = self._analyze_health(health_status)
            if issues:
                for issue in issues:
                    self.message_broker.publish('system_events', {
                        'event_type': 'health_issue',
                        'timestamp': time.time(),
                        'issue': issue
                    })
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    def _analyze_health(self, health_status: Dict[str, Any]) -> List[str]:
        """Analyze health status and return issues"""
        issues = []
        
        try:
            # Check memory usage
            memory = health_status.get('memory_usage', {})
            if memory.get('rss_mb', 0) > 500:  # 500MB threshold
                issues.append(f"High memory usage: {memory.get('rss_mb', 0):.1f} MB")
            
            # Check streaming errors
            streamer_stats = health_status.get('streamer_stats', {})
            error_rate = streamer_stats.get('errors', 0) / max(1, streamer_stats.get('total_streamed', 1))
            if error_rate > 0.05:  # 5% error rate threshold
                issues.append(f"High error rate: {error_rate:.2%}")
            
            # Check last activity
            last_activity = streamer_stats.get('last_activity')
            if last_activity and time.time() - last_activity > 60:  # 1 minute threshold
                issues.append("No streaming activity in last 60 seconds")
        
        except Exception as e:
            issues.append(f"Health analysis error: {e}")
        
        return issues
    
    # Public API methods
    def get_latest_sensor_data(self, max_items: int = 50) -> List[Dict[str, Any]]:
        """Get latest sensor data"""
        return self.message_broker.get_messages('sensor_data', max_items)
    
    def get_anomaly_alerts(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """Get latest anomaly alerts"""
        return self.message_broker.get_messages('anomaly_alerts', max_items)
    
    def get_rul_predictions(self, max_items: int = 20) -> List[Dict[str, Any]]:
        """Get latest RUL predictions"""
        return self.message_broker.get_messages('rul_predictions', max_items)
    
    def get_system_events(self, max_items: int = 20) -> List[Dict[str, Any]]:
        """Get latest system events"""
        return self.message_broker.get_messages('system_events', max_items)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            'is_running': self.is_running,
            'start_time': self.start_time,
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'last_health_check': self.last_health_check,
            'engine_status': self.data_streamer.get_engine_status(),
            'streaming_stats': self.data_streamer.get_streaming_stats(),
            'broker_stats': self.message_broker.get_stats()
        }
    
    def cleanup(self):
        """Cleanup all resources"""
        self.stop_service()
        self.data_streamer.cleanup()
        logger.info("ImprovedStreamingService cleanup completed")

if __name__ == "__main__":
    # Demo the improved streaming service
    async def demo():
        service = ImprovedStreamingService()
        
        try:
            logger.info("Starting improved streaming demo...")
            
            # Start streaming with first 3 engines at 2x speed
            success = await service.start_service(engines=[1, 2, 3], speed=2.0)
            
            if not success:
                logger.error("Failed to start service")
                return
            
            # Let it run for a while
            await asyncio.sleep(15)
            
            # Get status and data
            status = service.get_service_status()
            sensor_data = service.get_latest_sensor_data(5)
            alerts = service.get_anomaly_alerts(3)
            predictions = service.get_rul_predictions(5)
            events = service.get_system_events(5)
            
            logger.info(f"Service status: {status['is_running']}")
            logger.info(f"Collected {len(sensor_data)} sensor readings")
            logger.info(f"Generated {len(alerts)} anomaly alerts")
            logger.info(f"Made {len(predictions)} RUL predictions")
            logger.info(f"System events: {len(events)}")
            
            if sensor_data:
                logger.info(f"Sample sensor data: {sensor_data[0]}")
            
        finally:
            service.cleanup()
            logger.info("Demo completed")
    
    # Run the demo
    asyncio.run(demo())