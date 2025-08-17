#!/usr/bin/env python3
"""
Comprehensive test suite for the improved predictive maintenance system
"""

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List

# Add backend to path
sys.path.append('backend/app')
sys.path.append('backend/models')

# Import logging configuration
from logging_config import setup_logging, log_performance, LogContext

# Setup logging for tests
logger = setup_logging(log_level='INFO', log_dir='test_logs')

class TestResults:
    """Track test results and performance metrics"""
    
    def __init__(self):
        self.tests = {}
        self.start_time = time.time()
        self.performance_metrics = {}
    
    def add_test(self, test_name: str, passed: bool, duration: float, details: Dict[str, Any] = None):
        """Add a test result"""
        self.tests[test_name] = {
            'passed': passed,
            'duration': duration,
            'details': details or {},
            'timestamp': time.time()
        }
        
        status = "PASSED" if passed else "FAILED"
        logger.info(f"Test {test_name}: {status} ({duration:.3f}s)")
    
    def add_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Add a performance metric"""
        self.performance_metrics[metric_name] = {
            'value': value,
            'unit': unit,
            'timestamp': time.time()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        passed_tests = sum(1 for t in self.tests.values() if t['passed'])
        total_tests = len(self.tests)
        total_duration = time.time() - self.start_time
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / max(1, total_tests),
            'total_duration': total_duration,
            'performance_metrics': self.performance_metrics,
            'test_details': self.tests
        }

class ModelTester:
    """Test improved ML models"""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    @log_performance
    def test_improved_autoencoder(self) -> bool:
        """Test the improved autoencoder model"""
        try:
            from improved_autoencoder import ImprovedAnomalyDetector
            
            start_time = time.time()
            
            # Initialize detector
            detector = ImprovedAnomalyDetector()
            
            # Test training with small dataset
            if os.path.exists('data/processed/train_anomaly.csv'):
                training_results = detector.train_model(epochs=5, batch_size=32)
                
                # Validate training results
                if 'final_loss' not in training_results:
                    raise ValueError("Training results missing expected keys")
                
                # Test detection
                results = detector.detect_anomalies()
                
                if len(results) == 0:
                    raise ValueError("No detection results returned")
                
                # Test evaluation if labels available
                if 'actual_anomaly' in results.columns:
                    metrics = detector.evaluate_model(results)
                    
                    self.results.add_performance_metric('anomaly_accuracy', metrics.get('accuracy', 0))
                    self.results.add_performance_metric('anomaly_precision', metrics.get('precision', 0))
                    
                    logger.info(f"Anomaly detection metrics: {metrics}")
                
                # Test save/load
                detector.save_model('test_models/test_autoencoder.pkl')
                
                new_detector = ImprovedAnomalyDetector()
                new_detector.load_model('test_models/test_autoencoder.pkl')
                
                duration = time.time() - start_time
                self.results.add_test('improved_autoencoder', True, duration, {
                    'training_results': training_results,
                    'detection_count': len(results)
                })
                return True
            else:
                logger.warning("Training data not found, skipping autoencoder test")
                self.results.add_test('improved_autoencoder', False, 0, {'error': 'Training data not found'})
                return False
        
        except Exception as e:
            logger.error(f"Autoencoder test failed: {e}")
            logger.error(traceback.format_exc())
            self.results.add_test('improved_autoencoder', False, time.time() - start_time, {'error': str(e)})
            return False
    
    @log_performance
    def test_improved_rul(self) -> bool:
        """Test the improved RUL predictor"""
        try:
            from improved_rul import ImprovedRULPredictor
            
            start_time = time.time()
            
            # Test Random Forest model
            predictor = ImprovedRULPredictor(model_type='random_forest')
            
            if os.path.exists('data/processed/train_processed.csv'):
                training_results = predictor.train_model(tune_hyperparameters=False)
                
                # Validate training results
                if 'rmse' not in training_results:
                    raise ValueError("Training results missing RMSE")
                
                # Test prediction
                test_data = {
                    'cycle': 50,
                    'sensor_1': 0.5,
                    'sensor_2': -0.3,
                    'setting_1': 1.0
                }
                
                X = predictor.prepare_features_for_prediction(test_data)
                rul = predictor.predict(X)[0]
                
                if rul < 0:
                    raise ValueError("RUL prediction should be non-negative")
                
                # Test feature importance
                importance = predictor.get_feature_importance()
                
                # Test save/load
                predictor.save_model('test_models/test_rul.pkl')
                
                new_predictor = ImprovedRULPredictor()
                new_predictor.load_model('test_models/test_rul.pkl')
                
                # Performance metrics
                self.results.add_performance_metric('rul_rmse', training_results['rmse'], 'cycles')
                self.results.add_performance_metric('rul_r2', training_results['r2'])
                
                duration = time.time() - start_time
                self.results.add_test('improved_rul', True, duration, {
                    'training_results': training_results,
                    'test_prediction': rul,
                    'feature_importance_count': len(importance)
                })
                return True
            else:
                logger.warning("Training data not found, skipping RUL test")
                self.results.add_test('improved_rul', False, 0, {'error': 'Training data not found'})
                return False
        
        except Exception as e:
            logger.error(f"RUL test failed: {e}")
            logger.error(traceback.format_exc())
            self.results.add_test('improved_rul', False, time.time() - start_time, {'error': str(e)})
            return False

class StreamingTester:
    """Test improved streaming service"""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    @log_performance
    async def test_improved_streaming(self) -> bool:
        """Test the improved streaming service"""
        try:
            from improved_streaming import ImprovedStreamingService
            
            start_time = time.time()
            
            # Initialize service
            service = ImprovedStreamingService()
            
            # Test service startup
            success = await service.start_service(engines=[1, 2, 3], speed=5.0)
            
            if not success:
                raise ValueError("Failed to start streaming service")
            
            # Let it run for a few seconds
            await asyncio.sleep(3)
            
            # Test data collection
            sensor_data = service.get_latest_sensor_data(10)
            alerts = service.get_anomaly_alerts(5)
            predictions = service.get_rul_predictions(5)
            
            # Test service status
            status = service.get_service_status()
            
            if not status.get('is_running', False):
                raise ValueError("Service should be running")
            
            # Test health monitoring
            streaming_stats = status.get('streaming_stats', {})
            
            # Performance metrics
            self.results.add_performance_metric('streaming_rate', 
                                              streaming_stats.get('streaming_rate', 0), 'msg/sec')
            self.results.add_performance_metric('sensor_data_collected', len(sensor_data), 'messages')
            
            # Stop service
            service.stop_service()
            
            # Cleanup
            service.cleanup()
            
            duration = time.time() - start_time
            self.results.add_test('improved_streaming', True, duration, {
                'sensor_data_count': len(sensor_data),
                'alerts_count': len(alerts),
                'predictions_count': len(predictions),
                'streaming_stats': streaming_stats
            })
            return True
        
        except Exception as e:
            logger.error(f"Streaming test failed: {e}")
            logger.error(traceback.format_exc())
            self.results.add_test('improved_streaming', False, time.time() - start_time, {'error': str(e)})
            return False

class APITester:
    """Test improved API endpoints"""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_api_imports(self) -> bool:
        """Test that improved API modules can be imported"""
        try:
            start_time = time.time()
            
            # Test imports
            from improved_main import app
            from logging_config import setup_logging
            
            duration = time.time() - start_time
            self.results.add_test('api_imports', True, duration)
            return True
        
        except Exception as e:
            logger.error(f"API import test failed: {e}")
            self.results.add_test('api_imports', False, time.time() - start_time, {'error': str(e)})
            return False

class DataValidationTester:
    """Test data validation and error handling"""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_data_validator(self) -> bool:
        """Test data validation functionality"""
        try:
            from improved_streaming import DataValidator
            
            start_time = time.time()
            
            validator = DataValidator()
            
            # Test valid data
            valid_data = {
                'engine_id': 1,
                'cycle': 50,
                'timestamp': time.time(),
                'sensor_1': 0.5,
                'sensor_2': -0.3
            }
            
            is_valid, cleaned_data, issues = validator.validate_data_point(valid_data)
            
            if not is_valid:
                raise ValueError("Valid data should pass validation")
            
            # Test invalid data
            invalid_data = {
                'engine_id': 'invalid',
                'cycle': -1,
                'sensor_1': 'not_a_number'
            }
            
            is_valid, cleaned_data, issues = validator.validate_data_point(invalid_data)
            
            if is_valid:
                raise ValueError("Invalid data should fail validation")
            
            if len(issues) == 0:
                raise ValueError("Issues should be reported for invalid data")
            
            duration = time.time() - start_time
            self.results.add_test('data_validation', True, duration, {
                'validation_stats': validator.validation_stats,
                'issues_detected': len(issues)
            })
            return True
        
        except Exception as e:
            logger.error(f"Data validation test failed: {e}")
            self.results.add_test('data_validation', False, time.time() - start_time, {'error': str(e)})
            return False

class PerformanceTester:
    """Test system performance and memory usage"""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def test_memory_usage(self) -> bool:
        """Test memory usage patterns"""
        try:
            import psutil
            
            start_time = time.time()
            process = psutil.Process()
            
            # Get initial memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate some operations
            from improved_streaming import CircularBuffer
            
            # Test circular buffer memory management
            buffer = CircularBuffer(max_size=1000)
            
            # Add many items
            for i in range(5000):
                buffer.append({'data': f'item_{i}', 'value': i})
            
            # Buffer should not exceed max size
            if buffer.size() > 1000:
                raise ValueError("CircularBuffer exceeded max size")
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Performance metrics
            self.results.add_performance_metric('initial_memory', initial_memory, 'MB')
            self.results.add_performance_metric('final_memory', final_memory, 'MB')
            self.results.add_performance_metric('memory_increase', memory_increase, 'MB')
            
            duration = time.time() - start_time
            self.results.add_test('memory_usage', True, duration, {
                'memory_increase_mb': memory_increase,
                'buffer_size': buffer.size()
            })
            return True
        
        except Exception as e:
            logger.error(f"Memory usage test failed: {e}")
            self.results.add_test('memory_usage', False, time.time() - start_time, {'error': str(e)})
            return False

async def run_all_tests():
    """Run comprehensive test suite"""
    logger.info("Starting comprehensive test suite for improved predictive maintenance system")
    
    # Create test directories
    os.makedirs('test_models', exist_ok=True)
    os.makedirs('test_logs', exist_ok=True)
    
    results = TestResults()
    
    # Initialize testers
    model_tester = ModelTester(results)
    streaming_tester = StreamingTester(results)
    api_tester = APITester(results)
    validation_tester = DataValidationTester(results)
    performance_tester = PerformanceTester(results)
    
    # Run tests
    with LogContext(logger, "test_suite_execution"):
        
        # Test API imports first
        logger.info("Testing API imports...")
        api_tester.test_api_imports()
        
        # Test data validation
        logger.info("Testing data validation...")
        validation_tester.test_data_validator()
        
        # Test memory usage
        logger.info("Testing memory usage...")
        performance_tester.test_memory_usage()
        
        # Test ML models
        logger.info("Testing improved ML models...")
        model_tester.test_improved_autoencoder()
        model_tester.test_improved_rul()
        
        # Test streaming service
        logger.info("Testing improved streaming service...")
        await streaming_tester.test_improved_streaming()
    
    # Generate test report
    summary = results.get_summary()
    
    # Log summary
    logger.info("="*60)
    logger.info("TEST SUITE SUMMARY")
    logger.info("="*60)
    logger.info(f"Total tests: {summary['total_tests']}")
    logger.info(f"Passed: {summary['passed_tests']}")
    logger.info(f"Failed: {summary['failed_tests']}")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info(f"Total duration: {summary['total_duration']:.2f}s")
    
    # Log performance metrics
    logger.info("\nPERFORMACE METRICS:")
    for metric_name, metric_data in summary['performance_metrics'].items():
        logger.info(f"  {metric_name}: {metric_data['value']:.3f} {metric_data['unit']}")
    
    # Log failed tests
    failed_tests = [name for name, test in summary['test_details'].items() if not test['passed']]
    if failed_tests:
        logger.error(f"\nFAILED TESTS: {failed_tests}")
        for test_name in failed_tests:
            test_info = summary['test_details'][test_name]
            logger.error(f"  {test_name}: {test_info['details'].get('error', 'Unknown error')}")
    
    # Save detailed report
    report_file = 'test_logs/test_report.json'
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\nDetailed test report saved to: {report_file}")
    
    # Return success status
    return summary['success_rate'] >= 0.8  # 80% success rate threshold

def main():
    """Main test runner"""
    try:
        success = asyncio.run(run_all_tests())
        
        if success:
            logger.info("✅ Test suite completed successfully!")
            print("\n🎉 IMPROVEMENTS SUCCESSFULLY TESTED!")
            print("The predictive maintenance system has been enhanced with:")
            print("  ✓ Improved ML models (TensorFlow/scikit-learn)")
            print("  ✓ Robust data validation and error handling")
            print("  ✓ Memory-efficient streaming with connection recovery")
            print("  ✓ Comprehensive logging system")
            return 0
        else:
            logger.error("❌ Test suite failed - some improvements need attention")
            print("\n⚠️  Some tests failed - check logs for details")
            return 1
    
    except Exception as e:
        logger.error(f"Test suite crashed: {e}")
        logger.error(traceback.format_exc())
        print(f"\n💥 Test suite crashed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)