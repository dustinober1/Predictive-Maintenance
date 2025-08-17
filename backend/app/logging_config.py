import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class PerformanceFilter(logging.Filter):
    """Filter to track performance metrics"""
    
    def __init__(self):
        super().__init__()
        self.performance_stats = {
            'total_logs': 0,
            'errors': 0,
            'warnings': 0,
            'start_time': datetime.now()
        }
    
    def filter(self, record):
        # Update stats
        self.performance_stats['total_logs'] += 1
        
        if record.levelno >= logging.ERROR:
            self.performance_stats['errors'] += 1
        elif record.levelno >= logging.WARNING:
            self.performance_stats['warnings'] += 1
        
        # Add performance context to record
        record.log_id = self.performance_stats['total_logs']
        
        return True
    
    def get_stats(self):
        """Get performance statistics"""
        uptime = datetime.now() - self.performance_stats['start_time']
        return {
            **self.performance_stats,
            'uptime_seconds': uptime.total_seconds(),
            'logs_per_second': self.performance_stats['total_logs'] / max(1, uptime.total_seconds())
        }

def setup_logging(
    log_level: str = 'INFO',
    log_dir: str = 'logs',
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    enable_file: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        max_file_size: Maximum size of each log file in bytes
        backup_count: Number of backup files to keep
        enable_console: Enable console logging
        enable_file: Enable file logging
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create performance filter
    perf_filter = PerformanceFilter()
    
    # Console handler with colors
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(perf_filter)
        
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        # Main application log
        app_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'app.log'),
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        app_handler.setLevel(logging.INFO)
        
        app_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(log_id)s] - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        app_handler.setFormatter(app_formatter)
        app_handler.addFilter(perf_filter)
        
        root_logger.addHandler(app_handler)
        
        # Error log (errors and above only)
        error_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'errors.log'),
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        
        error_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(log_id)s] - %(funcName)s:%(lineno)d - %(message)s\n%(exc_info)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        error_handler.addFilter(perf_filter)
        
        root_logger.addHandler(error_handler)
        
        # Performance log (for metrics and timing)
        perf_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'performance.log'),
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        perf_handler.setLevel(logging.INFO)
        
        # Custom filter for performance logs
        class PerfLogFilter(logging.Filter):
            def filter(self, record):
                return 'performance' in record.getMessage().lower() or 'timing' in record.getMessage().lower()
        
        perf_handler.addFilter(PerfLogFilter())
        
        perf_formatter = logging.Formatter(
            fmt='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_formatter)
        
        root_logger.addHandler(perf_handler)
    
    # Configure specific loggers
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
    
    # Create application logger
    app_logger = logging.getLogger('predictive_maintenance')
    
    # Add performance stats method to logger
    app_logger.get_performance_stats = perf_filter.get_stats
    
    app_logger.info(f"Logging configured - Level: {log_level}, Console: {enable_console}, File: {enable_file}")
    
    return app_logger

def log_performance(func):
    """Decorator to log function performance"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('predictive_maintenance.performance')
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(f"Performance - {func.__name__}: {execution_time:.4f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Performance - {func.__name__} FAILED after {execution_time:.4f}s: {e}")
            raise
    
    return wrapper

class LogContext:
    """Context manager for structured logging"""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        import time
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
        self.time = time
    
    def __enter__(self):
        self.start_time = self.time.time()
        self.logger.info(f"Starting {self.operation}", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = self.time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation} in {execution_time:.4f}s", 
                extra={**self.context, 'execution_time': execution_time}
            )
        else:
            self.logger.error(
                f"Failed {self.operation} after {execution_time:.4f}s: {exc_val}", 
                extra={**self.context, 'execution_time': execution_time},
                exc_info=True
            )

def create_operation_logger(operation_name: str) -> logging.Logger:
    """Create a logger for specific operations"""
    logger = logging.getLogger(f'predictive_maintenance.{operation_name}')
    return logger

# Usage examples and utilities
class HealthLogger:
    """Specialized logger for health monitoring"""
    
    def __init__(self):
        self.logger = create_operation_logger('health')
        self.health_stats = {
            'checks_performed': 0,
            'issues_detected': 0,
            'last_check': None
        }
    
    def log_health_check(self, status: str, details: dict = None):
        """Log a health check result"""
        self.health_stats['checks_performed'] += 1
        self.health_stats['last_check'] = datetime.now()
        
        if status != 'healthy':
            self.health_stats['issues_detected'] += 1
        
        self.logger.info(
            f"Health check: {status}",
            extra={
                'health_status': status,
                'details': details or {},
                'check_number': self.health_stats['checks_performed']
            }
        )
    
    def get_health_stats(self):
        """Get health logging statistics"""
        return self.health_stats.copy()

class APILogger:
    """Specialized logger for API requests"""
    
    def __init__(self):
        self.logger = create_operation_logger('api')
        self.request_stats = {
            'total_requests': 0,
            'error_requests': 0,
            'response_times': []
        }
    
    def log_request(self, method: str, path: str, status_code: int, response_time: float):
        """Log an API request"""
        self.request_stats['total_requests'] += 1
        self.request_stats['response_times'].append(response_time)
        
        if status_code >= 400:
            self.request_stats['error_requests'] += 1
        
        # Keep only last 1000 response times for memory management
        if len(self.request_stats['response_times']) > 1000:
            self.request_stats['response_times'] = self.request_stats['response_times'][-1000:]
        
        level = logging.ERROR if status_code >= 400 else logging.INFO
        
        self.logger.log(
            level,
            f"{method} {path} - {status_code} - {response_time:.4f}s",
            extra={
                'method': method,
                'path': path,
                'status_code': status_code,
                'response_time': response_time,
                'request_number': self.request_stats['total_requests']
            }
        )
    
    def get_api_stats(self):
        """Get API logging statistics"""
        stats = self.request_stats.copy()
        
        if stats['response_times']:
            import statistics
            stats['avg_response_time'] = statistics.mean(stats['response_times'])
            stats['median_response_time'] = statistics.median(stats['response_times'])
            stats['max_response_time'] = max(stats['response_times'])
        else:
            stats['avg_response_time'] = 0
            stats['median_response_time'] = 0
            stats['max_response_time'] = 0
        
        # Don't include the full response times list in stats
        del stats['response_times']
        
        return stats

if __name__ == "__main__":
    # Demo the logging system
    import time
    
    # Setup logging
    logger = setup_logging(log_level='DEBUG', enable_console=True, enable_file=True)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test performance logging
    @log_performance
    def example_function():
        time.sleep(0.1)
        return "result"
    
    result = example_function()
    
    # Test context manager
    with LogContext(logger, "example_operation", user_id=123, action="test"):
        time.sleep(0.05)
        logger.info("Inside context operation")
    
    # Test specialized loggers
    health_logger = HealthLogger()
    health_logger.log_health_check("healthy", {"cpu": 45, "memory": 67})
    
    api_logger = APILogger()
    api_logger.log_request("GET", "/api/test", 200, 0.123)
    
    # Show performance stats
    print(f"Logger performance stats: {logger.get_performance_stats()}")
    print(f"Health stats: {health_logger.get_health_stats()}")
    print(f"API stats: {api_logger.get_api_stats()}")
    
    logger.info("Logging demo completed")