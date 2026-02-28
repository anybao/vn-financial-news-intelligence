from prometheus_client import Counter, Histogram
import time
from fastapi import Request

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    "api_request_count", 
    "Total API request count", 
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", 
    "API request latency in seconds",
    ["method", "endpoint"]
)

async def prometheus_middleware(request: Request, call_next):
    """FastAPI middleware to track request counts and latencies."""
    method = request.method
    endpoint = request.url.path
    
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    status_code = response.status_code
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(process_time)
    
    return response
