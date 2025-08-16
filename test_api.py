import requests
import time

# Test basic API endpoints
base_url = "http://localhost:8000"

def test_api():
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        
        # Test engines endpoint
        response = requests.get(f"{base_url}/engines")
        print(f"Engines: {response.status_code}")
        
        # Test dashboard summary
        response = requests.get(f"{base_url}/dashboard/summary")
        print(f"Dashboard: {response.status_code}")
        
    except requests.exceptions.ConnectionError:
        print("API server is not running. Start it with: uvicorn backend.app.main:app --reload")
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == "__main__":
    test_api()