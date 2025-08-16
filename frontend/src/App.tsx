import React, { useState, useEffect } from 'react'
import Dashboard from './components/Dashboard'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import './App.css'

const API_BASE_URL = 'http://localhost:8000'

interface DashboardData {
  summary: {
    total_engines: number
    active_engines: number
    critical_alerts: number
    avg_predicted_rul: number
    last_update: number
  }
  recent_data: {
    sensor_readings: number
    alerts: number
    predictions: number
  }
}

function App() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  // Fetch dashboard data
  const fetchDashboardData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/dashboard/summary`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setDashboardData(data)
      setError(null)
    } catch (err) {
      console.error('Error fetching dashboard data:', err)
      setError('Failed to connect to API server')
    } finally {
      setIsLoading(false)
    }
  }

  // Start/stop streaming
  const toggleStreaming = async () => {
    try {
      if (isStreaming) {
        const response = await fetch(`${API_BASE_URL}/streaming/stop`, {
          method: 'POST',
        })
        if (response.ok) {
          setIsStreaming(false)
        }
      } else {
        const response = await fetch(`${API_BASE_URL}/streaming/start`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            engines: [1, 2, 3],
            speed: 2.0
          })
        })
        if (response.ok) {
          setIsStreaming(true)
        }
      }
    } catch (err) {
      console.error('Error toggling streaming:', err)
      setError('Failed to toggle streaming')
    }
  }

  // Fetch data on component mount and set up polling
  useEffect(() => {
    fetchDashboardData()
    
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchDashboardData, 5000)
    
    return () => clearInterval(interval)
  }, [])

  if (isLoading) {
    return (
      <div className="app">
        <div className="loading">
          <h2>Loading Insight Dashboard...</h2>
          <p>Connecting to predictive maintenance system...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app">
        <div className="error">
          <h2>Connection Error</h2>
          <p>{error}</p>
          <button onClick={fetchDashboardData}>Retry Connection</button>
          <div className="error-details">
            <p>Make sure the API server is running:</p>
            <code>python3 backend/app/simple_server.py</code>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <Header 
        isStreaming={isStreaming}
        onToggleStreaming={toggleStreaming}
      />
      <div className="app-body">
        <Sidebar />
        <main className="main-content">
          <Dashboard data={dashboardData} />
        </main>
      </div>
    </div>
  )
}

export default App