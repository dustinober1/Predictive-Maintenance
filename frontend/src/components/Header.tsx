import React from 'react'
import './Header.css'

interface HeaderProps {
  isStreaming: boolean
  onToggleStreaming: () => void
}

const Header: React.FC<HeaderProps> = ({ isStreaming, onToggleStreaming }) => {
  return (
    <header className="header">
      <div className="header-left">
        <h1 className="logo">
          <span className="logo-icon">⚙️</span>
          Insight
        </h1>
        <span className="subtitle">Predictive Maintenance Dashboard</span>
      </div>
      
      <div className="header-center">
        <div className={`status-indicator ${isStreaming ? 'streaming' : 'stopped'}`}>
          <span className="status-dot"></span>
          <span className="status-text">
            {isStreaming ? 'Live Data Streaming' : 'Data Stream Stopped'}
          </span>
        </div>
      </div>
      
      <div className="header-right">
        <button 
          className={`stream-toggle ${isStreaming ? 'stop' : 'start'}`}
          onClick={onToggleStreaming}
        >
          {isStreaming ? 'Stop Stream' : 'Start Stream'}
        </button>
        
        <div className="user-info">
          <span className="user-name">System Admin</span>
          <div className="user-avatar">A</div>
        </div>
      </div>
    </header>
  )
}

export default Header