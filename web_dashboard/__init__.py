"""
Web Dashboard for PomegranteMuse
Provides web-based interface for managing projects and monitoring progress
"""

from .dashboard_server import (
    DashboardServer,
    create_dashboard_app,
    start_dashboard_server
)

from .api_routes import (
    APIRoutes,
    register_api_routes
)

from .websocket_handler import (
    WebSocketHandler,
    setup_websocket_routes
)

__all__ = [
    "DashboardServer",
    "create_dashboard_app", 
    "start_dashboard_server",
    "APIRoutes",
    "register_api_routes",
    "WebSocketHandler",
    "setup_websocket_routes"
]