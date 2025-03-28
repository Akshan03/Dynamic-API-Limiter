import os
import sys
import argparse
import subprocess
import time
import signal
import logging
from typing import List, Optional
import webbrowser
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dynamic_rate_limiter")

# Global variables for process management
api_process = None
dashboard_process = None
processes = []

def setup_environment():
    """Set up environment variables and directories."""
    # Ensure data directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    
    # Add project root to Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    logger.info("Environment setup complete")

def start_api(port: int = 8000, reload: bool = False):
    """Start the FastAPI application."""
    global api_process
    
    logger.info(f"Starting API server on port {port}")
    
    # Command to run the API
    cmd = [
        "uvicorn", 
        "app.main:app", 
        "--host", "0.0.0.0", 
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    # Start the API process
    api_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    processes.append(api_process)
    
    # Log the API output in a separate thread
    def log_output():
        for line in api_process.stdout:
            logger.info(f"API: {line.strip()}")
    
    threading.Thread(target=log_output, daemon=True).start()
    
    logger.info(f"API server started on http://localhost:{port}")
    return api_process

def start_dashboard(port: int = 8501):
    """Start the Streamlit dashboard."""
    global dashboard_process
    
    logger.info(f"Starting dashboard on port {port}")
    
    # Command to run the dashboard
    cmd = [
        "streamlit", 
        "run", 
        "dashboard/app.py",
        "--server.port", str(port),
        "--server.headless", "true"
    ]
    
    # Start the dashboard process
    dashboard_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    processes.append(dashboard_process)
    
    # Log the dashboard output in a separate thread
    def log_output():
        for line in dashboard_process.stdout:
            logger.info(f"Dashboard: {line.strip()}")
    
    threading.Thread(target=log_output, daemon=True).start()
    
    logger.info(f"Dashboard started on http://localhost:{port}")
    return dashboard_process

def open_browser(api_port: int, dashboard_port: int):
    """Open browser tabs for API and dashboard."""
    # Wait for services to start
    time.sleep(2)
    
    # Open API docs in browser
    api_url = f"http://localhost:{api_port}/docs"
    logger.info(f"Opening API documentation: {api_url}")
    webbrowser.open_new(api_url)
    
    # Open dashboard in browser
    dashboard_url = f"http://localhost:{dashboard_port}"
    logger.info(f"Opening dashboard: {dashboard_url}")
    webbrowser.open_new_tab(dashboard_url)

def handle_shutdown(signum, frame):
    """Handle graceful shutdown on signal."""
    logger.info("Shutdown signal received, stopping services...")
    
    for process in processes:
        if process and process.poll() is None:
            process.terminate()
    
    logger.info("All services stopped")
    sys.exit(0)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dynamic AI Rate Limiter")
    
    parser.add_argument(
        "--api-only", 
        action="store_true",
        help="Run only the API server"
    )
    
    parser.add_argument(
        "--dashboard-only", 
        action="store_true",
        help="Run only the dashboard"
    )
    
    parser.add_argument(
        "--api-port", 
        type=int, 
        default=8000,
        help="Port for the API server (default: 8000)"
    )
    
    parser.add_argument(
        "--dashboard-port", 
        type=int, 
        default=8501,
        help="Port for the dashboard (default: 8501)"
    )
    
    parser.add_argument(
        "--no-browser", 
        action="store_true",
        help="Don't open browser windows automatically"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for API development"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup environment
    setup_environment()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    try:
        if args.api_only:
            # Start only the API
            start_api(port=args.api_port, reload=args.reload)
            
            if not args.no_browser:
                webbrowser.open_new(f"http://localhost:{args.api_port}/docs")
            
        elif args.dashboard_only:
            # Start only the dashboard
            start_dashboard(port=args.dashboard_port)
            
            if not args.no_browser:
                webbrowser.open_new(f"http://localhost:{args.dashboard_port}")
            
        else:
            # Start both API and dashboard
            start_api(port=args.api_port, reload=args.reload)
            start_dashboard(port=args.dashboard_port)
            
            if not args.no_browser:
                open_browser(args.api_port, args.dashboard_port)
        
        logger.info("All services started. Press Ctrl+C to stop.")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
            # Check if any process has terminated unexpectedly
            for process in processes:
                if process and process.poll() is not None:
                    exit_code = process.poll()
                    if exit_code != 0:
                        logger.error(f"Process terminated with exit code {exit_code}")
                        handle_shutdown(None, None)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        handle_shutdown(None, None)
    
    except Exception as e:
        logger.error(f"Error running application: {e}", exc_info=True)
        handle_shutdown(None, None)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
