"""
Web Chatbot Launcher for Pad Krapao RAG System.

Run this script to start the local web server and open the Chatbot in your default browser.
Command: python run_web.py
"""

import os
import sys
import webbrowser
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from app import app, init_retriever

def open_browser():
    time.sleep(1.5)
    print("[RAG-Web Launcher] Opening http://127.0.0.1:5000 in browser...")
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    init_retriever()
    
    # Start browser in a background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    print("[RAG-Web Launcher] Starting Flask web server at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server.")
    app.run(host="127.0.0.1", port=5000, debug=False)
