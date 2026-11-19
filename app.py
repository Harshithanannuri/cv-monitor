"""Launch the GUI:   python app.py            ->  http://127.0.0.1:8000
   options:          --port 8080  --host 0.0.0.0 (share on your LAN)  --no-browser"""
import argparse
import os
import threading
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

from gui.server import create_app   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    app = create_app()
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{a.host if a.host != '0.0.0.0' else '127.0.0.1'}:{a.port}")).start()
    print(f"CV Monitor GUI  ->  http://{a.host}:{a.port}   (Ctrl+C to quit)")
    app.run(host=a.host, port=a.port, threaded=True, debug=False)
