import os
import sys
import http.server
import socketserver
import webbrowser

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRATCH_JSON = r"fly_model_web.json"

class FlyVisualizerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        # Route model requests to scratch file if not in base dir
        if self.path == "/fly_model_web.json" or self.path == "/data/malecns_circuit.json":
            if os.path.exists(SCRATCH_JSON):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                with open(SCRATCH_JSON, "rb") as f:
                    content = f.read()
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/web_visualizer.html"

        super().do_GET()

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), FlyVisualizerHandler) as httpd:
        url = f"http://localhost:{PORT}/web_visualizer.html"
        print("=" * 65)
        print("  FLYSWEEPER: 3D CONNECTOME WEB VISUALIZER")
        print("=" * 65)
        print(f"Server active at: {url}")
        print("Opening in your web browser...")
        print("Press Ctrl+C to stop the server.")
        print("-" * 65)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    run_server()
