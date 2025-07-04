from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import os
import logging

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route("/style.css")
def serve_css():
    return send_from_directory('.', 'style.css')

@app.route("/progress", methods=["GET"])
def progress():
    try:
        with open("outputs/results/matcher_progress.txt", "r") as f:
            progress = f.read()
        return jsonify({"progress": progress})
    except:
        return jsonify({"progress": "0"})

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    task = data.get("task")
    path = data.get("path", "").strip()

    if not path:
        path = "data/Task_A/val" if task == "taskA" else "data/Task_B/val"

    with open("test_path.txt", "w") as f:
        f.write(path)

    print(f"[INFO] Running evaluation for: {task} on {path}")

    if task == "taskA":
        result = subprocess.run(
            ["python", "evaluate_task_a.py", "--val_dir", path],
            capture_output=True,
            text=True
            )
        output = result.stdout


    elif task == "taskB":
        with open("outputs/results/matcher_progress.txt", "w") as f:
            f.write("0")
        try:
            matcher_process = subprocess.run(
                ["python", "matcher.py", "--test_dir", path],
                capture_output=True,
                text=True,
                check=True  
            )
            print("[DEBUG] Matcher Output:\n", matcher_process.stdout)

            # Only run evaluation if matcher succeeded
            result = subprocess.run(["python", "evaluate_matcher_results.py"], capture_output=True, text=True)
            output = result.stdout

        except subprocess.CalledProcessError as e:
            error_msg = f" Task B failed: {e.stderr or e.stdout or str(e)}"
            print("[ERROR]", error_msg)
            return jsonify({"output": error_msg})


    print("[DEBUG] Final Output:\n", output)
    return jsonify({"output": output})

if __name__ == "__main__":
    import logging
    from werkzeug.serving import WSGIRequestHandler

    # Suppress per-request logs only 
    class NoRequestLogHandler(WSGIRequestHandler):
        def log_request(self, code='-', size='-'):
            pass  # Disable logging for each HTTP request

    app.run(debug=False, use_reloader=False, request_handler=NoRequestLogHandler)