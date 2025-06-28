from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route("/style.css")
def serve_css():
    return send_from_directory('.', 'style.css')

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    task = data.get("task")
    path = data.get("path", "").strip()

    if not path:
        # Fallback default paths
        path = "data/Task_A/val" if task == "taskA" else "data/Task_B/val"

    with open("test_path.txt", "w") as f:
        f.write(path)

    print(f"[INFO] Running evaluation for: {task} on {path}")

    if task == "taskA":
        result = subprocess.run(["python", "evaluate_task_a.py"], capture_output=True, text=True)
    else:
        result = subprocess.run(["python", "evaluate_matcher_results.py"], capture_output=True, text=True)

    print("[DEBUG] Raw Output:\n", result.stdout)

    return jsonify({"output": result.stdout})

if __name__ == "__main__":
    app.run(debug=True)
