"""
Flask Web Application - Stage 6B
Minimal browser-based interface for the News Analyzer pipeline.

Routes:
    GET  /   -> Home page with search form
    POST /   -> Run pipeline and display results
"""

import json
from flask import Flask, render_template, request

from pipeline import run_pipeline

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """Home page: search form and results display."""
    result = None
    error = None
    topic = ""

    if request.method == "POST":
        topic = request.form.get("topic", "").strip()

        if not topic:
            error = "Please enter a news topic or claim."
        else:
            try:
                result = run_pipeline(topic)

                if result["status"] == "error":
                    error = result.get("error", "An unknown error occurred.")
                    # Still pass result so partial data can be shown
            except Exception as e:
                error = f"Pipeline error: {e}"

    return render_template(
        "index.html",
        topic=topic,
        result=result,
        error=error,
        result_json=json.dumps(result, indent=2, default=str) if result else None,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
