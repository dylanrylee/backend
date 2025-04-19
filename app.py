from flask import Flask, jsonify, request
import requests
from flask_cors import CORS
import os
import re
from dotenv import load_dotenv

# === Load environment variables (e.g., Hugging Face API key) ===
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for React frontend

HF_API_KEY = os.getenv("HF_API_KEY")  # Load API key from .env

# === GET /api/buildings ===
# Returns a limited batch of Calgary building data from public open data API
@app.route('/api/buildings', methods=['GET'])
def get_buildings():
    url = "https://data.calgary.ca/resource/cchr-krqg.json?$limit=100"  # Limit to first 100 buildings for performance
    response = requests.get(url)
    return jsonify(response.json())  # Return raw JSON list to frontend

# === POST /api/query ===
# Accepts a natural language query and converts it to a filter using Hugging Face's LLM
@app.route("/api/query", methods=["POST"])
def query_llm():
    # Ensure Hugging Face API key is loaded
    if not HF_API_KEY:
        return jsonify({"error": "Missing Hugging Face API key"}), 400

    data = request.get_json()
    user_input = data.get("query", "")

    # Construct prompt that instructs the model to return only structured JSON
    prompt = f"""
    Convert this natural language request into a JSON filter.

    Input: "{user_input}"

    Output format:
    {{
    "attribute": "height",
    "operator": ">",
    "value": 100
    }}

    Return only a valid JSON object as shown.
    """

    try:
        # Send prompt to Hugging Face Inference API
        response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-large",
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"inputs": prompt}
        )

        result = response.json()
        print("🧠 Raw LLM response:", result)

        # Handle possible Hugging Face API error
        if isinstance(result, dict) and "error" in result:
            return jsonify({"error": result["error"]}), 502

        # Extract generated text and attempt to locate valid JSON
        generated = result[0]["generated_text"]
        match = re.search(r"\{.*\}", generated, re.DOTALL)

        if match:
            clean_json = match.group(0)
            print("✅ Parsed filter JSON:", clean_json)
            return jsonify({"filter": clean_json})
        else:
            print("⚠️ No JSON object found in LLM output.")
            return jsonify({"error": "No JSON object found in LLM output"}), 500

    except Exception as e:
        print("❌ Exception querying LLM:", str(e))
        return jsonify({"error": str(e)}), 500

# === Run the Flask server ===
if __name__ == '__main__':
    app.run(debug=True)
