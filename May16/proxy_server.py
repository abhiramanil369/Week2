from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_FLASH_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")

app = Flask(__name__)

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
    print("Prompt received by proxy:\n", prompt)

    try:
        response = model.generate_content(prompt)
        return jsonify({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response.text
                }
            }]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8000)
