from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_FLASH_API_KEY"))

# Wrapper class around the Gemini model
class GeminiClientWrapper:
    def __init__(self, model_name):
        self.model = genai.GenerativeModel(model_name)
        # Add model_info attribute (adjust 'vision' as needed)
        self.model_info = {"vision": False}  

    def generate_content(self, prompt):
        return self.model.generate_content(prompt)

# Instantiate the wrapped client
model_client = GeminiClientWrapper("models/gemini-1.5-flash")

app = Flask(__name__)

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
    print("Prompt received by proxy:\n", prompt)

    try:
        response = model_client.generate_content(prompt)
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
