from flask import Flask, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_FLASH_API_KEY"))

app = Flask(__name__)

model = genai.GenerativeModel("models/gemini-1.5-flash")

def planner_agent():
    prompt = "Plan a 3 day trip to Nepal."
    return model.generate_content(prompt).text

def local_agent():
    prompt = "What are some authentic local experiences, offbeat places, or cultural activities a tourist should try in Nepal?"
    return model.generate_content(prompt).text

def language_agent():
    prompt = "What language or communication tips should travelers know when visiting Nepal? Include greetings, customs, and common phrases."
    return model.generate_content(prompt).text

def summary_agent(itinerary, local_info, language_tips):
    prompt = f"""
You are a travel planner. Combine and summarize the following information into a complete travel plan:
---
Itinerary:
{itinerary}

Local Experiences:
{local_info}

Language Tips:
{language_tips}

Return a well-integrated and engaging travel guide for a 3-day trip to Nepal. Conclude with the word TERMINATE.
"""
    return model.generate_content(prompt).text

@app.route("/travel/plan", methods=["GET"])
def generate_plan():
    try:
        itinerary = planner_agent()
        local_info = local_agent()
        language_tips = language_agent()
        final_plan = summary_agent(itinerary, local_info, language_tips)

        return jsonify({
            "itinerary": itinerary,
            "local_experiences": local_info,
            "language_tips": language_tips,
            "final_plan": final_plan
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8000)
