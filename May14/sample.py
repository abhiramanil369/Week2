import google.generativeai as genai

def list_available_models():
    genai.configure(api_key="AIzaSyDBJp_ZXsvzTVpClNNOD0i5D84tRLTBsfU")

    try:
        models = genai.list_models()

        print("List of available models for v1beta:")
        for model in models:
            print(f"Model ID: {model.name}")
            print(f"Description: {model.description}")
            print(f"Base Model: {model.base_model}")  
            print(f"Attributes: {model.attributes}")  
            print("----")
    except Exception as e:
        print(f"An error occurred: {e}")

list_available_models()