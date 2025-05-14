import google.generativeai as genai

def list_available_models():
    # Configure the API key
    genai.configure(api_key="AIzaSyDBJp_ZXsvzTVpClNNOD0i5D84tRLTBsfU")

    try:
        # Fetch the list of available models
        models = genai.list_models()

        # Print details of each model
        print("List of available models for v1beta:")
        for model in models:
            print(f"Model ID: {model.name}")
            print(f"Description: {model.description}")
            print(f"Base Model: {model.base_model}")  # Check if base_model exists
            print(f"Attributes: {model.attributes}")  # Check if attributes exist
            print("----")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to list models
list_available_models()