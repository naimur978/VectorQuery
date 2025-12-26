import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables")
    exit(1)

genai.configure(api_key=api_key)

print("Checking available Gemini models with your API key...")
print("=" * 60)

try:
    # List all available models
    models = genai.list_models()
    
    print(f"Found {len(list(models))} total models")
    print("\nAvailable models that support generateContent:")
    print("-" * 50)
    
    # Reset iterator since we consumed it with len()
    models = genai.list_models()
    
    content_models = []
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            content_models.append(model.name)
            print(f"✅ {model.name}")
            print(f"   Display Name: {model.display_name}")
            print(f"   Description: {model.description}")
            print(f"   Supported methods: {model.supported_generation_methods}")
            print()
    
    if content_models:
        print(f"\n🎯 Recommended models for your chat application:")
        print("-" * 50)
        for model_name in content_models:
            # Remove the 'models/' prefix for the model parameter
            clean_name = model_name.replace('models/', '')
            print(f"   Use: model=\"{clean_name}\"")
        
        print(f"\n🧪 Testing the first available model: {content_models[0].replace('models/', '')}")
        
        # Test the first available model
        test_model_name = content_models[0].replace('models/', '')
        model = genai.GenerativeModel(test_model_name)
        
        response = model.generate_content("Hello, this is a test message. Please respond briefly.")
        print(f"✅ Test successful! Response: {response.text}")
        
    else:
        print("❌ No models found that support generateContent")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nThis might be due to:")
    print("1. Invalid API key")
    print("2. API key doesn't have proper permissions")
    print("3. Network connectivity issues")
    print("4. API quota exceeded")