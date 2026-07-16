import os
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

models = ['gemini-1.5-flash', 'gemini-2.5-flash', 'gemini-2.0-flash']

for m in models:
    try:
        response = client.models.generate_content(
            model=m,
            contents="say hi"
        )
        print(f"{m} success: {response.text.strip()}")
    except Exception as e:
        print(f"{m} failed: {type(e).__name__} - {str(e)[:100]}")
