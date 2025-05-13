import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == '__main__':
    # Get API key from environment variable
    api_key = os.getenv('GENAI_API_KEY')
    if not api_key:
        raise ValueError("GENAI_API_KEY environment variable is not set")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = 'What is the airspeed velocity of an unladen swallow?'
    response = model.generate_content(prompt)

    print(response.text)
