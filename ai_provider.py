from openai import OpenAI
import os 
from dotenv import load_dotenv
load_dotenv()

class AIProvider():
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ['GEMINI_API_KEY'],
            base_url="https://generativelanguage.googleapis.com/v1beta/"
        )
    def get_client(self):
        return self.client

provider = AIProvider()

def get_ai_client():
    return provider.get_client()

def call_gemini(system_message, user_message, response_schema = None, model="gemini-2.5-flash-lite"):
    
    client = get_ai_client()
    
    if response_schema:
        response = client.chat.completions.parse(
            model = model,
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            temperature = 0.65,
            extra_body={
                
            },
            response_format = response_schema
        ).choices[0].message.parsed    
        return response
    else:
        response = client.chat.completions.create(
            model = model,
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            temperature = 0.65,
            extra_body={
                
            }
        ).choices[0].message.content    
        return response
