import asyncio
from openai import AsyncOpenAI
import os 
from dotenv import load_dotenv
from pydantic import ValidationError
load_dotenv()

class AIProvider():
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ['GEMINI_API_KEY'],
            base_url="https://generativelanguage.googleapis.com/v1beta/"
        )
    def get_client(self):
        return self.client

provider = AIProvider()

def get_ai_client():
    return provider.get_client()

async def call_gemini(user_message, system_message="", response_schema = None, model="gemini-2.5-flash-lite", temperature = 0.65):
    client = get_ai_client()
    max_attempts = 3
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    extra_body = {
        'extra_body': {
            "google": {
                "thinking_config": {
                    "thinking_budget": "-1",
                    "include_thoughts": False
                }
            }
        }
    }
    
    for attempt in range(max_attempts):
        try:
            if response_schema:
                response = await client.chat.completions.parse(
                    model = model,
                    messages = messages,
                    temperature = temperature,
                    extra_body = extra_body,
                    response_format = response_schema
                )  
                response_schema.validate(response.choices[0].message.parsed)
                return response.choices[0].message.parsed
            else:
                response = await client.chat.completions.create(
                    model = model,
                    messages = messages,
                    temperature = temperature,
                    extra_body = extra_body
                )   
                if response.choices[0].message.content == None:
                    raise Exception("Response content is None, retrying...")
                return response.choices[0].message.content
        
        except ValidationError as e:
            print(f"Pydantic validation failed on attempt {attempt + 1}/{max_attempts}. The API returned a malformed object. Error: {e}")
        
        except Exception as e:
            print(f"API call failed on attempt {attempt + 1}/{max_attempts}. Error: {type(e).__name__}: {e}")
        
        if attempt < max_attempts - 1:
            delay = 0.5 * (2 ** attempt)
            print(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
        else:
            print(f"Calling Gemini API failed after {max_attempts} attempts.")
            raise Exception("Gemini API call failed after multiple attempts.")