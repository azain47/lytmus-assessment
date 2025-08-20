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
    for attempt in range(max_attempts):
        try:
            if response_schema:
                response = await client.chat.completions.parse(
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
                    temperature = temperature,
                    extra_body = {
                        'extra_body': {
                            "google": {
                                "thinking_config": {
                                    "thinking_budget": "-1",
                                    "include_thoughts": False
                                }
                            }
                        }
                    },
                    response_format = response_schema
                )  
                return response.choices[0].message.parsed  
            else:
                response = await client.chat.completions.create(
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
                    temperature = temperature,
                    extra_body = {
                        'extra_body': {
                            "google": {
                                "thinking_config": {
                                    "thinking_budget": "-1",
                                    "include_thoughts": False
                                }
                            }
                        }
                    },
                )   
                return response.choices[0].message.content 
        except ValidationError as ve:
            if attempt == max_attempts - 1:
                raise Exception(f"Calling API failed after max attempts. Error: {ve}")  
            attempt += 1
            delay = 0.5 * (2 ** attempt)
            print(f"Gemini Failed to generate valid pydantic model, retrying after {delay} seconds...")
            await asyncio.sleep(delay)        
            continue