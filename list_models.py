from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
  api_key=os.environ.get("RESPONSE_API_KEY"),
  base_url=os.environ.get("RESPONSE_API_BASE_URL"),
)

models = client.models.list()
for model in models:
  print(model.id)