from fastapi import FastAPI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


class chatRequest(BaseModel):
    message: str


def get_bot_response(user_message):
    message = user_message.lower()

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "Your name is MindfulBot."
                    "You are a kind and supportive AI mental health assistant. "
                    "You are not a doctor or therapist, but you provide helpful, "
                    "empathetic, and safe emotional support. "
                    "If the user expresses thoughts of self-harm or crisis, "
                    "you must respond calmly with empathy and share crisis hotline resources. "
                    "Avoid giving medical or diagnostic advice."
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        model="llama-3.3-70b-versatile",
        stream=False,
    )

    return response.choices[0].message.content


@app.post("/chat")
async def chat(request: chatRequest):
    reply = get_bot_response(request.message)
    return {"reply": reply}
