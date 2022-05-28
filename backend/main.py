from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Prompt(BaseModel):
    prompt: str


model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qa-qg-hl")


@app.get("/gen")
async def faq(prompt: Prompt):
    res = model(prompt.prompt)
    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
