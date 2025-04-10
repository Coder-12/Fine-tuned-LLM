from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fine_tuning.model import LLMModel

app = FastAPI(title="LLM Fine-Tuning API", version="0.1")

# Initialize the fine-tuned model at startup
model_instance = LLMModel("config.yaml")

class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 100

class InferenceResponse(BaseModel):
    output: str

@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    try:
        generated_text = model_instance.generate(request.prompt, request.max_length)
        return InferenceResponse(output=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the LLM Fine-Tuning API"}
