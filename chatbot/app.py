from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model import train_model
import torch
from pyngrok import ngrok
import os

import uvicorn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Sample training data
training_data = [
    {"input_ids": tokenizer("What is the key concept of Rich Dad Poor Dad?", return_tensors="pt")["input_ids"].squeeze(), 
     "labels": tokenizer("The key concept is financial literacy.", return_tensors="pt")["input_ids"].squeeze()},
]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Train the model only if there is training data
if training_data:
    model = train_model(tokenizer=tokenizer, training_data=training_data, output_dir="./fin_intel_model")
    model.to(device)


@app.get("/test")
async def test():
    return "FastAPI is running locally!"

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_question = data.get("question")
        if not user_question:
            return JSONResponse(content={"error": "Missing 'question' in the request"}, status_code=400)
        inputs = tokenizer(user_question, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        return JSONResponse(content={"answer": answer}, status_code=200)

    except KeyError:
        return JSONResponse(content={"error": "Missing 'question' in the request"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# You can use uvicorn to run the FastAPI app (install uvicorn first: pip install uvicorn)
if __name__ == "__main__":
    os.environ['NGROK_AUTHTOKEN'] = '2DtAW5mT5k6MGbe3lJSMDDS0wyy_2gszvhmqUfDMdj3gVRgtv'

    uvicorn.run(app, host="0.0.0.0", port=5508)
    
    public_url = ngrok.connect(5508)
    print(f" * ngrok tunnel: {public_url}")