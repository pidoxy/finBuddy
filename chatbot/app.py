from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model import train_model
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
from pyngrok import ngrok
import gc
from sklearn.model_selection import train_test_split # Add this

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
app = FastAPI()

app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
    )

tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Sample training data
training_data_complete = [
    {"input_ids": tokenizer("What is the key concept of Rich Dad Poor Dad?", return_tensors="pt")["input_ids"].squeeze(), 
     "labels": tokenizer("The key concept is financial literacy.", return_tensors="pt")["input_ids"].squeeze()},
        {"input_ids": tokenizer("What is the importance of understanding money?", return_tensors="pt")["input_ids"].squeeze(),
        "labels": tokenizer("It is crucial for achieving financial independence and avoiding financial struggles.", return_tensors="pt")["input_ids"].squeeze()},
]

# Split the training data into training and evaluation sets # Add this
if len(training_data_complete) > 1:
    train_data, eval_data = train_test_split(training_data_complete, test_size=0.2, random_state=42)
else:
    train_data = training_data_complete
    eval_data = []  # Or you can use `None` if not needed.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Train the model only if there is training data
if train_data:
    model = train_model(tokenizer=tokenizer, training_data=train_data, eval_data = eval_data, output_dir="./fin_intel_model") # Change this
    #model.to(device)  #Remove this as model is already on CPU.

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

        logger.info(f"Received question: {user_question}")

        with torch.no_grad():
            inputs = tokenizer(user_question, return_tensors="pt")
            logger.info(f"Tokenized input: {inputs}")
            output = model.generate(**inputs)
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            logger.info(f"Generated answer: {answer}")

        gc.collect()  # Explicit garbage collection
        return JSONResponse(content={"answer": answer}, status_code=200)

    except KeyError:
        logger.error("KeyError: Missing 'question' in the request.")
        return JSONResponse(content={"error": "Missing 'question' in the request"}, status_code=400)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# You can use uvicorn to run the FastAPI app (install uvicorn first: pip install uvicorn)
if __name__ == "__main__":
    # Set your Ngrok authtoken
    os.environ['NGROK_AUTHTOKEN'] = 'your_ngrok_token' #Replace 'your_ngrok_token' with your actual token.

    # Start ngrok tunnel
    public_url = ngrok.connect(5508)
    print(f" * ngrok tunnel: {public_url}")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5508)