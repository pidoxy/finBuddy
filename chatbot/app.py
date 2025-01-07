from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model import train_model
import torch
from pyngrok import ngrok
import os
import logging
import gc

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # Disable MPS
import uvicorn
from sklearn.model_selection import train_test_split

device = torch.device("cpu") #Force model to run on CPU
torch.set_default_device(device)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], #Allow only requests from localhost:3000
    allow_credentials=True,
    allow_methods=["*"], # * to all all methods
    allow_headers=["*"], # * to allow all headers
)

# tokenizer = AutoTokenizer.from_pretrained("t5-small")
# google/flan-t5-large
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# Sample training data
training_data = [
    {"input_ids": tokenizer("What is the key concept of Rich Dad Poor Dad?", return_tensors="pt")["input_ids"].squeeze(), 
     "labels": tokenizer("The key concept is financial literacy.", return_tensors="pt")["input_ids"].squeeze()  },
    
     {"input_ids": tokenizer("What is the importance of understanding money?", return_tensors="pt")["input_ids"].squeeze(),
        "labels": tokenizer("It is crucial for achieving financial independence and avoiding financial struggles.", return_tensors="pt")["input_ids"].squeeze()},
     
    {"input_ids": tokenizer("How does 'Rich Dad' define assets?", return_tensors="pt")["input_ids"].squeeze(),
    "labels": tokenizer("Assets are things that put money in your pocket.", return_tensors="pt")["input_ids"].squeeze()},
    
    {"input_ids": tokenizer("What is the 'Rat Race' according to Kiyosaki?", return_tensors="pt")["input_ids"].squeeze(),
    "labels": tokenizer("The 'Rat Race' is the cycle of working for money without achieving financial independence.", return_tensors="pt")["input_ids"].squeeze()},
        
    {"input_ids": tokenizer("Why is financial education important?", return_tensors="pt")["input_ids"].squeeze(),
    "labels": tokenizer("Financial education empowers individuals to build wealth and achieve financial freedom.", return_tensors="pt")["input_ids"].squeeze()},
]

model_path = "./fin_intel_model"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Train the model only if it doesn't exist yet
if not os.path.exists(model_path):
    print("Training the model now...")
    model = train_model(tokenizer=tokenizer, training_data=training_data,  output_dir=model_path)
else:
    print("Loading the model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    #model.to(device) # No need to move the model explicitly to the device

if len(training_data) > 1:
        train_data, eval_data = train_test_split(training_data, test_size=0.2, random_state=42)
else:
        train_data = training_data
        eval_data = []

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
        logging.debug(f"User question: {user_question}")  # Log user question
        with torch.no_grad():
            inputs = tokenizer(user_question, return_tensors="pt")
            logging.debug(f"Input IDs: {inputs}")
            output = model.generate(**inputs)
            logging.debug(f"Generated Output: {output}")
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
        torch.mps.empty_cache()
        gc.collect()
        return JSONResponse(content={"answer": answer}, status_code=200)

    except KeyError:
        return JSONResponse(content={"error": "Missing 'question' in the request"}, status_code=400)
    except Exception as e:
        logging.error(f"Exception during chat processing: {e}", exc_info=True)  # Add exc_info for detail
        return JSONResponse(content={"error": str(e)}, status_code=500)

# You can use uvicorn to run the FastAPI app (install uvicorn first: pip install uvicorn)
if __name__ == "__main__":
    os.environ['NGROK_AUTHTOKEN'] = 'your_ngrok_token' #Replace your token

    uvicorn.run(app, host="0.0.0.0", port=5508)
    
    public_url = ngrok.connect(5508)
    print(f" * ngrok tunnel: {public_url}")