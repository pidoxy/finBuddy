# from flask import Flask, request, jsonify
# import os
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Import the required classes
# from model import train_model  # Assuming train_model is a custom function in model.py

# app = Flask(__name__)

# # Initialize the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("t5-small")  # Adjust the model name as per your use case
# # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# # Sample training data
# training_data = [
#     {"input_ids": tokenizer("What is the key concept of Rich Dad Poor Dad?", return_tensors="pt")["input_ids"].squeeze(), 
#      "labels": tokenizer("The key concept is financial literacy.", return_tensors="pt")["input_ids"].squeeze()},
# ]

# # Train the model only if there is training data
# if training_data:
#     model = train_model( tokenizer=tokenizer, training_data=training_data, output_dir="./fin_intel_model")

# @app.route('/test', methods=['GET'])
# def test():
#     return "Server is running!", 200

# @app.route('/chat', methods=['POST'])
# def chat():
#     """Handles chat requests."""
#     try:
#         user_question = request.json['question']
#         inputs = tokenizer(user_question, return_tensors="pt")
#         output = model.generate(**inputs)
#         answer = tokenizer.decode(output[0], skip_special_tokens=True)
#         return jsonify({'answer': answer})
#     except KeyError:
#         return jsonify({'error': 'Missing "question" in the request'}), 400
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     os.system('ngrok http 5508')
#     os.environ['NGROK_AUTHTOKEN'] = '2DtAW5mT5k6MGbe3lJSMDDS0wyy_2gszvhmqUfDMdj3gVRgtv' 
#     app.run(debug=False, port=5508)

from flask import Flask, request, jsonify
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Import the required classes
from model import train_model

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("t5-small")  # Adjust the model name as per your use case

# Sample training data
training_data = [
    {"input_ids": tokenizer("What is the key concept of Rich Dad Poor Dad?", return_tensors="pt")["input_ids"].squeeze(), 
     "labels": tokenizer("The key concept is financial literacy.", return_tensors="pt")["input_ids"].squeeze()},
]

# Train the model only if there is training data
if training_data:
    model = train_model( tokenizer=tokenizer, training_data=training_data, output_dir="./fin_intel_model")

# Define a test route
@app.route('/test', methods=['GET'])
def test():
    return "Flask is running locally!"

# Define your main route
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_question = request.json['question']
        inputs = tokenizer(user_question, return_tensors="pt").to(device)
        model.to(device)
        output = model.generate(**inputs)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({'answer': answer})

    except KeyError:
        return jsonify({'error': 'Missing "question" in the request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app
    os.system('ngrok http 5508')
    app.run(debug=True, port=5508)
