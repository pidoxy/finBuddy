import requests
import json

def get_response_from_gemini(user_input):
  # Replace with your Gemini API endpoint and token
  api_endpoint = 'https://api.example.com/v1/models/gemini'
  api_token = 'YOUR_API_TOKEN'

  headers = {'Authorization': f'Bearer {api_token}'}
  data = {'prompt': user_input}

  response = requests.post(api_endpoint, headers=headers, data=json.dumps(data))

  if response.status_code == 200:
    return response.json()['response']
  else:
    return "Oops! Something went wrong. Please try again later."

while True:
  user_input = input("You: ")
  if user_input == 'exit':
    break
  
  response = get_response_from_gemini(user_input)
  print(f"Chatbot: {response}")