import nltk
import string
from nltk.corpus import stopwords
from transformers import pipeline

# to remove punctuation, special characters, stop words...
def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove special characters
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    # Join back into a string
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

def generate_questions(text):
    question_generator = pipeline("question-generation", model="google/flan-t5-base")
    questions = question_generator(text, max_length=50)
    return [question['question'] for question in questions] 