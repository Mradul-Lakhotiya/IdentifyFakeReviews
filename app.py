from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the model and tokenizer
model_name = "bert-base-uncased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
try:
    model.load_state_dict(torch.load("nlp_model.pth", map_location=device))  # Load trained model
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file 'nlp_model.pth' not found!")
    exit(1)  # Stop execution if model is missing

model.to(device)
model.eval()

# Define the prediction function
def predict(text):
    inputs = tokenizer(
        text, 
        padding="max_length",  # Ensures consistent length
        truncation=True, 
        max_length=512,  # BERT's max length
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    return "Positive" if prediction == 1 else "Negative"

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    
    if not data or "text" not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    print(text)
    prediction = predict(text)
    print(f"Prediction: {prediction}")
    
    return jsonify({'prediction': prediction})

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
