from flask import Flask, render_template, request, jsonify, send_file
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch.nn.functional as F
import pandas as pd
import os
from datetime import datetime
import uuid

app = Flask(__name__)

# Global variables for model
model = None
tokenizer = None
device = None

# Excel file for feedback
FEEDBACK_FILE = 'model_feedback.xlsx'
RESULTS_FILE = 'prediction_results.xlsx'


def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer"""
    global model, tokenizer, device

    try:
        # Get the current directory and construct absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        adapter_path = os.path.join(current_dir, "..", "finetuning_pipeline", "marbertv2-lora-adapter")
        
        # Check if the adapter path exists
        if not os.path.exists(adapter_path):
            print(f"Error: Adapter path does not exist: {adapter_path}")
            print(f"Current directory: {current_dir}")
            print(f"Available directories: {os.listdir(current_dir)}")
            return False
        
        print(f"Loading model from: {adapter_path}")
        
        # Load the base model and tokenizer
        model_name = "UBC-NLP/MARBERTv2"
        print(f"Loading base model: {model_name}")
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Load tokenizer from the adapter directory
        print(f"Loading tokenizer from: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        # Load the LoRA adapter
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)

        # Set to evaluation mode
        model.eval()

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def predict_text(text):
    """Predict if text is offensive or not"""
    global model, tokenizer, device

    if model is None:
        return {"error": "Model not loaded"}

    try:
        # Tokenize the input
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        # Define labels
        labels = {0: "Not Offensive", 1: "Offensive"}

        # Generate unique ID for this prediction
        prediction_id = str(uuid.uuid4())

        result = {
            "id": prediction_id,
            "text": text,
            "prediction": labels[predicted_class],
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "Not Offensive": round(probabilities[0][0].item() * 100, 2),
                "Offensive": round(probabilities[0][1].item() * 100, 2)
            },
            "timestamp": datetime.now().isoformat(),
            "text_length": len(text),
            "word_count": len(text.split())
        }

        # Save result to Excel
        save_result_to_excel(result)

        return result

    except Exception as e:
        return {"error": f"Prediction error: {e}"}


def save_result_to_excel(result):
    """Save prediction result to Excel file"""
    try:
        # Prepare data for Excel
        data = {
            'ID': [result['id']],
            'Text': [result['text']],
            'Prediction': [result['prediction']],
            'Confidence': [result['confidence']],
            'Not_Offensive_Prob': [result['probabilities']['Not Offensive']],
            'Offensive_Prob': [result['probabilities']['Offensive']],
            'Text_Length': [result['text_length']],
            'Word_Count': [result['word_count']],
            'Timestamp': [result['timestamp']],
            'Feedback': ['']  # Empty initially
        }

        df_new = pd.DataFrame(data)

        # Check if file exists
        if os.path.exists(RESULTS_FILE):
            df_existing = pd.read_excel(RESULTS_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_excel(RESULTS_FILE, index=False)

    except Exception as e:
        print(f"Error saving to Excel: {e}")


def save_feedback_to_excel(prediction_id, feedback, text, prediction):
    """Save user feedback to Excel file"""
    try:
        # Prepare feedback data
        data = {
            'ID': [prediction_id],
            'Text': [text],
            'Model_Prediction': [prediction],
            'User_Feedback': [feedback],
            'Timestamp': [datetime.now().isoformat()]
        }

        df_new = pd.DataFrame(data)

        # Check if feedback file exists
        if os.path.exists(FEEDBACK_FILE):
            df_existing = pd.read_excel(FEEDBACK_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_excel(FEEDBACK_FILE, index=False)

        # Also update the results file with feedback
        if os.path.exists(RESULTS_FILE):
            df_results = pd.read_excel(RESULTS_FILE)
            df_results.loc[df_results['ID'] == prediction_id, 'Feedback'] = feedback
            df_results.to_excel(RESULTS_FILE, index=False)

    except Exception as e:
        print(f"Error saving feedback: {e}")


def get_model_metrics():
    """Get model performance metrics from saved results"""
    try:
        if not os.path.exists(RESULTS_FILE):
            return {
                "total_predictions": 0,
                "offensive_count": 0,
                "not_offensive_count": 0,
                "avg_confidence": 0,
                "feedback_stats": {"positive": 0, "negative": 0}
            }

        df = pd.read_excel(RESULTS_FILE)

        total_predictions = len(df)
        offensive_count = len(df[df['Prediction'] == 'Offensive'])
        not_offensive_count = len(df[df['Prediction'] == 'Not Offensive'])
        avg_confidence = df['Confidence'].mean() if total_predictions > 0 else 0

        # Feedback stats
        positive_feedback = len(df[df['Feedback'] == 'positive'])
        negative_feedback = len(df[df['Feedback'] == 'negative'])

        return {
            "total_predictions": total_predictions,
            "offensive_count": offensive_count,
            "not_offensive_count": not_offensive_count,
            "avg_confidence": round(avg_confidence, 2),
            "feedback_stats": {
                "positive": positive_feedback,
                "negative": negative_feedback
            }
        }

    except Exception as e:
        print(f"Error getting metrics: {e}")
        return {
            "total_predictions": 0,
            "offensive_count": 0,
            "not_offensive_count": 0,
            "avg_confidence": 0,
            "feedback_stats": {"positive": 0, "negative": 0}
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received prediction request")
        data = request.get_json()
        print(f"Request data: {data}")
        text = data.get('text', '').strip()

        if not text:
            print("No text provided")
            return jsonify({"error": "Please enter some text"}), 400

        print(f"Analyzing text: {text[:50]}...")
        result = predict_text(text)
        print(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"Error in predict route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        prediction_id = data.get('id')
        feedback_type = data.get('feedback')  # 'positive' or 'negative'
        text = data.get('text')
        prediction = data.get('prediction')

        if not all([prediction_id, feedback_type, text, prediction]):
            return jsonify({"error": "Missing required data"}), 400

        save_feedback_to_excel(prediction_id, feedback_type, text, prediction)
        return jsonify({"message": "Feedback saved successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/metrics')
def metrics():
    try:
        metrics_data = get_model_metrics()
        return jsonify(metrics_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/export/results')
def export_results():
    try:
        if os.path.exists(RESULTS_FILE):
            return send_file(RESULTS_FILE, as_attachment=True, download_name='prediction_results.xlsx')
        else:
            return jsonify({"error": "No results file found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/export/feedback')
def export_feedback():
    try:
        if os.path.exists(FEEDBACK_FILE):
            return send_file(FEEDBACK_FILE, as_attachment=True, download_name='model_feedback.xlsx')
        else:
            return jsonify({"error": "No feedback file found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Loading model...")
    if load_model_and_tokenizer():
        print("Model loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check your model path.")