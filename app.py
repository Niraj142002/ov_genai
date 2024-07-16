from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.intel import OVModelForCausalLM
import torch



app = Flask(__name__)

# Load the tokenizer and model from a local path
model_name_or_path = r"/Users/niraj/Downloads/Intel/saved-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    model_id = "./tinyllama-art-s-1"
    ov_model = OVModelForCausalLM.from_pretrained("ov_model", use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = ov_model.generate(**inputs)
    summary = tokenizer.decode(outputs[0])

    
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
