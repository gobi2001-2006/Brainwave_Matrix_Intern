from flask import Flask, render_template, request
import pickle
from preprocess import preprocess

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['news']
        clean_text = preprocess_text(input_text)
        transformed_input = vectorizer.transform([clean_text])
        prediction = model.predict(transformed_input)[0]
        
        result = "Real News 📰✅" if prediction == 1 else "Fake News 🚫📰"
        return render_template('index.html', prediction=result, original=input_text)

if __name__ == "__main__":
    app.run(debug=True)
