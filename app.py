from flask import Flask, request, render_template
import numpy as np
from model import get_model

app = Flask(__name__)
model = get_model()  # Load pre-trained model

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            num1 = float(request.form['num1'])
            num2 = float(request.form['num2'])
            input_data = np.array([[num1, num2]])
            prediction = model.predict(input_data, verbose=0)
            result = f"Predicted Sum: {prediction[0][0]:.2f}"
        except Exception as e:
            result = f"Error: {e}"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)  # Important for ngrok
