from flask import Flask,render_template,request, redirect, url_for,jsonify
import google.generativeai as genai
from PIL import Image
import base64
import io
import logging
import numpy as np
import pandas as pd
chat_model = genai.GenerativeModel('gemini-pro')
import warnings
warnings.filterwarnings('ignore')
import os
my_api_key_gemini = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=my_api_key_gemini)
app = Flask(__name__)
import pickle
import numpy as np
with open('model/diabetes.pkl', 'rb') as file:
    diabetes_model = pickle.load(file)

# Verify that the model has been loaded
print("Model loaded successfully!")
@app.route('/main')
def index():
    return render_template("main.html")
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/diabetes')
def diabetes():
    return render_template("diabetes.html")
@app.route('/predict_diabetes', methods=['POST'])          
def predict():
    # Collecting input data
    input_data = np.array([
        int(request.form.get('Pregnancies')),
        int(request.form.get('Glucose')),
        int(request.form.get('BloodPressure')),
        int(request.form.get('SkinThickness')),
        int(request.form.get('Insulin')),
        float(request.form.get('BMI')),
        float(request.form.get('DiabetesPedigreeFunction')),
        int(request.form.get('Age'))
    ]).reshape(1, -1)  # Reshaping to ensure it's a single sample
    input_df = pd.DataFrame(input_data, columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 
        'Age'])
    # Prediction
    diabetes_result = diabetes_model.predict(input_data)

    if diabetes_result[0] == 1: 
        prediction = "The patient seems to have Diabetes:("
    else:
        prediction = "The patient seems to be Normal:)"
    return render_template('diabetes_result.html', diabetes_result=prediction)


# Define your 404 error handler to redirect to the index page
@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))
@app.route('/heal_chat', methods=['POST', 'GET'])
def heal_chat():
    if request.method == 'POST':
        try:
            prompt = request.form['prompt']
            question = prompt

            response = chat_model.generate_content(question)

            if response.text:
                return response.text
            else:
                return "Sorry, but I think Gemini didn't want to answer that!"
        except Exception as e:
            return "Sorry, but Gemini didn't want to answer that!"

    return render_template('heal_chat.html', **locals())
@app.route('/foodmate')
def foodmate():
    return render_template('foodmate.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['uploadInput']
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Ensure correct mime type based on file extension
        if uploaded_file.filename.endswith('.jpg') or uploaded_file.filename.endswith('.jpeg'):
            mime_type = 'image/jpeg'
        elif uploaded_file.filename.endswith('.png'):
            mime_type = 'image/png'
        else:
            return jsonify(error='Unsupported file format'), 400
        
        # Encode image to base64 for sending to API
        buffered = io.BytesIO()
        image.save(buffered, format=image.format)
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        image_parts = [{
            "mime_type": mime_type,
            "data": encoded_image
        }]
        
        input_prompt = """
            You are an expert in nutritionist where you need to see the food items from the image
            and calculate the total calories, also provide the details of every food items with calories intake
            is below format

            1. Item 1 - no of calories, protein
            2. Item 2 - no of calories, protein
            ----
            ----
            Also mention disease risk from these items
            Finally you can also mention whether the food items are healthy or not and Suggest Some Healthy Alternative 
            is below format          
            1. Item 1 - no of calories, protein
            2. Item 2 - no of calories, protein
            ----
            ----
        """

        # Simulate API response (replace with actual API call)
        model1 = genai.GenerativeModel('gemini-1.5-flash')
        response = model1.generate_content([input_prompt, image_parts[0]])
        result = response.text

        return jsonify(result=result, image=encoded_image)
    
    return jsonify(error='No file uploaded'), 400
@app.route('/about')
def about():
    return render_template("about.html")
if __name__ == "__main__":
    app.run(debug=True)
    