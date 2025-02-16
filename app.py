from flask import Flask, render_template, request, redirect, url_for, session, flash
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for session management

# Load the pre-trained model
MODEL_PATH = r"C:\Users\ritan\Desktop\plant disease detection_00111\model\plant_disease_model_001.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load disease-cure data from CSV
CSV_PATH = r"C:\Users\ritan\Desktop\plant disease detection_00111\plant_disease_cures_001.csv"
disease_cure_df = pd.read_csv(CSV_PATH)
disease_cure_dict = dict(zip(disease_cure_df["Disease Name"], disease_cure_df["Cure"]))

# Hardcoded user credentials (since no database is used)
USER_CREDENTIALS = {
    "test@example.com": "password123"
}

# Allowed image file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Class labels for prediction
CLASS_LABELS = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Blueberry_healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)healthy',
    'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust',
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape_Black_rot',
    'Grape_Esca(Black_Measles)', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape_healthy',
    'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy',
    'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight',
    'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy',
    'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite',
    'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus',
    'Tomato__healthy'
]

# Preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size based on model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

@app.route("/", methods=["GET", "POST"])
def login():
    """ User login page """
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if email in USER_CREDENTIALS and USER_CREDENTIALS[email] == password:
            session["user"] = email
            return redirect(url_for("index"))
        else:
            flash("Invalid email or password. Try again!", "danger")
    return render_template("login.html")

@app.route("/index", methods=["GET", "POST"])
def index():
    """ Image upload and prediction """
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "warning")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            # Preprocess and predict
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]  # Get class index

            # Get class label and corresponding cure
            disease_name = CLASS_LABELS[predicted_class] if predicted_class < len(CLASS_LABELS) else "Unknown Disease"
            cure_name = disease_cure_dict.get(disease_name, "No Cure Found")

            # Format output as "DISEASE_NAME cured by CURE_NAME"
            result = f"{disease_name} cured by {cure_name}"

            return render_template("index.html", prediction=result, image_path=file_path)

    return render_template("index.html", prediction=None, image_path=None)

@app.route("/logout")
def logout():
    """ Logout and clear session """
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route('/add_card')
def add_card():
    return "Credit Card Addition Page"

if __name__ == "__main__":
    app.run(debug=True)