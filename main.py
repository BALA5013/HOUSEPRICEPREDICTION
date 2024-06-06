from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('C:\\Desktop\\HOUSE PRICE PREDICTION\\train.csv')
pipe = pickle.load(open("C:\\Desktop\\HOUSE PRICE PREDICTION\\HHPModel.pkl", 'rb'))

index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }

        header {
            background-color: #333;
            color: #fff;
            padding: 10px;
            text-align: center;
        }

        main {
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        footer {
            text-align: center;
            padding: 10px;
            background-color: #333;
            color: #fff;
            position: fixed;
            bottom: 0;
            width: 100%;
        }

        form {
            margin-top: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
        }

        select {
            width: 100%;
            padding: 8px;
            margin-bottom: 16px;
        }

        button {
            background-color: #333;
            color: #fff;
            padding: 10px;
            border: none;
            cursor: pointer;
        }

        #predictedPrice {
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <header>
        <h1>House Price Prediction</h1>
    </header>
    <main>
        <p>Welcome to House Price Prediction Model!</p>
        <form id="predictionForm">
            <label for="beds">Bedrooms:</label>
            <select id="beds" name="beds">
                <option value="" disabled selected>Select number of bedrooms</option>
                {% for bedroom in bedrooms %}
                    <option value="{{ bedroom }}">{{ bedroom }}</option>
                {% endfor %}
            </select>

            <label for="baths">Baths:</label>
            <select id="baths" name="baths">
                <option value="" disabled selected>Select number of bathrooms</option>
                {% for bathroom in bathrooms %}
                    <option value="{{ bathroom }}">{{ bathroom }}</option>
                {% endfor %}
            </select>

            <label for="size">Size:</label>
            <select id="size" name="size">
                <option value="" disabled selected>Select size of the house</option>
                {% for house_size in sizes %}
                    <option value="{{ house_size }}">{{ house_size }} sqft</option>
                {% endfor %}
            </select>

            <label for="zip_code">Zip Code:</label>
            <select id="zip_code" name="zip_code">
                <option value="" disabled selected>Select zip code</option>
                {% for zip_code in zip_codes %}
                    <option value="{{ zip_code }}">{{ zip_code }}</option>
                {% endfor %}
            </select>

            <button type="button" onclick="sendData()">Predict Price</button>
            <div id="predictedPrice"></div>
        </form>
    </main>
    <footer>
        <p>&copy; 2024 House Price Prediction. All rights reserved.</p>
    </footer>
    <script>
        function sendData() {
            const form = document.getElementById('predictionForm');
            const formData = new FormData(form);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById("predictedPrice").innerHTML = "Price: INR " + data.price;
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template_string(index_html, bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                              columns=['beds', 'baths', 'size', 'zip_code'])

    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    prediction = pipe.predict(input_data)[0]

    return jsonify(price=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
