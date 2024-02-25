from catboost import CatBoostClassifier
from flask import Flask, jsonify, request, send_file
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)


iris_dataset = load_iris()
X, y = iris_dataset.data, iris_dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = CatBoostClassifier(
    iterations=100, learning_rate=0.1, depth=6, loss_function="MultiClass"
)

model.fit(X_train, y_train, verbose=False)
acc = accuracy_score(y_test, model.predict(X_test))*100
print(f"Accuracy: {acc: .0f}%")


@app.route("/")
def start_page():
    return """
<!DOCTYPE html>
<html>
<head>
    <style>
        h1 {
            text-align: center;
            font-size: 32px;
        }
        p {
            text-align: center;
            font-size: 18px;
        }
        img {
            display: block;
            margin: auto;
        }
    </style>
</head>
<body>
    <h1>Welcome to the Iris Flower Classifier</h1>
    <p>The classifier used is CatBoostClassifier. It was trained and got 100% accuracy on the test data.</p>
    <img src="/display_image" alt="Features">
</body>
</html>
    """

@app.route("/display_image")
def display_image():
    return send_file("features.png", mimetype='image/png')


@app.route("/predict", methods=["POST"])
def predict():
    features = list(request.get_json()["features"])
    target_name = iris_dataset.target_names[model.predict(features)][0]
    return jsonify({"prediction": target_name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# CLI local: curl -X POST -H "Content-Type: application/json" -d "{\"features\": [5.1, 3.5, 1.4, 0.2]}" http://127.0.0.1:5000/predict
# $env:FLASK_APP="iris_backend.py"
# flask run
# python iris_backend.py