import joblib
import argparse
import pandas as pd


# src/predict.py

def main(input_file):
    #df = pd.read_csv("/data/iris.csv")
    df = pd.read_csv(input_file) 
    classifier_loaded = joblib.load("output/iris_KNN_model.pkl")
    encoder_loaded = joblib.load("output/iris_encoder.pkl") # Read input data
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1]
    y = encoder_loaded.transform(y)
    X = X[:10]
    y = y[:10]

    prediction_raw = classifier_loaded.predict(X)         # Get predictions
    pd.DataFrame(prediction_raw).to_csv("output_predictions.csv", index=False)  # Save predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions with the model.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    args = parser.parse_args()
    main(args.input_csv)
