import joblib
import argparse
import pandas as pd


# src/predict.py

def main():
    df = pd.DataFrame({'SepalLengthCm' : [5.1,4.9,4.3], 'SepalWidthCm' : [3.4,3.9,2.8], 'PetalLengthCM' : [1.4,1.91,2.1], 'PetalWidthCm': [0.2,0.1,0.3]})
    #df = pd.read_csv(input_file) 
    classifier_loaded = joblib.load("iris_KNN_model.pkl")
    encoder_loaded = joblib.load("iris_encoder.pkl") # Read input data
    X = df.iloc[:,:].values
    print(X)
    #y = df.iloc[:, -1]
    #y = encoder_loaded.transform(y)
    #X = X[:10]
    #y = y[:10]

    #prediction_raw = classifier_loaded.predict(X) 
    prediction_real = encoder_loaded.inverse_transform(classifier_loaded.predict(X))        # Get predictions
    pd.DataFrame(prediction_real).to_csv("output_predictions.csv", index=False)  # Save predictions
    print(prediction_real)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Run predictions with the model.")
    #parser.add_argument("input_csv", help="Path to the input CSV file")
    #args = parser.parse_args()
    main()
