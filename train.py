import pandas as pd
def main() :
    # read data
    #df = pd.read_csv("iris.csv")
    #print(df.head())
    df = pd.DataFrame({'SepalLengthCm' : [5.1,4.9,4.3], 'SepalWidthCm' : [3.4,3.9,2.8], 'PetalLengthCM' : [1.4,1.91,2.1], 'PetalWidthCm': [0.2,0.1,0.3], 'Species' : ['A', 'B', 'A']})
    # Feature matrix
    #X = df.drop(columns=['Species'], axis=1)
    X = df.iloc[:,:-1].values
    #print(X[:1])
    # Output variable
    y = df.iloc[:, -1]

    # Label encoder
    from sklearn.preprocessing import LabelEncoder
    import joblib

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    joblib.dump(encoder, "iris_encoder.pkl")

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # train model
    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, "iris_KNN_model.pkl")

    df_ = pd.DataFrame({'SepalLengthCm' : [5.1,4.9,4.3], 'SepalWidthCm' : [3.4,3.9,2.8], 'PetalLengthCM' : [1.4,1.91,2.1], 'PetalWidthCm': [0.2,0.1,0.3]})
    #df = pd.read_csv(input_file) 
    classifier_loaded = joblib.load("iris_KNN_model.pkl")
    encoder_loaded = joblib.load("iris_encoder.pkl") # Read input data
    X = df_.iloc[:,:].values
    #print(X)

    #prediction_raw = classifier_loaded.predict(X) 
    prediction_real = encoder_loaded.inverse_transform(classifier_loaded.predict(X))        # Get predictions
    pd.DataFrame(prediction_real).to_csv("output_predictions.csv", index=False)  # Save predictions
    print(prediction_real)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Run predictions with the model.")
    #parser.add_argument("input_csv", help="Path to the input CSV file")
    #args = parser.parse_args()
    main()