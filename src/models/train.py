import pandas as pd

# read data
df = pd.read_csv("data/iris.csv")
print(df.head())

# Feature matrix
X = df.iloc[:, :-1].values

# Output variable
y = df.iloc[:, -1]

# Label encoder
from sklearn.preprocessing import LabelEncoder
import joblib

encoder = LabelEncoder()
y = encoder.fit_transform(y)

joblib.dump(encoder, "output/iris_encoder.pkl")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# train model
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

joblib.dump(classifier, "output/iris_KNN_model.pkl")

