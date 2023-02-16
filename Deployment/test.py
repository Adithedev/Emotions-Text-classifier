import joblib
clf = joblib.load("model.joblib")

sample = ["I am so pissed off rn"]

print(clf.predict(sample))