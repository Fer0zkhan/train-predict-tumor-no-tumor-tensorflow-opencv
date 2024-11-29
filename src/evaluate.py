from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, x_test, y_test):
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
