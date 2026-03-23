from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    print(classification_report(y_test, preds))
    
    return model