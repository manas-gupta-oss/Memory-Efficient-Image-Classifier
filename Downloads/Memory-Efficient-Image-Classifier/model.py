from sklearn.svm import LinearSVC

class Classifier:
    def __init__(self):
        self.model = LinearSVC()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)