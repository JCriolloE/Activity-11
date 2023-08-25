import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from starlette.responses import JSONResponse

from models import Iris

class IrisClassifier:
    def __init__(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.clf = self.train_model()
        self.iris_type = {
            0: 'setosa',
            1: 'versicolor',
            2: 'virginica'
        }

    def train_model(self) -> LogisticRegression:
        clf = LogisticRegression(solver='lbfgs',
                          max_iter=1000,
                          multi_class='multinomial').fit(self.X, self.y)
        return clf
    
    def classify_iris(self, iris: Iris) -> JSONResponse:
        if not isinstance(iris, Iris):
            raise ValueError('iris must be an instance of Iris')

        X = [iris.sepal_length, iris.sepal_width, 
             iris.petal_length, iris.petal_width]    
       
        prediction = self.clf.predict_proba([X])
        return JSONResponse({'class': self.iris_type[np.argmax(prediction)],
                'probability': round(max(prediction[0]), 2)})
