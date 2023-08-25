from fastapi import FastAPI
from starlette.responses import JSONResponse
from iris_classifier import IrisClassifier
from models import Iris

app = FastAPI()

@app.get('/', status_code=200)
async def healthcheck():
    return 'Iris classifier is all ready to go!'

@app.post('/classify_iris')
def extract_name(iris_features: Iris):

    # Instantiate the classifier
    iris_classifier = IrisClassifier()

    # Classify the iris
    iris_type = iris_classifier.classify_iris(iris_features)
    
    return JSONResponse(iris_classifier.classify_iris(iris_features))