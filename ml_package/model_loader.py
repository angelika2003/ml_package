from joblib import load
import os
from pkg_resources import resource_filename

def load_model():
    model_path = resource_filename(__name__, 'model/random_forest_model.joblib')
    return load(model_path)

def predict(data):
    model = load_model()
    return model.predict(data)