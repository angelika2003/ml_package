from joblib import load
import os
from pkg_resources import resource_filename

def load_model():
    model_path = resource_filename("ml_package", 'model/random_forest_model.joblib')
    return load(model_path)

def load_scaler():  # Новая функция для загрузки scaler
    scaler_path = resource_filename(__name__, 'model/scaler.joblib')
    return load(scaler_path)

def predict(data):
    model = load_model()
    scaler = load_scaler()
    
    # Если data это numpy array, преобразуем в DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=['batch_size', 'num_gpus', 'flops', 'parameters'])
    
    data_scaled = scaler.transform(data)
    return model.predict(data_scaled)