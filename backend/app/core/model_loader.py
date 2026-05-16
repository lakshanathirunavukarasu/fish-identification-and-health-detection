from tensorflow.keras.models import load_model

species_model = None
health_model = None

def load_models():
    global species_model, health_model
    
    if species_model is None:
        species_model = load_model("saved_models/species_model.keras", compile=False)

    if health_model is None:
        health_model = load_model("saved_models/health_model.keras", compile=False)

    return species_model, health_model