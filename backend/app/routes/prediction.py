from fastapi import APIRouter, UploadFile, File
from app.core.model_loader import load_models
from app.utils.class_reader import load_species_mapping, load_health_mapping
from PIL import Image
import numpy as np
import io
from app.db.database import get_db
from app.db.models import Species,HealthCondition
from fastapi import Depends
from sqlalchemy.orm import Session

router = APIRouter()

species_classes = load_species_mapping()
health_classes = load_health_mapping()


def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.expand_dims(image, axis=0)
    return image


@router.post("/predict")
async def predict(file: UploadFile = File(...),db:Session = Depends(get_db)):

    # 🔥 THIS IS THE FIX
    species_model, health_model = load_models()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed = preprocess_image(image)

    species_prediction = species_model.predict(processed)
    health_prediction = health_model.predict(processed)

    species_index = np.argmax(species_prediction)
    health_index = np.argmax(health_prediction)

    # 🔥 Scientific name from DB
    species_obj = db.query(Species).filter(
        Species.common_name == species_classes[species_index]
    ).first()

    scientific_name = species_obj.scientific_name if species_obj else "Not Found"

    #Health condition details from DB
    health_obj = db.query(HealthCondition).filter(
        HealthCondition.name == health_classes[health_index]
    ).first()

    health_details = {
        "category": health_obj.category if health_obj else "Not Found",
        "symptoms": health_obj.symptoms if health_obj else "Not Found",
    }

    return {
        "species": species_classes[species_index],
        "species_confidence": float(np.max(species_prediction) * 100),
        "scientific_name": scientific_name,
        "health": health_classes[health_index],
        "health_confidence": float(np.max(health_prediction) * 100),
        "health_details": health_details
    }
