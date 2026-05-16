from fastapi import APIRouter
from app.utils.class_reader import load_species_mapping, load_health_mapping
from app.db.database import get_db
from app.db.models import Species,HealthCondition
from fastapi import Depends
from sqlalchemy.orm import Session

router = APIRouter()

#Get Species and Health Mappings
@router.get("/species")
async def get_species(db:Session = Depends(get_db)):
    species_list = db.query(Species).all()
    return [{"common_name": s.common_name, "scientific_name": s.scientific_name} for s in species_list]

@router.get("/health_conditions")
async def get_health_conditions(db:Session = Depends(get_db)):
    health_list = db.query(HealthCondition).all()
    return [{"name": h.name, "category": h.category, "symptoms": h.symptoms} for h in health_list]


#Get Species by Common Name
@router.get("/species/{common_name}")
async def get_species_by_name(common_name: str, db:Session = Depends(get_db)):
    species = db.query(Species).filter(Species.common_name == common_name).first()
    if not species:
        return {"error": "Species not found"}
    return {
        "scientific_name": species.scientific_name
    }