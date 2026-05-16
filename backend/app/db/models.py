from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship,base
from datetime import datetime
from app.db.database import Base


# ---------------- Species Table ----------------
class Species(Base):
    __tablename__ = "species"

    id = Column(Integer, primary_key=True, index=True)
    class_index = Column(Integer, unique=True, nullable=False) 
    common_name = Column(String, unique=True, index=True, nullable=False)
    scientific_name = Column(String)
    description = Column(Text)

    predictions = relationship("PredictionHistory", back_populates="species")


# ---------------- Health Condition Table ----------------
class HealthCondition(Base):
    __tablename__ = "health_conditions"

    id = Column(Integer, primary_key=True, index=True)
    class_index = Column(Integer, nullable=False)
    name = Column(String, unique=True, index=True, nullable=False)
    category = Column(String)
    symptoms = Column(Text)
    severity_level = Column(String)

    predictions = relationship("PredictionHistory", back_populates="health_condition")


# ---------------- Prediction History ----------------
class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    species_id = Column(Integer, ForeignKey("species.id"))
    health_condition_id = Column(Integer, ForeignKey("health_conditions.id"))

    species_confidence = Column(Float)
    health_confidence = Column(Float)

    image_path = Column(String)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    species = relationship("Species", back_populates="predictions")
    health_condition = relationship("HealthCondition", back_populates="predictions")