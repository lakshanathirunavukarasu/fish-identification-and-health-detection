from fastapi import FastAPI
from app.routes.prediction import router as prediction_router
from app.routes.service import router as service_router
from app.core.model_loader import load_models
from app.db.database import engine, Base
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FishScan AI")

# ---------------- CORS SETUP ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router, prefix="/api")
app.include_router(service_router, prefix="/api/v1")
# app.include_router(router, prefix="/api/species")
# app.include_router(router, prefix="/api/health_conditions")

Base.metadata.create_all(bind=engine)

@app.on_event("startup")
def startup_event():
    load_models()  # preload models once