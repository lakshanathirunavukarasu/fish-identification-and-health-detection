from app.db.database import SessionLocal, Base, engine
from app.db.models import Species, HealthCondition

# 🔥 CREATE TABLES FIRST
Base.metadata.create_all(bind=engine)

db = SessionLocal()

# ---------------- SPECIES DATA ----------------
species_data = [
    ("Bangus", "Chanos chanos"),
    ("Big Head Carp", "Hypophthalmichthys nobilis"),
    ("Black Spotted Barb", "Puntius binotatus"),
    ("Catfish", "Clarias batrachus"),
    ("Climbing Perch", "Anabas testudineus"),
    ("Fourfinger Threadfin", "Eleutheronema tetradactylum"),
    ("Freshwater Eel", "Anguilla anguilla"),
    ("Glass Perchlet", "Parambassis ranga"),
    ("Goby", "Glossogobius giuris"),
    ("Gold Fish", "Carassius auratus"),
    ("Gourami", "Trichogaster trichopterus"),
    ("Grass Carp", "Ctenopharyngodon idella"),
    ("Green Spotted Puffer", "Tetraodon nigroviridis"),
    ("Indian Carp", "Catla catla"),
    ("Indo-Pacific Tarpon", "Megalops cyprinoides"),
    ("Jaguar Gapote", "Parachromis managuensis"),
    ("Janitor Fish", "Pterygoplichthys pardalis"),
    ("Knifefish", "Notopterus notopterus"),
    ("Long-Snouted Pipefish", "Syngnathus typhle"),
    ("Mosquito Fish", "Gambusia affinis"),
    ("Mudfish", "Channa striata"),
    ("Mullet", "Mugil cephalus"),
    ("Pangasius", "Pangasius pangasius"),
    ("Perch", "Perca fluviatilis"),
    ("Scat Fish", "Scatophagus argus"),
    ("Silver Barb", "Barbonymus gonionotus"),
    ("Silver Carp", "Hypophthalmichthys molitrix"),
    ("Silver Perch", "Bidyanus bidyanus"),
    ("Snakehead", "Channa marulius"),
    ("Tenpounder", "Elops machnata"),
    ("Tilapia", "Oreochromis niloticus"),
]

for index, (common, scientific) in enumerate(species_data):
    db.add(
        Species(
            class_index=index,   # 🔥 IMPORTANT
            common_name=common,
            scientific_name=scientific
        )
    )

db.commit()
print("Species inserted successfully ✅")


# ---------------- HEALTH DATA ----------------
health_data = [
    ("Bacterial Red disease", "Bacterial", "High", "Red lesions, hemorrhages, fin rot"),
    ("Bacterial diseases - Aeromoniasis", "Bacterial", "High", "Ulcers, swollen abdomen, scale loss"),
    ("Bacterial gill disease", "Bacterial", "Medium", "Damaged gills, breathing difficulty"),
    ("FreshFish", "Normal", "None", "No disease detected"),
    ("Fungal diseases Saprolegniasis", "Fungal", "Medium", "Cotton-like growth on skin"),
    ("Healthy Fish", "Normal", "None", "Healthy appearance"),
    ("Parasitic diseases", "Parasitic", "Medium", "Skin irritation, abnormal swimming"),
    ("Viral diseases White tail disease", "Viral", "High", "White muscle tissue in tail region"),
]

for index, (name, category, severity, symptoms) in enumerate(health_data):
    db.add(
        HealthCondition(
            class_index=index,   # 🔥 IMPORTANT
            name=name,
            category=category,
            severity_level=severity,
            symptoms=symptoms
        )
    )

db.commit()
db.close()

print("Health conditions inserted successfully ✅")