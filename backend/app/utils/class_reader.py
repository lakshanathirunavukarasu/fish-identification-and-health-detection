import json

def load_species_mapping():
    with open("app/utils/species_class_names.json", "r") as f:
        return json.load(f)
    
def load_health_mapping():
    with open("app/utils/health_class_names.json", "r") as f:
        return json.load(f)