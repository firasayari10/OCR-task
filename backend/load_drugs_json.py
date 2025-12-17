"""
Load medications from JSON file into vector database.
"""

import json
import logging
from medication_vector_db import MedicationVectorDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_drugs_from_json(json_path: str, db_path: str = "medication_db"):
    """
    Load medications from JSON file and add to vector database.
    
    Args:
        json_path: Path to JSON file with drug data
        db_path: Path to vector database directory
    """
    # Read JSON file
    logger.info(f"Loading drugs from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        drugs = json.load(f)
    
    logger.info(f"Found {len(drugs)} drugs in JSON file")
    
    # Convert to medication format for vector DB
    medications = []
    for drug in drugs:
        # Skip duplicate entries
        if "duplicate" in drug.get('name', '').lower():
            continue
        
        # Extract main information
        med = {
            "name": drug.get('name', ''),
            "generic_name": drug.get('generic', ''),
            "drug_class": drug.get('class', ''),
            "common_uses": ', '.join(drug.get('common_uses', [])) if isinstance(drug.get('common_uses'), list) else drug.get('common_uses', ''),
            "precautions": ', '.join(drug.get('precautions_contraindications', [])) if isinstance(drug.get('precautions_contraindications'), list) else drug.get('precautions_contraindications', ''),
            "side_effects": ', '.join(drug.get('common_side_effects', [])) if isinstance(drug.get('common_side_effects'), list) else drug.get('common_side_effects', ''),
            "listed_in_tunisia_neml": drug.get('listed_in_tunisia_neml', False),
            "dosage": "",  # Not provided in JSON
            "forme": "Various",  # Not specified in JSON
            "usage": "ESSENTIAL" if drug.get('listed_in_tunisia_neml') else "GENERAL",
            "category": _infer_category(drug.get('class', ''), drug.get('common_uses', [])),
            "description": f"{drug.get('class', '')} - {', '.join(drug.get('common_uses', [])) if isinstance(drug.get('common_uses'), list) else drug.get('common_uses', '')}"
        }
        medications.append(med)
    
    logger.info(f"Processed {len(medications)} medications")
    
    # Create/load database
    db = MedicationVectorDB(db_path)
    
    # Clear existing data to avoid duplicates
    logger.info("Clearing existing database...")
    db.index = None
    db.metadata = []
    db._load_or_create_index()
    
    # Add medications
    db.add_medications(medications)
    
    # Show statistics
    stats = db.get_stats()
    logger.info("\n" + "="*60)
    logger.info("DATABASE STATISTICS")
    logger.info("="*60)
    logger.info(f"Total medications: {stats['total_medications']}")
    
    logger.info("\nTop Categories:")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {cat}: {count}")
    
    logger.info("\nUsage Types:")
    for usage, count in sorted(stats['usage_types'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {usage}: {count}")
    
    logger.info("="*60)
    
    return db


def _infer_category(drug_class: str, uses: list) -> str:
    """Infer medication category from class and uses."""
    drug_class_lower = drug_class.lower()
    uses_str = ' '.join(uses).lower() if isinstance(uses, list) else str(uses).lower()
    
    # Categorize based on drug class
    if 'antibiotic' in drug_class_lower:
        return "Antibiotic"
    elif 'nsaid' in drug_class_lower or 'analgesic' in drug_class_lower:
        return "Analgesic/Anti-inflammatory"
    elif 'antihypertensive' in drug_class_lower or 'ace inhibitor' in drug_class_lower or 'beta-blocker' in drug_class_lower or 'calcium channel' in drug_class_lower:
        return "Cardiovascular"
    elif 'diuretic' in drug_class_lower:
        return "Diuretic"
    elif 'statin' in drug_class_lower or 'lipid' in drug_class_lower:
        return "Lipid-lowering"
    elif 'antidiabetic' in drug_class_lower or 'diabetes' in uses_str or 'insulin' in drug_class_lower:
        return "Diabetes Management"
    elif 'antihypertensive' in drug_class_lower or 'hypertension' in uses_str:
        return "Antihypertensive"
    elif 'bronchodilator' in drug_class_lower or 'asthma' in uses_str or 'copd' in uses_str:
        return "Respiratory"
    elif 'proton pump' in drug_class_lower or 'h2 blocker' in drug_class_lower or 'antacid' in drug_class_lower:
        return "Gastrointestinal"
    elif 'anticoagulant' in drug_class_lower or 'antiplatelet' in drug_class_lower:
        return "Anticoagulant/Antiplatelet"
    elif 'antidepressant' in drug_class_lower or 'ssri' in drug_class_lower or 'snri' in drug_class_lower:
        return "Antidepressant"
    elif 'antipsychotic' in drug_class_lower:
        return "Antipsychotic"
    elif 'anticonvulsant' in drug_class_lower or 'antiepileptic' in drug_class_lower:
        return "Anticonvulsant"
    elif 'antihistamine' in drug_class_lower or 'allergy' in uses_str:
        return "Antihistamine"
    elif 'corticosteroid' in drug_class_lower or 'steroid' in drug_class_lower:
        return "Corticosteroid"
    elif 'antifungal' in drug_class_lower:
        return "Antifungal"
    elif 'antiviral' in drug_class_lower:
        return "Antiviral"
    elif 'immunosuppressant' in drug_class_lower:
        return "Immunosuppressant"
    elif 'hormone' in drug_class_lower or 'thyroid' in drug_class_lower:
        return "Hormone/Endocrine"
    elif 'vitamin' in drug_class_lower or 'supplement' in drug_class_lower:
        return "Vitamin/Supplement"
    elif 'chemotherapy' in drug_class_lower or 'cytotoxic' in drug_class_lower or 'cancer' in uses_str:
        return "Oncology"
    else:
        return "General"


if __name__ == "__main__":
    import sys
    
    json_file = "drugs.json" if len(sys.argv) < 2 else sys.argv[1]
    db_path = "medication_db" if len(sys.argv) < 3 else sys.argv[2]
    
    db = load_drugs_from_json(json_file, db_path)
    
    # Test search
    print("\n" + "="*60)
    print("TESTING DATABASE - Sample Searches")
    print("="*60)
    
    test_queries = ["paracetamol", "diabetes", "blood pressure", "antibiotic", "pain"]
    for query in test_queries:
        print(f"\nSearch: '{query}'")
        results = db.search(query, top_k=3)
        for r in results[:3]:
            print(f"  - {r['name']} ({r.get('drug_class', 'N/A')}) - Score: {r['similarity_score']:.3f}")
    
    print("\n" + "="*60)
    print("✓ Database built successfully from JSON!")
    print(f"✓ Total medications: {len(db.metadata)}")
    print("="*60)
