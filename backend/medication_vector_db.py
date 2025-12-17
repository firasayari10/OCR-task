"""
Vector Database for Essential Medications

This module creates and manages a vector database of essential medications
that can be queried by the DrugInformationAgent.
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging

logger = logging.getLogger(__name__)


class MedicationVectorDB:
    """Vector database for medication information using FAISS."""
    
    def __init__(self, db_path: str = "medication_db"):
        """
        Initialize the medication vector database.
        
        Args:
            db_path: Directory to store database files
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        
        self.index_path = os.path.join(db_path, "faiss_index.bin")
        self.metadata_path = os.path.join(db_path, "metadata.json")
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Load or initialize embedding model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Load or initialize FAISS index
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info("Loading existing vector database...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} medications from database")
        else:
            logger.info("Creating new vector database...")
            # Create FAISS index (L2 distance)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
    
    def add_medications(self, medications: List[Dict[str, Any]]):
        """
        Add medications to the vector database.
        
        Args:
            medications: List of medication dictionaries with fields:
                - name: str
                - dosage: str
                - forme: str (Comprime, Suspension, etc.)
                - usage: str (ESSENTIAL, RECOMMENDED, LIFESAVING)
                - category: str (medical category)
                - alternatives: List[str] (optional)
                - description: str (optional)
        """
        if not medications:
            logger.warning("No medications to add")
            return
        
        logger.info(f"Adding {len(medications)} medications to database...")
        
        # Create embeddings for each medication
        texts = []
        for med in medications:
            # Create rich text representation for embedding
            text = self._create_embedding_text(med)
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        if self.index is not None:
            self.index.add(embeddings)
        
        # Add metadata
        self.metadata.extend(medications)
        
        # Save to disk
        self._save_index()
        
        logger.info(f"Successfully added {len(medications)} medications. Total: {len(self.metadata)}")
    
    def _create_embedding_text(self, med: Dict[str, Any]) -> str:
        """Create rich text representation for embedding."""
        parts = [
            f"Medication: {med.get('name', 'Unknown')}",
            f"Form: {med.get('forme', '')}",
            f"Dosage: {med.get('dosage', '')}",
            f"Usage: {med.get('usage', '')}",
            f"Category: {med.get('category', '')}",
        ]
        
        if med.get('alternatives'):
            parts.append(f"Alternatives: {', '.join(med['alternatives'])}")
        
        if med.get('description'):
            parts.append(f"Description: {med['description']}")
        
        return " | ".join(parts)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for medications similar to the query.
        
        Args:
            query: Search query (medication name, category, etc.)
            top_k: Number of results to return
        
        Returns:
            List of medication dictionaries with similarity scores
        """
        if len(self.metadata) == 0:
            logger.warning("Database is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search in FAISS
        if self.index is None:
            return []
        
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.metadata)))
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(1 / (1 + dist))  # Convert distance to similarity
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def search_by_name(self, name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for medications by name."""
        return self.search(f"Medication: {name}", top_k)
    
    def search_by_category(self, category: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for medications by category."""
        return self.search(f"Category: {category}", top_k)
    
    def get_alternatives(self, medication_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get alternative medications for a given medication."""
        # Search for similar medications
        results = self.search_by_name(medication_name, top_k + 1)
        
        # Filter out the exact match
        alternatives = [r for r in results if r.get('name', '').lower() != medication_name.lower()]
        
        return alternatives[:top_k]
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Database saved to {self.db_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.metadata:
            return {"total_medications": 0}
        
        categories: Dict[str, int] = {}
        usage_types: Dict[str, int] = {}
        forms: Dict[str, int] = {}
        
        for med in self.metadata:
            # Count categories
            cat = med.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count usage types
            usage = med.get('usage', 'Unknown')
            usage_types[usage] = usage_types.get(usage, 0) + 1
            
            # Count forms
            forme = med.get('forme', 'Unknown')
            forms[forme] = forms.get(forme, 0) + 1
        
        return {
            "total_medications": len(self.metadata),
            "categories": categories,
            "usage_types": usage_types,
            "forms": forms
        }


def create_sample_database():
    """Create a sample database with essential medications."""
    
    # Sample medications based on the image format
    sample_medications = [
        {
            "name": "Glicédioline",
            "dosage": "125 mg + 250 mg/10mg",
            "forme": "Comprime",
            "usage": "ESSENTIAL",
            "category": "Diabetes Management",
            "description": "Combination medication for type 2 diabetes management",
            "alternatives": ["Metformin", "Glipizide"]
        },
        {
            "name": "Glicédioline",
            "dosage": "125 mg + 500 mg",
            "forme": "Suspension",
            "usage": "ESSENTIAL",
            "category": "Diabetes Management"
        },
        {
            "name": "N-Systane",
            "dosage": "100,000 UI",
            "forme": "Comprime",
            "usage": "RECOMMENDED",
            "category": "Antifungal",
            "description": "Antifungal medication"
        },
        {
            "name": "N-Systane",
            "dosage": "500,000 UI",
            "forme": "Comprime",
            "usage": "ESSENTIAL",
            "category": "Antifungal"
        },
        {
            "name": "N-Systane",
            "dosage": "100,000 UI/ml",
            "forme": "Sirop",
            "usage": "ESSENTIAL",
            "category": "Antifungal"
        },
        {
            "name": "N-Systane",
            "dosage": "10,000 UI",
            "forme": "Ovule",
            "usage": "ESSENTIAL",
            "category": "Antifungal"
        },
        {
            "name": "Activator",
            "dosage": "200 mg",
            "forme": "Comprime",
            "usage": "LIFESAVING",
            "category": "Cardiovascular",
            "description": "Used for cardiac emergencies"
        },
        {
            "name": "Activator",
            "dosage": "200 mg/5ml",
            "forme": "Suspension",
            "usage": "LIFESAVING",
            "category": "Cardiovascular"
        },
        {
            "name": "Activator",
            "dosage": "250 mg/ml",
            "forme": "Injectable",
            "usage": "LIFESAVING",
            "category": "Cardiovascular"
        }
    ]
    
    # Create database
    db = MedicationVectorDB()
    db.add_medications(sample_medications)
    
    return db


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample database
    db = create_sample_database()
    
    # Test queries
    print("\n=== Database Statistics ===")
    stats = db.get_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n=== Search Test: 'diabetes' ===")
    results = db.search("diabetes", top_k=3)
    for r in results:
        print(f"- {r['name']} ({r['dosage']}) - Score: {r['similarity_score']:.3f}")
    
    print("\n=== Search by Category: 'Antifungal' ===")
    results = db.search_by_category("Antifungal", top_k=3)
    for r in results:
        print(f"- {r['name']} ({r['dosage']}) - {r['forme']}")
    
    print("\n=== Get Alternatives for 'N-Systane' ===")
    alternatives = db.get_alternatives("N-Systane", top_k=3)
    for r in alternatives:
        print(f"- {r['name']} ({r['dosage']}) - Score: {r['similarity_score']:.3f}")
