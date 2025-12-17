"""
Build Medication Vector Database from PDF

This script processes a PDF containing essential medication lists
and creates a searchable vector database.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def build_database_from_pdf(pdf_path: str, db_path: str = "medication_db"):
    """
    Build vector database from PDF file.
    
    Args:
        pdf_path: Path to PDF file with medication list
        db_path: Directory to store the database
    """
    from pdf_medication_extractor import PDFMedicationExtractor, process_pdf_to_database
    from medication_vector_db import MedicationVectorDB
    import json
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return False
    
    logger.info(f"Processing PDF: {pdf_path}")
    logger.info(f"Database will be saved to: {db_path}")
    
    try:
        # Extract and build database
        db = process_pdf_to_database(str(pdf_file), db_path)
        
        # Show statistics
        stats = db.get_stats()
        logger.info("\n" + "="*60)
        logger.info("DATABASE STATISTICS")
        logger.info("="*60)
        logger.info(f"Total medications: {stats['total_medications']}")
        
        logger.info("\nCategories:")
        for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {cat}: {count}")
        
        logger.info("\nUsage Types:")
        for usage, count in sorted(stats['usage_types'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {usage}: {count}")
        
        logger.info("\nForms:")
        for forme, count in sorted(stats['forms'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {forme}: {count}")
        
        logger.info("="*60)
        
        # Test search
        logger.info("\n" + "="*60)
        logger.info("TESTING DATABASE - Sample Searches")
        logger.info("="*60)
        
        test_queries = ["diabetes", "antifungal", "cardiac", "vitamin"]
        for query in test_queries:
            logger.info(f"\nSearch: '{query}'")
            results = db.search(query, top_k=3)
            for r in results[:3]:
                logger.info(f"  - {r['name']} ({r['dosage']}) - {r['forme']} - Score: {r['similarity_score']:.3f}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ Database created successfully!")
        logger.info(f"✓ Location: {db_path}")
        logger.info(f"✓ Total medications: {len(db.metadata)}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error building database: {e}", exc_info=True)
        return False


def test_database(db_path: str = "medication_db"):
    """Test the vector database with sample queries."""
    from medication_vector_db import MedicationVectorDB
    
    logger.info(f"Loading database from: {db_path}")
    
    try:
        db = MedicationVectorDB(db_path)
        
        if len(db.metadata) == 0:
            logger.warning("Database is empty!")
            return
        
        logger.info(f"Database loaded: {len(db.metadata)} medications")
        
        # Interactive testing
        print("\n" + "="*60)
        print("MEDICATION DATABASE - Interactive Search")
        print("="*60)
        print("Enter medication names to search (or 'quit' to exit)")
        print("Examples: 'insulin', 'amoxicillin', 'paracetamol'")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("Search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                results = db.search_by_name(query, top_k=5)
                
                if results:
                    print(f"\nFound {len(results)} results for '{query}':\n")
                    for i, r in enumerate(results, 1):
                        print(f"{i}. {r['name']}")
                        print(f"   Dosage: {r['dosage']}")
                        print(f"   Form: {r['forme']}")
                        print(f"   Usage: {r['usage']}")
                        print(f"   Category: {r['category']}")
                        print(f"   Similarity: {r['similarity_score']:.3f}")
                        print()
                else:
                    print(f"No results found for '{query}'\n")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}\n")
        
        print("\nGoodbye!")
        
    except Exception as e:
        logger.error(f"Error testing database: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build medication vector database from PDF")
    parser.add_argument("--pdf", type=str, help="Path to PDF file with medication list")
    parser.add_argument("--db", type=str, default="medication_db", help="Database directory (default: medication_db)")
    parser.add_argument("--test", action="store_true", help="Test the database interactively")
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode
        test_database(args.db)
    elif args.pdf:
        # Build from PDF
        success = build_database_from_pdf(args.pdf, args.db)
        sys.exit(0 if success else 1)
    else:
        # No arguments - show help
        parser.print_help()
        print("\nExamples:")
        print("  # Build database from PDF:")
        print("  python build_medication_db.py --pdf essential_medications.pdf")
        print()
        print("  # Test database interactively:")
        print("  python build_medication_db.py --test")
        print()
        print("  # Build with custom database path:")
        print("  python build_medication_db.py --pdf medications.pdf --db my_custom_db")
