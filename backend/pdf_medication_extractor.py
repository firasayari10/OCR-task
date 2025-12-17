"""
PDF Medication Extractor

Extracts medication information from PDF files containing essential medication lists.
"""

import re
import logging
from typing import List, Dict, Any
import pdfplumber
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFMedicationExtractor:
    """Extract medication data from PDF files."""
    
    def __init__(self):
        self.usage_keywords = {
            'ESSENTIAL': ['ESSENTIAL', 'ESSENTIEL', 'X'],
            'RECOMMENDED': ['RECOMMENDED', 'RECOMMANDE'],
            'LIFESAVING': ['LIFESAVING', 'VITAL', 'SAUVETAGE']
        }
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract medication information from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            List of medication dictionaries
        """
        medications = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing PDF: {pdf_path}")
                logger.info(f"Total pages: {len(pdf.pages)}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}...")
                    
                    # Extract tables from page
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        logger.info(f"  Found table {table_idx + 1} with {len(table)} rows")
                        page_meds = self._parse_table(table, page_num)
                        medications.extend(page_meds)
                        logger.info(f"  Extracted {len(page_meds)} medications from table")
                
                logger.info(f"Total medications extracted: {len(medications)}")
                
        except Exception as e:
            logger.error(f"Error processing PDF: {e}", exc_info=True)
        
        return medications
    
    def _parse_table(self, table: List[List[str]], page_num: int) -> List[Dict[str, Any]]:
        """Parse a table and extract medication information."""
        medications = []
        
        if not table or len(table) < 2:
            return medications
        
        # Find header row
        header_row = None
        for i, row in enumerate(table[:5]):  # Check first 5 rows for header
            if row and any(cell and ('DCI' in str(cell).upper() or 'FORME' in str(cell).upper()) for cell in row):
                header_row = i
                break
        
        if header_row is None:
            logger.warning(f"Could not find header row on page {page_num}")
            return medications
        
        # Parse header to get column indices
        headers = table[header_row]
        col_mapping = self._map_columns(headers)
        
        if not col_mapping:
            logger.warning(f"Could not map columns on page {page_num}")
            return medications
        
        # Parse data rows
        for row_idx in range(header_row + 1, len(table)):
            row = table[row_idx]
            
            if not row or len(row) < 3:
                continue
            
            # Skip empty rows
            if not any(cell and str(cell).strip() for cell in row):
                continue
            
            try:
                med = self._parse_medication_row(row, col_mapping, page_num)
                if med:
                    medications.append(med)
            except Exception as e:
                logger.debug(f"Error parsing row {row_idx}: {e}")
        
        return medications
    
    def _map_columns(self, headers: List[str]) -> Dict[str, int]:
        """Map column names to indices."""
        col_mapping = {}
        
        for i, header in enumerate(headers):
            if not header:
                continue
            
            header_upper = str(header).upper().strip()
            
            # Map columns based on keywords
            if 'DCI' in header_upper or 'NOM' in header_upper or 'NO' in header_upper:
                col_mapping['name'] = i
            elif 'FORME' in header_upper:
                col_mapping['forme'] = i
            elif 'DOSAGE' in header_upper or 'DOSE' in header_upper:
                col_mapping['dosage'] = i
            elif 'ESSENTIEL' in header_upper or 'ESSENTIAL' in header_upper:
                col_mapping['essential'] = i
            elif 'RECOMMANDE' in header_upper or 'RECOMMENDED' in header_upper:
                col_mapping['recommended'] = i
            elif 'VITAL' in header_upper or 'LIFESAVING' in header_upper or 'SAUVETAGE' in header_upper:
                col_mapping['lifesaving'] = i
        
        return col_mapping
    
    def _parse_medication_row(self, row: List[str], col_mapping: Dict[str, int], page_num: int) -> Dict[str, Any]:
        """Parse a single medication row."""
        # Extract name
        name_idx = col_mapping.get('name', 1)
        name = str(row[name_idx]).strip() if name_idx < len(row) and row[name_idx] else None
        
        if not name or name in ['', 'None', '-']:
            return None
        
        # Extract forme
        forme_idx = col_mapping.get('forme')
        forme = str(row[forme_idx]).strip() if forme_idx is not None and forme_idx < len(row) and row[forme_idx] else "Unknown"
        
        # Extract dosage
        dosage_idx = col_mapping.get('dosage')
        dosage = str(row[dosage_idx]).strip() if dosage_idx is not None and dosage_idx < len(row) and row[dosage_idx] else ""
        
        # Determine usage type
        usage = "UNKNOWN"
        
        # Check each usage column
        if 'lifesaving' in col_mapping:
            idx = col_mapping['lifesaving']
            if idx < len(row) and row[idx] and self._is_marked(str(row[idx])):
                usage = "LIFESAVING"
        
        if 'essential' in col_mapping and usage == "UNKNOWN":
            idx = col_mapping['essential']
            if idx < len(row) and row[idx] and self._is_marked(str(row[idx])):
                usage = "ESSENTIAL"
        
        if 'recommended' in col_mapping and usage == "UNKNOWN":
            idx = col_mapping['recommended']
            if idx < len(row) and row[idx] and self._is_marked(str(row[idx])):
                usage = "RECOMMENDED"
        
        # Try to infer category from name or other fields
        category = self._infer_category(name, forme, dosage)
        
        medication = {
            "name": name,
            "dosage": dosage,
            "forme": forme,
            "usage": usage,
            "category": category,
            "page": page_num
        }
        
        return medication
    
    def _is_marked(self, cell: str) -> bool:
        """Check if a cell is marked (has X or checkmark)."""
        cell_upper = cell.upper().strip()
        return cell_upper in ['X', '✓', '✔', 'YES', 'OUI', '1', 'TRUE']
    
    def _infer_category(self, name: str, forme: str, dosage: str) -> str:
        """Infer medication category from name and other fields."""
        name_lower = name.lower()
        
        # Common medication categories
        if any(word in name_lower for word in ['insulin', 'metformin', 'glicl', 'diabet']):
            return "Diabetes Management"
        elif any(word in name_lower for word in ['nystatin', 'fluconazole', 'fungal', 'mycotic']):
            return "Antifungal"
        elif any(word in name_lower for word in ['amoxicillin', 'penicillin', 'cillin', 'mycin', 'ciprofloxacin']):
            return "Antibiotic"
        elif any(word in name_lower for word in ['paracetamol', 'ibuprofen', 'aspirin', 'pain', 'doleur']):
            return "Analgesic"
        elif any(word in name_lower for word in ['cardiac', 'heart', 'cardio', 'atenolol', 'amlodipine']):
            return "Cardiovascular"
        elif any(word in name_lower for word in ['vitamin', 'vitamine']):
            return "Vitamin/Supplement"
        elif any(word in name_lower for word in ['antihistamine', 'cetirizine', 'loratadine', 'allerg']):
            return "Antihistamine"
        else:
            return "General"
    
    def extract_and_deduplicate(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract medications and remove duplicates."""
        medications = self.extract_from_pdf(pdf_path)
        
        # Deduplicate based on name + dosage + forme
        unique_meds = {}
        for med in medications:
            key = f"{med['name']}|{med['dosage']}|{med['forme']}"
            if key not in unique_meds:
                unique_meds[key] = med
            else:
                # Keep the one with more specific usage
                if med['usage'] != "UNKNOWN":
                    unique_meds[key] = med
        
        return list(unique_meds.values())


def process_pdf_to_database(pdf_path: str, db_path: str = "medication_db"):
    """
    Extract medications from PDF and add to vector database.
    
    Args:
        pdf_path: Path to PDF file
        db_path: Path to database directory
    """
    from medication_vector_db import MedicationVectorDB
    
    # Extract medications
    extractor = PDFMedicationExtractor()
    medications = extractor.extract_and_deduplicate(pdf_path)
    
    logger.info(f"Extracted {len(medications)} unique medications from PDF")
    
    # Add to database
    db = MedicationVectorDB(db_path)
    db.add_medications(medications)
    
    # Show statistics
    stats = db.get_stats()
    logger.info(f"Database statistics: {stats}")
    
    return db


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    # pdf_path = "essential_medications.pdf"
    # if Path(pdf_path).exists():
    #     db = process_pdf_to_database(pdf_path)
    # else:
    #     print(f"PDF file not found: {pdf_path}")
    
    print("PDF Medication Extractor ready!")
    print("Usage:")
    print("  from pdf_medication_extractor import process_pdf_to_database")
    print("  db = process_pdf_to_database('your_medication_list.pdf')")
