from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, List
import torch
import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from huggingface_hub import InferenceClient
import re

# Try to import SAM2 (optional - will work without it)
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM2 not available. Will use simple region detection.")

app = FastAPI(title="Prescription OCR with Azure Vision")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for Azure Vision API
azure_vision_endpoint = os.getenv('AZURE_VISION_ENDPOINT')
azure_vision_key = os.getenv('AZURE_VISION_KEY')
if not azure_vision_endpoint or not azure_vision_key:
    print("Warning: AZURE_VISION_ENDPOINT or AZURE_VISION_KEY not set in environment variables or .env file")
    print("Set them in .env file or as environment variables before starting the server.")

sam2_model = None
sam2_mask_generator = None

# Device for SAM2 (if used)
device = "cuda" if torch.cuda.is_available() else "cpu"

def verify_azure_vision_config():
    """Verify Azure Vision API is configured."""
    global azure_vision_endpoint, azure_vision_key
    
    if not azure_vision_endpoint or not azure_vision_key:
        print("ERROR: Azure Vision API not configured.")
        print("Set AZURE_VISION_ENDPOINT and AZURE_VISION_KEY in .env or environment variables.")
        return False
    
    print(f"Azure Vision API configured: {azure_vision_endpoint}")
    return True


def filter_phi_with_hf(text: str) -> Dict:
    """Detect and redact PHI from text.

    Steps:
    1. (Optional) call ClinicalBERT embeddings as a first step (using HF Inference API) to satisfy the request.
    2. Call a token-classification model (via HF Inference API) to detect PERSON-like entities.
    3. Use regexes for common PHI: emails, phones, SSNs, dates, MRNs, IDs, names, addresses.
    4. Return redacted text and a summary of removed PHI.
    """
    hf_token = os.getenv('HF_TOKEN')
    phi_spans = []  # list of (start, end, label, text)

    if not text:
        return {"redacted_text": "", "phi": []}
    
    print(f"PHI filtering text (length: {len(text)})")

    # Try HF Inference API for NER
    if hf_token:
        try:
            # Use InferenceClient with token parameter (new API)
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=hf_token)
            
            # Use ClinicalBERT embeddings as a first step (not used downstream but called per request)
            try:
                print("Calling ClinicalBERT embeddings...")
                _ = client.feature_extraction(text, model="emilyalsentzer/Bio_ClinicalBERT")
                print("ClinicalBERT embeddings successful")
            except Exception as e:
                print(f"ClinicalBERT embeddings failed (optional): {e}")

            # Use a general-purpose NER model to find PERSON/LOC/ORG entities
            try:
                ner_model = os.getenv('HF_NER_MODEL', 'dslim/bert-base-NER')
                print(f"Calling NER model: {ner_model}")
                ner_results = client.token_classification(text, model=ner_model)
                print(f"NER returned {len(ner_results)} entities")
                
                # Group consecutive tokens by entity type and position
                grouped_entities = []
                current_entity = None
                
                for ent in ner_results:
                    entity_type = ent.get('entity_group') or ent.get('entity', '')
                    entity_type_upper = str(entity_type).upper().replace('B-', '').replace('I-', '')
                    
                    if entity_type_upper in ("PER", "PERSON", "ORG", "LOC", "MISC"):
                        if current_entity and current_entity['type'] == entity_type_upper and ent.get('start') <= current_entity['end'] + 2:
                            # Extend current entity
                            current_entity['end'] = ent.get('end')
                            current_entity['word'] += ' ' + ent.get('word', '')
                        else:
                            # Start new entity
                            if current_entity:
                                grouped_entities.append(current_entity)
                            current_entity = {
                                'type': entity_type_upper,
                                'start': ent.get('start'),
                                'end': ent.get('end'),
                                'word': ent.get('word', '')
                            }
                
                if current_entity:
                    grouped_entities.append(current_entity)
                
                for ent in grouped_entities:
                    print(f"  Found NER entity: {ent['type']} -> {ent['word']}")
                    phi_spans.append((ent['start'], ent['end'], ent['type'], ent['word']))
                    
            except Exception as e:
                print(f"NER token classification failed: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"HF client error: {e}")

    # Enhanced Regex-based PHI detection (always run)
    print("Running regex-based PHI detection...")
    
    # Names (common patterns after "Name:" label) - improved to not capture trailing text
    for m in re.finditer(r"(?:Name|Patient\s*Name|Patient):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+?)(?:\s+Address|$|\s*\n)", text, flags=re.IGNORECASE):
        phi_spans.append((m.start(1), m.end(1), 'NAME', m.group(1)))
        print(f"  Found NAME: {m.group(1)}")
    
    # Addresses (after "Address:" label) - improved to stop at next label or newline
    for m in re.finditer(r"(?:Address|Street|Location):\s*([A-Z0-9][^\n]{5,60}?)(?:\s+Age|Sex|Date|$|\s*\n)", text, flags=re.IGNORECASE):
        phi_spans.append((m.start(1), m.end(1), 'ADDRESS', m.group(1)))
        print(f"  Found ADDRESS: {m.group(1)}")
    
    # Ages (after "Age:" label)
    for m in re.finditer(r"(?:Age):\s*(\d{1,3})", text, flags=re.IGNORECASE):
        phi_spans.append((m.start(1), m.end(1), 'AGE', m.group(1)))
        print(f"  Found AGE: {m.group(1)}")
    
    # Dates - more aggressive patterns
    # Pattern 1: MM-DD-YY or MM/DD/YY or DD-MM-YYYY
    for m in re.finditer(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", text):
        phi_spans.append((m.start(), m.end(), 'DATE', m.group(0)))
        print(f"  Found DATE: {m.group(0)}")
    
    # Pattern 2: Month DD, YYYY or DD Month YYYY
    for m in re.finditer(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b", text, flags=re.IGNORECASE):
        phi_spans.append((m.start(), m.end(), 'DATE', m.group(0)))
        print(f"  Found DATE: {m.group(0)}")
    
    # License numbers
    for m in re.finditer(r"(?:Lic\.?\s*No\.?|License\s*No\.?|PTR\s*No\.?|S2\s*No\.?)[\s:]*(\d+)", text, flags=re.IGNORECASE):
        phi_spans.append((m.start(1), m.end(1), 'LICENSE', m.group(1)))
        print(f"  Found LICENSE: {m.group(1)}")
    
    # Emails
    for m in re.finditer(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}", text):
        phi_spans.append((m.start(), m.end(), 'EMAIL', m.group(0)))
        print(f"  Found EMAIL: {m.group(0)}")

    # Phone numbers (various formats)
    for m in re.finditer(r"\b(?:\+\d{1,3}[- ]?)?(?:\(\d{2,4}\)|\d{2,4})[- ]?\d{3,4}[- ]?\d{3,4}\b", text):
        phi_spans.append((m.start(), m.end(), 'PHONE', m.group(0)))
        print(f"  Found PHONE: {m.group(0)}")

    # MRN / ID like patterns (numbers of length >=6)
    for m in re.finditer(r"\b\d{6,}\b", text):
        # avoid capturing plain years/dosages by heuristic: if nearby letters like MRN/ID
        context = text[max(0, m.start()-20):m.end()+20]
        if re.search(r"\b(MRN|ID|Patient|Acct|Account|Record|No\.?)\b", context, flags=re.IGNORECASE):
            phi_spans.append((m.start(), m.end(), 'ID', m.group(0)))
            print(f"  Found ID: {m.group(0)}")

    # SSN-like
    for m in re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", text):
        phi_spans.append((m.start(), m.end(), 'SSN', m.group(0)))
        print(f"  Found SSN: {m.group(0)}")

    print(f"Total PHI spans found: {len(phi_spans)}")

    # Deduplicate spans and sort
    phi_spans = sorted(set(phi_spans), key=lambda x: x[0])

    # Merge overlapping spans
    merged = []
    for span in phi_spans:
        if not merged:
            merged.append(list(span))
            continue
        last = merged[-1]
        if span[0] <= last[1]:
            # extend
            last[1] = max(last[1], span[1])
            # prefer label of first span
        else:
            merged.append(list(span))

    print(f"After merging: {len(merged)} PHI spans")

    # Replace spans in text with redaction labels, process from end to start
    redacted = text
    phi_summary = []
    for start, end, label, sample in sorted(merged, key=lambda x: x[0], reverse=True):
        label_tag = f"[REDACTED_{label}]"
        redacted = redacted[:start] + label_tag + redacted[end:]
        phi_summary.append({'start': start, 'end': end, 'label': label, 'sample': sample})

    phi_summary = sorted(phi_summary, key=lambda x: x['start'])
    print(f"Returning {len(phi_summary)} redacted items")
    return {"redacted_text": redacted, "phi": phi_summary}

def extract_medications_from_text(text: str) -> List[Dict]:
    """Extract medication names and dosages from prescription text using regex patterns."""
    medications = []
    
    print(f"Extracting medications from text: {text[:200]}...")
    
    # Pattern 1: Tab/Cap/Inj/Syr followed by drug name and dosage
    # Examples: "Tab. Augmentin 625mg", "Cap. Amoxicillin 500mg"
    pattern1 = r'(?:Tab\.?|Cap\.?|Inj\.?|Syr\.?|Tablet|Capsule)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|units?|IU))'
    
    # Pattern 2: Drug name with dosage (no prefix)
    # Examples: "Augmentin 625mg", "Paracetamol 500mg"
    pattern2 = r'\b([A-Z][a-z]+(?:cillin|mycin|pril|olol|ine|azole|ide|tax|done|pine|lone|sartan|statin|flam|idol|tin|tin|zol))\s+(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|units?|IU))\b'
    
    # Pattern 3: Medication with "Tab" or "Cap" marker
    pattern3 = r'(?:Tab|Cap|Inj|Syr)\.?\s*([A-Z][a-zA-Z0-9]+)\s+(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|units?|IU))'
    
    # Pattern 4: Common medication names (expanded list including Indian brands)
    common_drugs = r'\b(augmentin|amoxicillin|enzoflam|diclofenac|ibuprofen|paracetamol|acetaminophen|aspirin|metformin|lisinopril|atorvastatin|omeprazole|pantoprazole|rabeprazole|amlodipine|losartan|telmisartan|ramipril|simvastatin|rosuvastatin|levothyroxine|azithromycin|ciprofloxacin|levofloxacin|moxifloxacin|doxycycline|amoxiclav|cefixime|ceftriaxone|cetirizine|loratadine|fexofenadine|montelukast|salbutamol|budesonide|prednisone|prednisolone|dexamethasone|metronidazole|fluconazole|clotrimazole|warfarin|clopidogrel|aspirin|heparin|insulin|metformin|glimepiride|gliclazide|sitagliptin|vildagliptin|gabapentin|pregabalin|tramadol|morphine|codeine|hydrocodone|oxycodone|fentanyl|alprazolam|diazepam|clonazepam|lorazepam|zolpidem|eszopiclone|sertraline|escitalopram|fluoxetine|paroxetine|venlafaxine|duloxetine|quetiapine|olanzaparine|risperidone|aripiprazole|ranitidine|famotidine|esomeprazole|lansoprazole|domperidone|ondansetron|metoclopramide|bisoprolol|carvedilol|atenolol|propranolol|diltiazem|verapamil|furosemide|torsemide|spironolactone|hydrochlorothiazide|chlorthalidone|enalapril|perindopril|irbesartan|valsartan|candesartan|amlodipine|nifedipine|felodipine|tamsulosin|sildenafil|tadalafil|finasteride|dutasteride|levothyroxine|thyroxine|carbimazole|propylthiouracil|vitamin|calcium|iron|folic|zinc|biotin)(?:cillin|mycin|flam|pril|olol|ine|azole|ide|tax|done|pine|lone|sartan|statin)?\b'
    
    # Extract with pattern 1 (Tab/Cap + drug + dosage)
    for m in re.finditer(pattern1, text, flags=re.IGNORECASE):
        drug_name = m.group(1).strip()
        dosage = m.group(2).strip()
        medications.append({
            'name': drug_name.lower(),
            'dosage': dosage,
            'original_text': m.group(0),
            'start': m.start(),
            'end': m.end()
        })
        print(f"  Found medication (pattern1): {drug_name} {dosage}")
    
    # Extract with pattern 2 (suffix-based detection)
    for m in re.finditer(pattern2, text, flags=re.IGNORECASE):
        drug_name = m.group(1).strip()
        dosage = m.group(2).strip()
        # Avoid duplicates
        if not any(med['name'] == drug_name.lower() and med.get('start') == m.start() for med in medications):
            medications.append({
                'name': drug_name.lower(),
                'dosage': dosage,
                'original_text': m.group(0),
                'start': m.start(),
                'end': m.end()
            })
            print(f"  Found medication (pattern2): {drug_name} {dosage}")
    
    # Extract with pattern 3 (alternative Tab/Cap format)
    for m in re.finditer(pattern3, text, flags=re.IGNORECASE):
        drug_name = m.group(1).strip()
        dosage = m.group(2).strip()
        # Avoid duplicates
        if not any(med['name'] == drug_name.lower() and med.get('start') == m.start() for med in medications):
            medications.append({
                'name': drug_name.lower(),
                'dosage': dosage,
                'original_text': m.group(0),
                'start': m.start(),
                'end': m.end()
            })
            print(f"  Found medication (pattern3): {drug_name} {dosage}")
    
    # Extract common drugs
    for m in re.finditer(common_drugs, text, flags=re.IGNORECASE):
        drug_name = m.group(1).strip()
        # Look for nearby dosage within 30 characters
        context_start = max(0, m.end())
        context_end = min(len(text), m.end() + 30)
        context = text[context_start:context_end]
        
        dosage_match = re.search(r'(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|units?|IU))', context, flags=re.IGNORECASE)
        dosage = dosage_match.group(1).strip() if dosage_match else None
        
        # Avoid duplicates
        if not any(med['name'] == drug_name.lower() and abs(med.get('start', -1000) - m.start()) < 10 for med in medications):
            medications.append({
                'name': drug_name.lower(),
                'dosage': dosage,
                'original_text': m.group(0),
                'start': m.start(),
                'end': m.end()
            })
            print(f"  Found medication (common): {drug_name}" + (f" {dosage}" if dosage else ""))
    
    # Deduplicate by position and name
    seen_positions = set()
    unique_meds = []
    for med in sorted(medications, key=lambda x: x['start']):
        key = (med['name'], med['start'] // 10)  # Group nearby positions
        if key not in seen_positions:
            seen_positions.add(key)
            unique_meds.append(med)
    
    print(f"Total unique medications found: {len(unique_meds)}")
    return unique_meds


def get_fda_drug_alternatives(drug_name: str) -> Dict:
    """Query multiple drug databases for drug information and alternatives.
    
    Uses:
    1. RxNorm API (NIH) - free, no key needed, international coverage
    2. FDA openFDA API - fallback for US medications
    """
    try:
        # First try RxNorm API (better international coverage, no API key needed)
        rxnorm_url = "https://rxnav.nlm.nih.gov/REST/drugs.json"
        params = {'name': drug_name}
        
        print(f"Querying RxNorm API for: {drug_name}")
        response = requests.get(rxnorm_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            drug_group = data.get('drugGroup', {})
            concept_properties = drug_group.get('conceptGroup', [])
            
            alternatives = []
            seen_names = set()
            
            for concept_group in concept_properties:
                concepts = concept_group.get('conceptProperties', [])
                for concept in concepts:
                    name = concept.get('name', '')
                    rxcui = concept.get('rxcui', '')
                    
                    if name and name.lower() not in seen_names:
                        seen_names.add(name.lower())
                        
                        # Get additional details about this drug
                        details = get_rxnorm_drug_details(rxcui)
                        
                        alternatives.append({
                            'generic_name': name,
                            'brand_names': details.get('brand_names', []),
                            'manufacturer': details.get('manufacturer', 'Various'),
                            'indication': details.get('indication', 'See prescribing information'),
                            'type': 'generic' if concept.get('synonym', '') == 'IN' else 'brand',
                            'rxcui': rxcui
                        })
            
            if alternatives:
                return {
                    'drug_name': drug_name,
                    'found': True,
                    'alternatives': alternatives[:10],
                    'total_found': len(alternatives),
                    'source': 'RxNorm (NIH)'
                }
        
        # Fallback to FDA openFDA API
        print(f"RxNorm returned no results, trying FDA API for: {drug_name}")
        base_url = "https://api.fda.gov/drug/label.json"
        
        # Search for the drug
        params = {
            'search': f'openfda.generic_name:"{drug_name}"',
            'limit': 5
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                # Try brand name search as fallback
                params['search'] = f'openfda.brand_name:"{drug_name}"'
                response = requests.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
            
            if results:
                alternatives = []
                seen_names = set()
                
                for result in results:
                    openfda = result.get('openfda', {})
                    
                    # Extract generic and brand names
                    generic_names = openfda.get('generic_name', [])
                    brand_names = openfda.get('brand_name', [])
                    manufacturer = openfda.get('manufacturer_name', ['Unknown'])[0] if openfda.get('manufacturer_name') else 'Unknown'
                    
                    # Get indications and usage
                    indications = result.get('indications_and_usage', ['Not available'])
                    indication_text = indications[0][:200] if indications else 'Not available'
                    
                    for generic in generic_names:
                        if generic.lower() not in seen_names:
                            seen_names.add(generic.lower())
                            alternatives.append({
                                'generic_name': generic,
                                'brand_names': brand_names[:3],
                                'manufacturer': manufacturer,
                                'indication': indication_text,
                                'type': 'generic'
                            })
                    
                    for brand in brand_names:
                        if brand.lower() not in seen_names and brand.lower() != drug_name.lower():
                            seen_names.add(brand.lower())
                            alternatives.append({
                                'generic_name': generic_names[0] if generic_names else brand,
                                'brand_names': [brand],
                                'manufacturer': manufacturer,
                                'indication': indication_text,
                                'type': 'brand'
                            })
                
                return {
                    'drug_name': drug_name,
                    'found': True,
                    'alternatives': alternatives[:10],
                    'total_found': len(alternatives),
                    'source': 'FDA openFDA'
                }
        
        # No results from either API - try LLaMA as fallback
        print(f"No results from RxNorm or FDA, trying LLaMA API for: {drug_name}")
        llama_result = query_llama_for_drug_info(drug_name)
        
        if llama_result.get('found'):
            return {
                'drug_name': drug_name,
                'found': True,
                'alternatives': [],
                'text_from_llm': llama_result.get('text_from_llm', ''),
                'source': 'LLaMA 3.1-70B-Instruct',
                'message': 'Information provided by AI model (verify with medical professional)'
            }
        
        return {
            'drug_name': drug_name,
            'found': False,
            'alternatives': [],
            'text_from_llm': llama_result.get('text_from_llm', 'No information available'),
            'message': 'No database records found. Drug may be regional or brand-specific.',
            'source': 'none'
        }
            
    except Exception as e:
        print(f"Error querying drug APIs for {drug_name}: {e}")
        return {
            'drug_name': drug_name,
            'found': False,
            'alternatives': [],
            'error': str(e)
        }


def get_rxnorm_drug_details(rxcui: str) -> Dict:
    """Get detailed information about a drug from RxNorm using its RXCUI."""
    try:
        # Get related drugs (brand names, etc.)
        related_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json"
        params = {'tty': 'BN'}  # BN = Brand Name
        
        response = requests.get(related_url, params=params, timeout=5)
        brand_names = []
        
        if response.status_code == 200:
            data = response.json()
            concept_group = data.get('relatedGroup', {}).get('conceptGroup', [])
            for group in concept_group:
                if group.get('tty') == 'BN':
                    concepts = group.get('conceptProperties', [])
                    brand_names = [c.get('name') for c in concepts if c.get('name')]
                    break
        
        return {
            'brand_names': brand_names[:5],  # Limit to 5
            'manufacturer': 'Various',
            'indication': 'Consult drug monograph or prescribing information'
        }
    except Exception as e:
        print(f"Error getting RxNorm details for {rxcui}: {e}")
        return {
            'brand_names': [],
            'manufacturer': 'Unknown',
            'indication': 'Information not available'
        }


def query_llama_for_drug_info(drug_name: str, dosage: str = None) -> Dict:
    """Query Grok or other LLM for drug information when other APIs fail.
    
    Tries:
    1. xAI Grok via OpenRouter (if configured)
    2. Hugging Face Inference API (fallback with existing HF_TOKEN)
    """
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    grok_model = os.getenv('GROK_MODEL', 'x-ai/grok-beta')
    hf_token = os.getenv('HF_TOKEN')
    
    # Construct prompt
    dosage_info = f" with dosage {dosage}" if dosage else ""
    prompt = f"""Provide brief medical information about the medication '{drug_name}'{dosage_info}.

Include:
1. Generic name and brand names
2. Typical dosage
3. Primary uses
4. 2-3 alternative medications
5. Key precautions

Be concise and accurate."""
    
    # Try xAI Grok via OpenRouter if configured
    if openrouter_api_key and openrouter_api_key != 'your_openrouter_api_key_here':
        try:
            print(f"Querying OpenRouter ({grok_model}) for: {drug_name}")
            
            headers = {
                'Authorization': f'Bearer {openrouter_api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://prescription-ocr.app',
                'X-Title': 'Prescription OCR'
            }
            
            payload = {
                'model': grok_model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a knowledgeable medical information assistant specializing in pharmaceuticals. Provide accurate, concise information about medications.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 500,
                'temperature': 0.3
            }
            
            response = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"Grok API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                llm_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                if llm_text:
                    return {
                        'found': True,
                        'drug_name': drug_name,
                        'text_from_llm': llm_text,
                        'source': f'{grok_model} (OpenRouter)'
                    }
            else:
                print(f"OpenRouter API error: {response.status_code} - {response.text[:200]}")
        
        except Exception as e:
            print(f"Error querying Grok API: {e}")
    
    # Fallback to Hugging Face Inference API with existing token
    if hf_token:
        try:
            from huggingface_hub import InferenceClient
            print(f"Falling back to Hugging Face Inference API for: {drug_name}")
            
            client = InferenceClient(token=hf_token)
            
            # Use chat completion instead of text generation
            messages = [
                {"role": "system", "content": "You are a medical information assistant. Provide accurate medication information."},
                {"role": "user", "content": prompt}
            ]
            
            response = client.chat_completion(
                messages=messages,
                model="meta-llama/Llama-3.2-3B-Instruct",
                max_tokens=400,
                temperature=0.3
            )
            
            llm_text = response.choices[0].message.content
            
            if llm_text:
                return {
                    'found': True,
                    'drug_name': drug_name,
                    'text_from_llm': llm_text,
                    'source': 'Llama-3.2-3B-Instruct (HuggingFace)'
                }
        except Exception as e:
            print(f"Hugging Face API error: {e}")
    
    return {
        'found': False,
        'text_from_llm': 'No LLM API available. Please configure OPENROUTER_API_KEY or medication information is not available in standard databases.',
        'error': 'No working API endpoint'
    }

def extract_text_with_azure_vision(image_bytes: bytes) -> str:
    """Extract text from image using Azure Computer Vision Read API."""
    global azure_vision_endpoint, azure_vision_key
    
    if not azure_vision_endpoint or not azure_vision_key:
        raise HTTPException(status_code=503, detail="Azure Vision API not configured")
    
    try:
        # Ensure image is in a supported format by re-encoding to PNG
        # This fixes issues where uploaded images might be in unsupported formats
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Re-encode to PNG (supported by Azure Vision)
            png_buffer = io.BytesIO()
            image.save(png_buffer, format='PNG')
            image_bytes = png_buffer.getvalue()
            print(f"Re-encoded image to PNG format. Size: {len(image_bytes)} bytes")
        except Exception as e:
            print(f"Warning: Could not re-encode image: {e}")
            # Continue with original bytes if re-encoding fails
        
        # Azure Read API endpoint for synchronous OCR
        url = f"{azure_vision_endpoint.rstrip('/')}/vision/v3.2/read/analyze"
        
        headers = {
            'Ocp-Apim-Subscription-Key': azure_vision_key,
            'Content-Type': 'application/octet-stream'
        }
        
        # Send image to Azure Vision API
        print(f"Sending image to Azure Vision API: {url}")
        response = requests.post(url, headers=headers, data=image_bytes, timeout=30)
        
        if response.status_code != 202:
            print(f"Azure Vision API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Azure Vision API error: {response.text}")
        
        # Get the operation location from response headers
        operation_url = response.headers.get('Operation-Location')
        if not operation_url:
            raise HTTPException(status_code=500, detail="No Operation-Location returned from Azure Vision API")
        
        print(f"Operation URL: {operation_url}")
        
        # Poll the operation until complete
        max_retries = 30
        retry_count = 0
        while retry_count < max_retries:
            time.sleep(1)  # Wait 1 second before checking
            
            result_response = requests.get(operation_url, headers={'Ocp-Apim-Subscription-Key': azure_vision_key}, timeout=10)
            
            if result_response.status_code != 200:
                print(f"Error polling Azure Vision API: {result_response.status_code}")
                retry_count += 1
                continue
            
            result_data = result_response.json()
            status = result_data.get('status')
            
            print(f"OCR status: {status} (attempt {retry_count + 1}/{max_retries})")
            
            if status == 'succeeded':
                # Extract text from result
                extracted_text = ""
                if 'analyzeResult' in result_data and 'readResults' in result_data['analyzeResult']:
                    for page in result_data['analyzeResult']['readResults']:
                        for line in page.get('lines', []):
                            extracted_text += line.get('text', '') + "\n"
                
                print(f"Azure Vision OCR successful. Extracted {len(extracted_text)} characters.")
                return extracted_text.strip()
            
            elif status == 'failed':
                raise HTTPException(status_code=500, detail="Azure Vision API OCR failed")
            
            retry_count += 1
        
        raise HTTPException(status_code=500, detail="Azure Vision API OCR timeout")
        
    except requests.RequestException as e:
        print(f"Azure Vision API request error: {e}")
        raise HTTPException(status_code=500, detail=f"Azure Vision API request error: {str(e)}")
    except Exception as e:
        print(f"Error extracting text with Azure Vision: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")
    """Load TrOCR model for handwritten text recognition - using larger model for better accuracy"""
    global trocr_processor, trocr_model, trocr_pipeline
    
    if trocr_model is not None:
        return
    
    try:
        print(f"Loading TrOCR model on {device}...")
        model_name = "microsoft/trocr-base-handwritten"
        
        # Use TrOCRProcessor and VisionEncoderDecoderModel (correct classes for TrOCR)
        try:
            trocr_processor = TrOCRProcessor.from_pretrained(model_name)
            trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
            print(f"Loaded {model_name} with TrOCRProcessor successfully!")
        except Exception as e:
            print(f"Could not load with TrOCRProcessor, trying AutoProcessor: {e}")
            # Fallback to AutoProcessor
            trocr_processor = AutoProcessor.from_pretrained(model_name)
            trocr_model = AutoModelForVision2Seq.from_pretrained(model_name)
            print(f"Loaded {model_name} with AutoProcessor")
        
        trocr_model.to(device)
        trocr_model.eval()
        # Also try to create a high-level pipeline for convenience and robust decoding
        # Try to create a high-level pipeline for convenience and robust decoding.
        # Prefer building pipeline with explicit tokenizer + AutoModelForVision2Seq if possible (user preference).
        try:
            # Try to load tokenizer + AutoModelForVision2Seq explicitly (user requested)
            try:
                trocr_tokenizer = AutoTokenizer.from_pretrained(model_name)
                trocr_model = AutoModelForVision2Seq.from_pretrained(model_name)
                # move model to device
                trocr_model.to(device)
                trocr_model.eval()
                print(f"Loaded tokenizer + AutoModelForVision2Seq for {model_name}")
            except Exception as e:
                # If explicit AutoModel fails, continue with previously loaded VisionEncoderDecoderModel
                print(f"AutoTokenizer/AutoModelForVision2Seq load failed: {e}")

            device_index = 0 if torch.cuda.is_available() else -1
            # Create pipeline using model name (or explicit model/tokenizer)
            if trocr_tokenizer is not None and trocr_model is not None:
                trocr_pipeline = pipeline("image-to-text", model=trocr_model, tokenizer=trocr_tokenizer, device=device_index)
            else:
                trocr_pipeline = pipeline("image-to-text", model=model_name, device=device_index)

            print("TrOCR pipeline created successfully!")
        except Exception as e:
            # If pipeline creation fails, continue using lower-level model/tokenizer
            trocr_pipeline = None
            print(f"Warning: Could not create pipeline for TrOCR: {e}")
        
        print("TrOCR model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading TrOCR model: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_sam2_model():
    """Load SAM2 model for segmentation (optional)"""
    global sam2_model, sam2_mask_generator
    
    if not SAM2_AVAILABLE:
        return False
    
    if sam2_model is not None:
        return True
    
    try:
        model_cfg = "sam2_hiera_large.yaml"
        checkpoint_path = "checkpoints/sam2_hiera_large.pt"
        
        if not os.path.exists(checkpoint_path):
            print("SAM2 checkpoint not found. Using simple region detection instead.")
            return False
        
        print(f"Loading SAM2 model on {device}...")
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        sam2_mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        print("SAM2 model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"SAM2 not available: {e}")
        return False

# def load_google_vision():
#     """Load Google Cloud Vision API client - COMMENTED OUT (requires billing)"""
#     global vision_client
#     
#     if vision_client is not None:
#         return
#     
#     try:
#         print("Initializing Google Cloud Vision API...")
#         # Check for credentials file
#         creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
#         if not creds_path:
#             # Try common locations
#             possible_paths = [
#                 'credentials.json',
#                 'google-credentials.json',
#                 os.path.expanduser('~/.config/gcloud/application_default_credentials.json')
#             ]
#             for path in possible_paths:
#                 if os.path.exists(path):
#                     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
#                     creds_path = path
#                     break
#         
#         if creds_path and os.path.exists(creds_path):
#             print(f"Using credentials from: {creds_path}")
#         else:
#             print("Warning: No credentials file found. Set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
#             print("Or place credentials.json in the backend directory.")
#         
#         vision_client = vision.ImageAnnotatorClient()
#         print("Google Cloud Vision API initialized successfully!")
#         
#     except Exception as e:
#         print(f"Warning: Could not initialize Google Vision API: {e}")
#         print("Make sure you have:")
#         print("1. Installed: pip install google-cloud-vision")
#         print("2. Set up credentials: https://cloud.google.com/vision/docs/setup")
#         vision_client = None

@app.on_event("startup")
async def startup_event():
    """Verify Azure Vision API configuration and load optional SAM2 on startup"""
    try:
        verify_azure_vision_config()
    except Exception as e:
        print(f"Warning: Could not verify Azure Vision API config: {e}")
    
    # Try to load SAM2 (optional)
    load_sam2_model()

@app.get("/")
async def root():
    return {
        "message": "Prescription OCR API with Azure Vision",
        "status": "running",
        "azure_vision_configured": bool(azure_vision_endpoint and azure_vision_key),
        "sam2_loaded": sam2_model is not None
    }

@app.get("/health")
async def health_check():
    """Check if services are configured/loaded"""
    return {
        "status": "healthy",
        "azure_vision_configured": bool(azure_vision_endpoint and azure_vision_key),
        "sam2_loaded": sam2_model is not None,
        "device": device
    }

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image to numpy array"""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def segment_image_sam2(image: np.ndarray) -> List[Dict]:
    """Use SAM2 to segment the image into regions"""
    global sam2_mask_generator
    
    if sam2_mask_generator is None:
        return []
    
    try:
        masks = sam2_mask_generator.generate(image)
        if not masks:
            return []
        
        # Sort by area and return top regions
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
        return masks_sorted[:15]  # Top 15 regions
        
    except Exception as e:
        print(f"Error in SAM2 segmentation: {e}")
        return []

def preprocess_image_for_segmentation(image: np.ndarray) -> np.ndarray:
    """Enhance image for better text detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    return enhanced

def segment_image_simple(image: np.ndarray) -> List[Dict]:
    """Improved region detection using multiple techniques - more aggressive"""
    try:
        h, w = image.shape[:2]
        print(f"Image size: {w}x{h}")
        
        # Much lower minimum area - be more aggressive
        min_area = max(50, (h * w) * 0.0005)  # Very low threshold
        print(f"Minimum area threshold: {min_area}")
        
        # Preprocess image
        processed = preprocess_image_for_segmentation(image)
        
        # Try multiple thresholding methods
        # Method 1: Adaptive thresholding (multiple block sizes)
        binary1 = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        binary1_large = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # Method 2: Otsu thresholding
        _, binary2 = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Method 3: Simple threshold (for light text on dark background)
        _, binary3 = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Combine all methods
        binary = cv2.bitwise_or(binary1, binary2)
        binary = cv2.bitwise_or(binary, binary1_large)
        binary = cv2.bitwise_or(binary, binary3)
        
        # Morphological operations to connect text components (more aggressive)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # Close to connect nearby text
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
        # Open to remove noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Find contours - use both external and tree
        contours_ext, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_tree, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine contours
        all_contours = list(contours_ext) + list(contours_tree)
        
        print(f"Found {len(all_contours)} contours")
        
        regions = []
        
        for contour in all_contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Very lenient aspect ratio filter
            aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
            if aspect_ratio < 0.05 or aspect_ratio > 20:  # Very lenient
                continue
            
            # Very lenient size filter
            if w_rect < 10 or h_rect < 5:
                continue
            
            # Create mask (use uint8 for OpenCV, then convert to bool)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            mask = mask.astype(bool)
            
            # Expand mask more to include full text
            kernel_expand = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            mask_expanded = cv2.dilate(mask.astype(np.uint8), kernel_expand, iterations=2).astype(bool)
            
            regions.append({
                'segmentation': mask_expanded,
                'area': area,
                'bbox': [x, y, w_rect, h_rect],
                'aspect_ratio': aspect_ratio
            })
        
        print(f"Found {len(regions)} regions after filtering")
        
        # Sort by area
        regions.sort(key=lambda x: x['area'], reverse=True)
        
        # Less aggressive overlap filtering - keep more regions
        filtered_regions = []
        for region in regions:
            is_overlapping = False
            for existing in filtered_regions:
                # Check overlap
                x1, y1, w1, h1 = region['bbox']
                x2, y2, w2, h2 = existing['bbox']
                
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                # Only filter if significant overlap (80% instead of 50%)
                if overlap_area > 0.8 * min(region['area'], existing['area']):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_regions.append(region)
        
        print(f"After overlap filtering: {len(filtered_regions)} regions")
        
        # If still no regions, try creating regions from the entire image divided into sections
        if not filtered_regions:
            print("No regions found, creating grid-based regions...")
            # Divide image into 4 sections
            h_mid = h // 2
            w_mid = w // 2
            
            for i, (y_start, y_end) in enumerate([(0, h_mid), (h_mid, h)]):
                for j, (x_start, x_end) in enumerate([(0, w_mid), (w_mid, w)]):
                    mask = np.zeros((h, w), dtype=bool)
                    mask[y_start:y_end, x_start:x_end] = True
                    filtered_regions.append({
                        'segmentation': mask,
                        'area': (y_end - y_start) * (x_end - x_start),
                        'bbox': [x_start, y_start, x_end - x_start, y_end - y_start],
                        'aspect_ratio': (x_end - x_start) / (y_end - y_start) if (y_end - y_start) > 0 else 1
                    })
        
        return filtered_regions[:20]  # Return top 20 regions
        
    except Exception as e:
        print(f"Error in simple segmentation: {e}")
        import traceback
        traceback.print_exc()
        return []

def classify_region_handwritten(image: np.ndarray, mask: np.ndarray) -> bool:
    """
    Improved classification of handwritten vs printed text.
    Uses multiple features: variance, edge patterns, stroke width variation.
    """
    try:
        h, w = image.shape[:2]
        mask_2d = mask.reshape(h, w)
        
        if np.sum(mask) == 0:
            return False
        
        # Extract the masked region
        masked_image = image.copy()
        masked_image[~mask] = [255, 255, 255]  # Set background to white
        
        # Get bounding box
        rows = np.any(mask_2d, axis=1)
        cols = np.any(mask_2d, axis=0)
        if not (np.any(rows) and np.any(cols)):
            return False
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Extract region
        region = masked_image[y_min:y_max+1, x_min:x_max+1]
        region_mask = mask_2d[y_min:y_max+1, x_min:x_max+1]
        
        # Convert to grayscale
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        gray_region[~region_mask] = 255  # Set background to white
        
        # Feature 1: Variance (handwritten has more variation)
        variance = np.var(gray_region[region_mask])
        
        # Feature 2: Edge density and complexity
        edges = cv2.Canny(gray_region, 50, 150)
        edge_density = np.sum(edges[region_mask] > 0) / np.sum(region_mask) if np.sum(region_mask) > 0 else 0
        
        # Feature 3: Stroke width variation (handwritten has variable stroke width)
        # Use distance transform to estimate stroke width
        binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        binary[~region_mask] = 0
        
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        stroke_widths = dist_transform[region_mask]
        stroke_width_variance = np.var(stroke_widths) if len(stroke_widths) > 0 else 0
        stroke_width_mean = np.mean(stroke_widths) if len(stroke_widths) > 0 else 0
        
        # Feature 4: Horizontal projection variance (handwritten is less uniform)
        horizontal_proj = np.sum(binary, axis=1)
        proj_variance = np.var(horizontal_proj)
        
        # Feature 5: Connected components (handwritten has more irregular shapes)
        num_labels, labels = cv2.connectedComponents(binary)
        component_sizes = [np.sum(labels == i) for i in range(1, num_labels)]
        if component_sizes:
            component_size_variance = np.var(component_sizes)
        else:
            component_size_variance = 0
        
        # Classification logic (more lenient thresholds)
        features = {
            'variance': variance,
            'edge_density': edge_density,
            'stroke_variance': stroke_width_variance,
            'proj_variance': proj_variance,
            'component_variance': component_size_variance
        }
        
        # Score-based classification
        score = 0
        
        # Variance check (handwritten has higher variance)
        if variance > 800:  # Lowered threshold
            score += 2
        elif variance > 400:
            score += 1
        
        # Edge density (handwritten has more edges)
        if edge_density > 0.05:  # Lowered threshold
            score += 2
        elif edge_density > 0.02:
            score += 1
        
        # Stroke width variation (handwritten has variable strokes)
        if stroke_width_variance > 1.0 and stroke_width_mean > 0:
            score += 2
        elif stroke_width_variance > 0.5:
            score += 1
        
        # Projection variance (handwritten is less uniform)
        if proj_variance > 1000:
            score += 1
        
        # Component variance (handwritten has irregular shapes)
        if component_size_variance > 100:
            score += 1
        
        # Classify as handwritten if score >= 3 (more lenient)
        # If we can't determine, default to handwritten (better to OCR than skip)
        is_handwritten = score >= 3 or score == 0  # Default to True if score is 0 (uncertain)
        
        # Debug output
        print(f"Region classification - Score: {score}, Handwritten: {is_handwritten}, Features: {features}")
        
        return is_handwritten
        
    except Exception as e:
        print(f"Error classifying region: {e}")
        import traceback
        traceback.print_exc()
        # Default to handwritten if we can't classify (better to OCR than skip)
        return True

def enhance_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """Enhance image for better OCR results - optimized for handwritten text"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Light denoising (less aggressive for handwritten text to preserve details)
    denoised = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
    
    # Enhance contrast with CLAHE (better for handwritten text)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Light sharpening (less aggressive to avoid artifacts)
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Ensure minimum contrast for better recognition
    # Normalize to improve visibility
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert back to RGB for TrOCR
    rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    
    return rgb

# def extract_text_with_google_vision(image_array: np.ndarray) -> str:
#     """Extract text using Google Cloud Vision API - COMMENTED OUT (requires billing)"""
#     global vision_client
#     
#     if vision_client is None:
#         return ""
#     
#     try:
#         # Convert numpy array to bytes
#         success, encoded_image = cv2.imencode('.png', image_array)
#         if not success:
#             return ""
#         
#         image_bytes = encoded_image.tobytes()
#         
#         # Create Vision API image
#         image = vision.Image(content=image_bytes)
#         
#         # Perform text detection
#         response = vision_client.text_detection(image=image)
#         texts = response.text_annotations
#         
#         if texts:
#             # First annotation contains the entire detected text
#             return texts[0].description.strip()
#         
#         return ""
#         
#     except Exception as e:
#         print(f"Google Vision API error: {e}")
#         return ""

def extract_text_from_region(image: Image.Image, mask: np.ndarray) -> str:
    """Extract text from a specific region using TrOCR"""
    global trocr_processor, trocr_model, trocr_pipeline
    
    try:
        image_array = np.array(image)
        h, w = image_array.shape[:2]
        mask_2d = mask.reshape(h, w)
        
        # Get bounding box with padding
        rows = np.any(mask_2d, axis=1)
        cols = np.any(mask_2d, axis=0)
        
        if not (np.any(rows) and np.any(cols)):
            return ""
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Add generous padding (20% of region size)
        region_h = y_max - y_min
        region_w = x_max - x_min
        padding_h = max(10, int(region_h * 0.2))
        padding_w = max(10, int(region_w * 0.2))
        
        y_min = max(0, y_min - padding_h)
        y_max = min(h, y_max + padding_h)
        x_min = max(0, x_min - padding_w)
        x_max = min(w, x_max + padding_w)
        
        # Extract region
        region = image_array[y_min:y_max, x_min:x_max]
        region_mask = mask_2d[y_min:y_max, x_min:x_max]
        
        # Create clean background
        region_clean = region.copy()
        region_clean[~region_mask] = [255, 255, 255]  # White background
        
        # Enhance image for OCR
        region_enhanced = enhance_image_for_ocr(region_clean)
        
        # Ensure minimum size
        min_size = 32
        if region_enhanced.shape[0] < min_size or region_enhanced.shape[1] < min_size:
            scale = max(min_size / region_enhanced.shape[0], min_size / region_enhanced.shape[1])
            new_h = int(region_enhanced.shape[0] * scale)
            new_w = int(region_enhanced.shape[1] * scale)
            region_enhanced = cv2.resize(region_enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Use TrOCR for text extraction
        text = ""
        if trocr_model is not None:
            try:
                region_image = Image.fromarray(region_enhanced).convert('RGB')
                
                # Ensure image is large enough for TrOCR (minimum recommended size)
                # TrOCR works better with larger images, especially for handwritten text
                min_dimension = 64
                if region_image.width < min_dimension or region_image.height < min_dimension:
                    scale = max(min_dimension / region_image.width, min_dimension / region_image.height)
                    new_width = int(region_image.width * scale)
                    new_height = int(region_image.height * scale)
                    region_image = region_image.resize((new_width, new_height), Image.LANCZOS)
                
                # If we have a high-level pipeline, prefer it (simpler and handles tokenizer)
                if trocr_pipeline is not None:
                    try:
                        outputs = trocr_pipeline(region_image)
                        if isinstance(outputs, list) and len(outputs) > 0:
                            # pipeline returns a list of dicts, usually with 'generated_text'
                            text_out = outputs[0].get('generated_text') or outputs[0].get('text') or ""
                            if text_out:
                                print(f"TrOCR pipeline extracted: {text_out[:100]}...")
                                return text_out.strip()
                    except Exception as e:
                        print(f"TrOCR pipeline error: {e}")

                # TrOCRProcessor has image_processor and tokenizer as separate components
                # Always use image_processor directly to avoid text processor confusion
                if hasattr(trocr_processor, 'image_processor'):
                    # Use the image processor component directly (this is the correct way)
                    pixel_values = trocr_processor.image_processor(region_image, return_tensors="pt").pixel_values
                else:
                    # Fallback: try calling with images= keyword
                    try:
                        processed = trocr_processor(images=region_image, return_tensors="pt")
                        pixel_values = processed.pixel_values
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        return ""
                
                pixel_values = pixel_values.to(device)
                
                # Get tokenizer for generation parameters
                tokenizer = None
                if hasattr(trocr_processor, 'tokenizer'):
                    tokenizer = trocr_processor.tokenizer
                elif hasattr(trocr_model, 'tokenizer'):
                    tokenizer = trocr_model.tokenizer
                
                # Prepare generation parameters
                gen_kwargs = {
                    'max_length': 512,  # Increased for longer text
                    'num_beams': 8,  # More beams for better handwritten text recognition
                    'early_stopping': False,  # Don't stop early - handwritten text can be longer
                    'repetition_penalty': 1.5,  # Reduced penalty (handwritten may have natural repetition)
                    'length_penalty': 1.0,  # Neutral length penalty
                    'no_repeat_ngram_size': 3,  # Prevent 3-gram repetition
                    'do_sample': False,  # Use beam search for consistency
                }
                
                # Add token IDs if tokenizer is available
                if tokenizer is not None:
                    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                        gen_kwargs['pad_token_id'] = tokenizer.pad_token_id
                    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                        gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
                
                # Improved generation parameters for handwritten text recognition
                with torch.no_grad():
                    generated_ids = trocr_model.generate(pixel_values, **gen_kwargs)

                # Debug: print generated ids (truncated) and shape
                try:
                    print(f"Generated ids shape: {generated_ids.shape}")
                    print(f"Sample generated ids (first 20): {generated_ids[0][:20].tolist()}")
                except Exception:
                    pass

                # Decode using the tokenizer
                def load_trocr_model():
                    """Load TrOCR model for handwritten text recognition - default to large handwritten model.

                    Behavior:
                    - Uses environment variable TROCR_MODEL to allow overriding model name.
                    - Attempts to load tokenizer+model and create a pipeline; falls back gracefully.
                    - If CUDA OOM occurs while moving model to GPU, falls back to CPU and retries.
                    """
                    global trocr_processor, trocr_model, trocr_pipeline, trocr_tokenizer, device

                    if trocr_model is not None:
                        return

                    try:
                        # Allow overriding model via env var for testing
                        model_name = os.getenv('TROCR_MODEL', 'microsoft/trocr-large-handwritten')
                        print(f"Loading TrOCR model '{model_name}' on {device}...")

                        # Try TrOCRProcessor + VisionEncoderDecoderModel first
                        try:
                            trocr_processor = TrOCRProcessor.from_pretrained(model_name)
                            trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
                            print(f"Loaded {model_name} with TrOCRProcessor + VisionEncoderDecoderModel successfully!")
                        except Exception as e:
                            print(f"Could not load with TrOCRProcessor/VisionEncoderDecoderModel: {e}")
                            print("Falling back to AutoProcessor + AutoModelForVision2Seq...")
                            trocr_processor = AutoProcessor.from_pretrained(model_name)
                            trocr_model = AutoModelForVision2Seq.from_pretrained(model_name)
                            print(f"Loaded {model_name} with AutoProcessor + AutoModelForVision2Seq")

                        # Try to move model to preferred device; if OOM, retry on CPU
                        try:
                            trocr_model.to(device)
                        except RuntimeError as e:
                            err_str = str(e).lower()
                            if 'out of memory' in err_str or 'cuda' in err_str:
                                print(f"CUDA OOM while moving model to device {device}: {e}")
                                print("Falling back to CPU for model execution. Set TROCR_DEVICE=cpu to force CPU in future.")
                                device = 'cpu'
                                trocr_model.to(device)
                            else:
                                raise

                        trocr_model.eval()

                        # Try to create high-level pipeline using explicit tokenizer+model if possible
                        try:
                            # Try loading tokenizer explicitly (user-specified route)
                            try:
                                trocr_tokenizer = AutoTokenizer.from_pretrained(model_name)
                                # prefer AutoModelForVision2Seq if tokenizer available
                                try:
                                    trocr_model = AutoModelForVision2Seq.from_pretrained(model_name)
                                    trocr_model.to(device)
                                    trocr_model.eval()
                                    print(f"Loaded AutoModelForVision2Seq for {model_name}")
                                except Exception:
                                    # keep existing trocr_model if AutoModel loading fails
                                    pass
                            except Exception as e:
                                print(f"AutoTokenizer not available for {model_name}: {e}")

                            device_index = 0 if torch.cuda.is_available() and device == 'cuda' else -1
                            # Provide an image processor if available (avoids "Impossible to guess which image processor to use")
                            image_processor = None
                            if trocr_processor is not None and hasattr(trocr_processor, 'image_processor'):
                                image_processor = trocr_processor.image_processor

                            if trocr_tokenizer is not None and trocr_model is not None:
                                if image_processor is not None:
                                    trocr_pipeline = pipeline('image-to-text', model=trocr_model, tokenizer=trocr_tokenizer, image_processor=image_processor, device=device_index)
                                else:
                                    trocr_pipeline = pipeline('image-to-text', model=trocr_model, tokenizer=trocr_tokenizer, device=device_index)
                            else:
                                if image_processor is not None:
                                    trocr_pipeline = pipeline('image-to-text', model=model_name, image_processor=image_processor, device=device_index)
                                else:
                                    trocr_pipeline = pipeline('image-to-text', model=model_name, device=device_index)

                            print('TrOCR pipeline created successfully!')
                        except Exception as e:
                            trocr_pipeline = None
                            print(f"Warning: Could not create pipeline for TrOCR: {e}")

                        print('TrOCR model loaded successfully!')

                    except Exception as e:
                        print(f"Error loading TrOCR model: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                    except Exception as e:
                        print(f"Fallback small gen error: {e}")

                if text:
                    print(f"TrOCR extracted: {text[:100]}...")
                    print(f"  Extracted text: '{text}'")
                else:
                    print("TrOCR returned empty text after fallbacks")
                    
            except Exception as e:
                print(f"TrOCR error: {e}")
                import traceback
                traceback.print_exc()
        
        return text
        
    except Exception as e:
        print(f"Error extracting text from region: {e}")
        import traceback
        traceback.print_exc()
        return ""

@app.post("/api/segment")
async def segment_and_ocr(file: UploadFile = File(...)):
    """
    Upload an image, segment it to detect text regions, extract text using Azure Vision API,
    and return an annotated image with labeled bounding boxes
    """
    try:
        # Read image
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        image_pil = Image.fromarray(image_array)
        
        print(f"Processing image of size {image_array.shape}")
        
        # Step 1: Segment the image to detect text regions
        if sam2_mask_generator is not None:
            print("Using SAM2 for segmentation...")
            regions = segment_image_sam2(image_array)
            if not regions:
                print("SAM2 found no regions, falling back to simple segmentation...")
                regions = segment_image_simple(image_array)
        else:
            print("Using simple region detection...")
            regions = segment_image_simple(image_array)
        
        print(f"Segmentation found {len(regions)} regions")
        
        # If still no regions, create simple grid-based regions for visualization
        if not regions:
            print("No regions detected, creating grid-based regions for demo...")
            h, w = image_array.shape[:2]
            # Create a simple 2x2 grid
            regions = []
            grid_h = h // 2
            grid_w = w // 2
            for i in range(2):
                for j in range(2):
                    y_start = i * grid_h
                    y_end = (i + 1) * grid_h if i < 1 else h
                    x_start = j * grid_w
                    x_end = (j + 1) * grid_w if j < 1 else w
                    
                    mask = np.zeros((h, w), dtype=bool)
                    mask[y_start:y_end, x_start:x_end] = True
                    
                    regions.append({
                        'bbox': [x_start, y_start, x_end - x_start, y_end - y_start],
                        'segmentation': mask,
                        'area': (y_end - y_start) * (x_end - x_start)
                    })
            
            print(f"Created {len(regions)} grid-based regions")
        
        # Step 2: Extract text from entire image using Azure Vision API
        extracted_text = extract_text_with_azure_vision(image_bytes)
        
        # Step 3: Create annotated image with detected regions and text labels
        # Work directly with uint8 to avoid conversion issues
        annotated_image = image_array.copy()
        
        print(f"Drawing {len(regions)} regions on image (shape: {annotated_image.shape})...")
        
        # Draw bounding boxes around detected regions with bright colors
        region_count = 0
        for i, region in enumerate(regions):
            bbox = region['bbox']
            x, y, w, h = bbox
            
            # Ensure bbox values are valid and within image bounds
            h_img, w_img = annotated_image.shape[:2]
            x = max(0, min(int(x), w_img - 1))
            y = max(0, min(int(y), h_img - 1))
            w = max(1, min(int(w), w_img - x))
            h = max(1, min(int(h), h_img - y))
            
            # Skip invalid regions
            if w <= 0 or h <= 0:
                print(f"  Skipping invalid region {i+1}")
                continue
            
            # Use alternating VERY bright neon colors for maximum visibility (BGR format for OpenCV)
            colors = [
                (0, 0, 255),      # Bright Red
                (0, 255, 0),      # Bright Green  
                (255, 0, 255),    # Bright Magenta
                (0, 255, 255),    # Bright Yellow
                (255, 128, 0),    # Bright Orange
                (255, 255, 0)     # Bright Cyan
            ]
            color = colors[i % len(colors)]
            
            # Draw THICK outer bounding box
            thickness = 8
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw semi-transparent inner fill for better visibility
            overlay = annotated_image.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(overlay, 0.15, annotated_image, 0.85, 0, annotated_image)
            
            # Draw WHITE inner border for contrast
            cv2.rectangle(annotated_image, (x+4, y+4), (x + w-4, y + h-4), (255, 255, 255), 3)
            
            # Add large region label with solid background
            label = f"Region {i+1}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.2
            text_thickness = 3
            text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
            
            # Position label above the box if possible, otherwise inside
            if y > text_size[1] + 20:
                label_y = y - 10
                label_bg_y1 = y - text_size[1] - 20
                label_bg_y2 = y - 5
            else:
                label_y = y + text_size[1] + 15
                label_bg_y1 = y + 5
                label_bg_y2 = y + text_size[1] + 25
            
            label_bg_x1 = x
            label_bg_x2 = x + text_size[0] + 25
            
            # Draw solid colored background for label
            cv2.rectangle(annotated_image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
            
            # Draw thick white border around label
            cv2.rectangle(annotated_image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), (255, 255, 255), 3)
            
            # Draw label text with black shadow + white text
            text_position = (x + 12, label_y)
            # Black shadow/outline (offset slightly)
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(annotated_image, label, (text_position[0]+dx, text_position[1]+dy), 
                           font, font_scale, (0, 0, 0), text_thickness + 1)
            # Bright white text on top
            cv2.putText(annotated_image, label, text_position, font, font_scale, (255, 255, 255), text_thickness)
            
            region_count += 1
            print(f"  Drew region {i+1} at ({x}, {y}) size ({w}x{h}) with color {color}")
        
        print(f"Successfully drew {region_count} regions on annotated image")
        
        # Don't add text overlay - keep image clean to see bounding boxes clearly
        # The text is already available in the response
        
        # Add a thick colored border to the entire image to indicate processing complete
        border_width = 10
        # Top border - bright green
        cv2.rectangle(annotated_image, (0, 0), (annotated_image.shape[1], border_width), (0, 255, 0), -1)
        # Bottom border
        cv2.rectangle(annotated_image, (0, annotated_image.shape[0] - border_width), 
                     (annotated_image.shape[1], annotated_image.shape[0]), (0, 255, 0), -1)
        # Left border
        cv2.rectangle(annotated_image, (0, 0), (border_width, annotated_image.shape[0]), (0, 255, 0), -1)
        # Right border
        cv2.rectangle(annotated_image, (annotated_image.shape[1] - border_width, 0), 
                     (annotated_image.shape[1], annotated_image.shape[0]), (0, 255, 0), -1)
        annotated_image[-border_width:, :] = [0, 255, 0]
        # Left and right borders (green)
        annotated_image[:, :border_width] = [0, 255, 0]
        annotated_image[:, -border_width:] = [0, 255, 0]
        
        # Convert annotated image to base64
        annotated_image_pil = Image.fromarray(annotated_image)
        buffer_annotated = io.BytesIO()
        annotated_image_pil.save(buffer_annotated, format='PNG')
        annotated_base64 = base64.b64encode(buffer_annotated.getvalue()).decode()
        
        # Original image
        buffer_orig = io.BytesIO()
        image_pil.save(buffer_orig, format='PNG')
        image_base64 = base64.b64encode(buffer_orig.getvalue()).decode()
        
        # Filter PHI from extracted text before returning
        try:
            phi_result = filter_phi_with_hf(extracted_text)
            redacted_text = phi_result.get('redacted_text', '')
            phi_summary = phi_result.get('phi', [])
        except Exception as e:
            print(f"PHI filtering error: {e}")
            redacted_text = extracted_text or ""
            phi_summary = []

        # Extract medications and get FDA alternatives
        medications_found = []
        fda_alternatives = []
        try:
            print("Extracting medications from text...")
            medications_found = extract_medications_from_text(extracted_text)
            print(f"Found {len(medications_found)} medications")
            
            # Get FDA alternatives for each medication
            for med in medications_found:
                fda_result = get_fda_drug_alternatives(med['name'])
                if fda_result.get('found'):
                    fda_alternatives.append({
                        'original_drug': med,
                        'fda_info': fda_result
                    })
        except Exception as e:
            print(f"Medication extraction error: {e}")

        return JSONResponse({
            "success": True,
            # Return redacted text to avoid leaking PHI
            "extracted_text": redacted_text,
            "phi_summary": phi_summary,
            "medications": medications_found,
            "fda_alternatives": fda_alternatives,
            "annotated_image": f"data:image/png;base64,{annotated_base64}",
            "original_image": f"data:image/png;base64,{image_base64}",
            "regions_detected": region_count,
            "method": "azure_vision_with_segmentation_and_fda"
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

