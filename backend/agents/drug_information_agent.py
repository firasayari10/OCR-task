"""
Drug Information Agent

This agent extracts medications from text and queries drug databases (RxNorm, FDA, LLaMA)
for alternatives and detailed information.
"""

import re
import requests
import logging
from typing import Dict, List, Any
from agents.base_agent import BaseAgent, AgentResponse, Tool
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DrugInformationAgent(BaseAgent):
    """Agent specialized in extracting medications and finding alternatives."""
    
    def __init__(self):
        super().__init__(
            name="DrugInformationAgent",
            description="Extracts medications from prescription text and finds drug alternatives using FDA and RxNorm APIs",
            system_prompt="""You are a specialized agent that:
1. Extracts medication names and dosages from prescription text
2. Queries drug databases (RxNorm, FDA openFDA) for drug information
3. Provides alternative medications and detailed drug information
4. Uses LLaMA AI as fallback when databases have no information
5. Checks local vector database of essential medications first

You work with PHI-filtered text to protect patient privacy."""
        )
        
        # Initialize vector database
        self.vector_db = None
        try:
            from medication_vector_db import MedicationVectorDB
            self.vector_db = MedicationVectorDB()
            logger.info(f"Vector DB loaded with {len(self.vector_db.metadata)} medications")
        except Exception as e:
            logger.warning(f"Could not load vector database: {e}")
        
        self._register_tools()
    
    def _register_tools(self):
        """Register medication extraction and drug query tools."""
        
        # Tool 1: Extract medications
        extract_tool = Tool(
            name="extract_medications",
            description="Extract medication names and dosages from prescription text",
            function=self._extract_medications_impl,
            parameters={
                "text": {"type": "str", "description": "Prescription text to analyze"}
            }
        )
        self.register_tool(extract_tool)
        
        # Tool 2: Query drug databases
        drug_info_tool = Tool(
            name="query_drug_info",
            description="Query RxNorm, FDA, and LLaMA APIs for drug information and alternatives",
            function=self._query_drug_info_impl,
            parameters={
                "drug_name": {"type": "str", "description": "Medication name to query"}
            }
        )
        self.register_tool(drug_info_tool)
    
    async def process(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """Process medication extraction and drug information query tasks."""
        try:
            text = context.get('text', '')
            if not text:
                return AgentResponse(
                    success=False,
                    error="No text provided for medication extraction",
                    agent_name=self.name
                )
            
            logger.info(f"Processing medication extraction from {len(text)} characters of text")
            
            # Step 1: Extract medications
            medications = await self.use_tool("extract_medications", text=text)
            
            if not medications:
                return AgentResponse(
                    success=True,
                    data={
                        "medications": [],
                        "drug_alternatives": [],
                        "message": "No medications detected in text"
                    },
                    agent_name=self.name,
                    tools_used=["extract_medications"]
                )
            
            logger.info(f"Found {len(medications)} medications")
            
            # Step 2: Query drug information for each medication
            drug_alternatives = []
            tools_used = ["extract_medications", "query_drug_info"]
            
            for med in medications:
                drug_name = med['name']
                logger.info(f"Querying drug information for: {drug_name}")
                
                try:
                    drug_info = await self.use_tool("query_drug_info", drug_name=drug_name)
                    
                    if drug_info.get('found') or drug_info.get('text_from_llm'):
                        drug_alternatives.append({
                            "original_drug": med,
                            "drug_info": drug_info
                        })
                except Exception as e:
                    logger.error(f"Failed to query drug info for {drug_name}: {e}")
            
            return AgentResponse(
                success=True,
                data={
                    "medications": medications,
                    "drug_alternatives": drug_alternatives,
                    "total_medications": len(medications),
                    "medications_with_info": len(drug_alternatives)
                },
                metadata={
                    "medications_found": len(medications),
                    "alternatives_found": len(drug_alternatives)
                },
                agent_name=self.name,
                tools_used=tools_used
            )
            
        except Exception as e:
            logger.error(f"DrugInformationAgent error: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name
            )
    
    def _extract_medications_impl(self, text: str, **kwargs) -> List[Dict]:
        """Extract medication names and dosages from prescription text."""
        medications = []
        
        # Pattern 1: Tab/Cap/Inj/Syr followed by drug name and dosage
        pattern1 = r'(?:Tab\.?|Cap\.?|Inj\.?|Syr\.?|Tablet|Capsule)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|units?|IU))'
        
        # Pattern 2: Drug name with dosage (suffix-based)
        pattern2 = r'\b([A-Z][a-z]+(?:cillin|mycin|pril|olol|ine|azole|ide|tax|done|pine|lone|sartan|statin|flam|idol|tin|zol))\s+(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|units?|IU))\b'
        
        # Pattern 3: Common medication names with dosages
        common_drugs = r'\b(augmentin|amoxicillin|enzoflam|diclofenac|ibuprofen|paracetamol|acetaminophen|aspirin|metformin|lisinopril|atorvastatin|omeprazole|pantoprazole|rabeprazole|amlodipine|losartan|telmisartan|azithromycin|ciprofloxacin|cetirizine|loratadine|montelukast|salbutamol|prednisone|metronidazole|fluconazole|warfarin|clopidogrel|insulin|gabapentin|pregabalin|tramadol|alprazolam|diazepam|sertraline|escitalopram|fluoxetine|quetiapine|ranitidine|esomeprazole|domperidone|bisoprolol|atenolol|furosemide|spironolactone|enalapril|valsartan|tamsulosin|sildenafil|levothyroxine|vitamin|calcium|iron)\b'
        
        # Extract with pattern 1
        for m in re.finditer(pattern1, text, flags=re.IGNORECASE):
            medications.append({
                'name': m.group(1).strip().lower(),
                'dosage': m.group(2).strip(),
                'original_text': m.group(0),
                'start': m.start(),
                'end': m.end()
            })
        
        # Extract with pattern 2
        for m in re.finditer(pattern2, text, flags=re.IGNORECASE):
            if not any(med['start'] == m.start() for med in medications):
                medications.append({
                    'name': m.group(1).strip().lower(),
                    'dosage': m.group(2).strip(),
                    'original_text': m.group(0),
                    'start': m.start(),
                    'end': m.end()
                })
        
        # Extract common drugs
        for m in re.finditer(common_drugs, text, flags=re.IGNORECASE):
            # Look for nearby dosage
            context = text[m.end():min(len(text), m.end() + 30)]
            dosage_match = re.search(r'(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|units?|IU))', context, flags=re.IGNORECASE)
            dosage = dosage_match.group(1) if dosage_match else None
            
            # Check if this medication is not a duplicate
            def get_start_pos(med_dict: Dict) -> int:
                start_val = med_dict.get('start', -1000)
                return int(start_val) if isinstance(start_val, int) else -1000
            
            if not any(abs(get_start_pos(existing_med) - m.start()) < 10 for existing_med in medications):
                medications.append({
                    'name': m.group(1).strip().lower(),
                    'dosage': dosage,
                    'original_text': m.group(0),
                    'start': m.start(),
                    'end': m.end()
                })
        
        # Deduplicate
        seen = set()
        unique_meds = []
        for med in sorted(medications, key=lambda x: get_start_pos(x) if x else 0):
            # Safely handle start position
            start_val = med.get('start', 0)
            start_pos: int = int(start_val) if isinstance(start_val, int) else 0
            key = (med['name'], start_pos // 10)
            if key not in seen:
                seen.add(key)
                unique_meds.append(med)
        
        logger.info(f"Extracted {len(unique_meds)} unique medications")
        return unique_meds
    
    def _query_drug_info_impl(self, drug_name: str, **kwargs) -> Dict:
        """Query drug databases for information and alternatives."""
        try:
            all_sources = []
            vector_db_results = None
            
            # Step 1: Always query vector database if available (essential medications)
            if self.vector_db and len(self.vector_db.metadata) > 0:
                logger.info(f"Querying vector database for: {drug_name}")
                try:
                    vector_results = self.vector_db.search_by_name(drug_name, top_k=5)
                    
                    if vector_results and vector_results[0]['similarity_score'] > 0.6:
                        # Found in vector database
                        logger.info(f"Found in vector DB: {vector_results[0]['name']} (score: {vector_results[0]['similarity_score']:.2f})")
                        vector_db_results = self._format_vector_db_result(drug_name, vector_results)
                        all_sources.append({
                            'source': 'Essential Medications DB',
                            'data': vector_db_results,
                            'priority': 1
                        })
                        
                        # If high confidence match, return immediately
                        if vector_results[0]['similarity_score'] > 0.85:
                            logger.info(f"High confidence match in vector DB, returning early")
                            return vector_db_results
                except Exception as e:
                    logger.error(f"Vector DB query failed: {e}")
            
            # Step 2: Query RxNorm API (always try, even if found in vector DB)
            logger.info(f"Querying RxNorm API for: {drug_name}")
            try:
                rxnorm_result = self._query_rxnorm(drug_name)
                if rxnorm_result['found']:
                    logger.info(f"Found in RxNorm API")
                    all_sources.append({
                        'source': 'RxNorm (NIH)',
                        'data': rxnorm_result,
                        'priority': 2
                    })
            except Exception as e:
                logger.error(f"RxNorm query failed: {e}")
            
            # Step 3: Query FDA openFDA API
            logger.info(f"Querying FDA API for: {drug_name}")
            try:
                fda_result = self._query_fda(drug_name)
                if fda_result['found']:
                    logger.info(f"Found in FDA API")
                    all_sources.append({
                        'source': 'FDA openFDA',
                        'data': fda_result,
                        'priority': 3
                    })
            except Exception as e:
                logger.error(f"FDA query failed: {e}")
            
            # Step 4: If nothing found, try LLaMA AI
            if not all_sources:
                logger.info(f"No results from databases, trying LLaMA AI")
                try:
                    llama_result = self._query_llama(drug_name)
                    if llama_result.get('found') or llama_result.get('text_from_llm'):
                        all_sources.append({
                            'source': 'LLaMA AI',
                            'data': llama_result,
                            'priority': 4
                        })
                except Exception as e:
                    logger.error(f"LLaMA query failed: {e}")
            
            # Combine results from all sources
            if all_sources:
                return self._combine_multi_source_results(drug_name, all_sources)
            
            # Nothing found anywhere
            return {
                'drug_name': drug_name,
                'found': False,
                'message': 'No information found in any database',
                'sources_checked': ['Vector DB', 'RxNorm', 'FDA', 'LLaMA']
            }
            
        except Exception as e:
            logger.error(f"Error querying drug APIs: {e}")
            return {
                'drug_name': drug_name,
                'found': False,
                'error': str(e)
            }
    
    def _combine_multi_source_results(self, drug_name: str, sources: List[Dict]) -> Dict:
        """Combine results from multiple sources into a unified response."""
        # Sort by priority
        sources.sort(key=lambda x: x['priority'])
        
        # Primary result (highest priority source)
        primary = sources[0]['data']
        
        # Collect all alternatives from all sources
        all_alternatives = []
        seen_names = set()
        
        for src in sources:
            data = src['data']
            alts = data.get('alternatives', [])
            
            for alt in alts:
                name = alt.get('generic_name', '').lower()
                if name and name not in seen_names:
                    alt['source'] = src['source']
                    all_alternatives.append(alt)
                    seen_names.add(name)
        
        # Build combined result
        result = {
            'drug_name': drug_name,
            'found': True,
            'primary_source': sources[0]['source'],
            'sources_found': [s['source'] for s in sources],
            'alternatives': all_alternatives[:15],  # Limit to top 15
            'total_sources_checked': len(sources)
        }
        
        # Include primary source specific data
        if 'match_confidence' in primary:
            result['match_confidence'] = primary['match_confidence']
            result['matched_name'] = primary.get('matched_name')
        
        if 'category' in primary:
            result['category'] = primary['category']
        
        if 'usage_type' in primary:
            result['usage_type'] = primary['usage_type']
        
        if 'text_from_llm' in primary:
            result['text_from_llm'] = primary['text_from_llm']
        
        return result
    
    def _format_vector_db_result(self, query_name: str, results: List[Dict]) -> Dict:
        """Format vector database results."""
        main_result = results[0]
        
        alternatives = []
        for i, result in enumerate(results[:5], 1):
            alternatives.append({
                'generic_name': result['name'],
                'brand_names': [],
                'manufacturer': 'Various',
                'indication': f"{result.get('category', 'Unknown category')} - {result.get('usage', 'Unknown usage')} medication",
                'dosage': result.get('dosage', 'N/A'),
                'forme': result.get('forme', 'N/A'),
                'usage_type': result.get('usage', 'UNKNOWN'),
                'similarity': result.get('similarity_score', 0),
                'rank': i
            })
        
        return {
            'drug_name': query_name,
            'found': True,
            'alternatives': alternatives,
            'source': 'Essential Medications Database (Local)',
            'match_confidence': main_result['similarity_score'],
            'matched_name': main_result['name'],
            'category': main_result.get('category', 'Unknown'),
            'usage_type': main_result.get('usage', 'UNKNOWN')
        }
    
    def _query_rxnorm(self, drug_name: str) -> Dict:
        """Query RxNorm API for drug information."""
        try:
            url = "https://rxnav.nlm.nih.gov/REST/drugs.json"
            response = requests.get(url, params={'name': drug_name}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                concepts = []
                
                for group in data.get('drugGroup', {}).get('conceptGroup', []):
                    for concept in group.get('conceptProperties', []):
                        rxcui = concept.get('rxcui', '')
                        details = self._get_rxnorm_details(rxcui) if rxcui else {}
                        
                        concepts.append({
                            'generic_name': concept.get('name', ''),
                            'brand_names': details.get('brand_names', []),
                            'manufacturer': 'Various',
                            'indication': 'See prescribing information',
                            'rxcui': rxcui
                        })
                
                if concepts:
                    return {
                        'drug_name': drug_name,
                        'found': True,
                        'alternatives': concepts[:10],
                        'source': 'RxNorm (NIH)'
                    }
            
            return {'drug_name': drug_name, 'found': False}
        except Exception as e:
            logger.error(f"RxNorm query failed: {e}")
            return {'drug_name': drug_name, 'found': False}
    
    def _get_rxnorm_details(self, rxcui: str) -> Dict:
        """Get detailed drug information from RxNorm."""
        try:
            url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json"
            response = requests.get(url, params={'tty': 'BN'}, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                brand_names = []
                
                for group in data.get('relatedGroup', {}).get('conceptGroup', []):
                    for prop in group.get('conceptProperties', []):
                        brand_names.append(prop.get('name', ''))
                
                return {'brand_names': brand_names[:5]}
        except:
            pass
        
        return {'brand_names': []}
    
    def _query_fda(self, drug_name: str) -> Dict:
        """Query FDA openFDA API."""
        try:
            url = "https://api.fda.gov/drug/label.json"
            params = {'search': f'openfda.generic_name:"{drug_name}"', 'limit': 5}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                results = response.json().get('results', [])
                
                if not results:
                    # Try brand name
                    params['search'] = f'openfda.brand_name:"{drug_name}"'
                    response = requests.get(url, params=params, timeout=10)
                    results = response.json().get('results', []) if response.status_code == 200 else []
                
                if results:
                    alternatives = []
                    for result in results:
                        openfda = result.get('openfda', {})
                        alternatives.append({
                            'generic_name': openfda.get('generic_name', ['Unknown'])[0],
                            'brand_names': openfda.get('brand_name', [])[:3],
                            'manufacturer': openfda.get('manufacturer_name', ['Unknown'])[0],
                            'indication': result.get('indications_and_usage', ['Not available'])[0][:200]
                        })
                    
                    return {
                        'drug_name': drug_name,
                        'found': True,
                        'alternatives': alternatives[:10],
                        'source': 'FDA openFDA'
                    }
            
            return {'drug_name': drug_name, 'found': False}
        except Exception as e:
            logger.error(f"FDA query failed: {e}")
            return {'drug_name': drug_name, 'found': False}
    
    def _query_llama(self, drug_name: str) -> Dict:
        """Query LLaMA API as fallback."""
        try:
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                return {
                    'drug_name': drug_name,
                    'found': False,
                    'text_from_llm': 'No information available'
                }
            
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=hf_token)
            
            prompt = f"""Provide brief medical information about the medication "{drug_name}":
1. What is it used for?
2. Common dosages
3. 2-3 alternative medications with similar effects

Keep response under 150 words. Format as plain text."""
            
            # Use chat completion format instead of text_generation
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = client.chat_completion(
                messages=messages,
                model="meta-llama/Llama-3.1-70B-Instruct",
                max_tokens=250,
                temperature=0.3
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content if response.choices else "No information available"
            
            return {
                'drug_name': drug_name,
                'found': True,
                'text_from_llm': response_text,
                'source': 'LLaMA 3.1-70B-Instruct',
                'message': 'AI-generated information (verify with medical professional)'
            }
        except Exception as e:
            logger.error(f"LLaMA query failed: {e}")
            return {
                'drug_name': drug_name,
                'found': False,
                'text_from_llm': f'Drug information not available in databases. {drug_name} may be a brand-specific or regional medication.'
            }
