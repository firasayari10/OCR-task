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

You work with PHI-filtered text to protect patient privacy."""
        )
        
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
            
            if not any(abs(med.get('start', -1000) - m.start()) < 10 for med in medications):
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
        for med in sorted(medications, key=lambda x: x['start']):
            key = (med['name'], med['start'] // 10)
            if key not in seen:
                seen.add(key)
                unique_meds.append(med)
        
        logger.info(f"Extracted {len(unique_meds)} unique medications")
        return unique_meds
    
    def _query_drug_info_impl(self, drug_name: str, **kwargs) -> Dict:
        """Query drug databases for information and alternatives."""
        try:
            # Try RxNorm API first (NIH, free, no key needed)
            rxnorm_result = self._query_rxnorm(drug_name)
            if rxnorm_result['found']:
                return rxnorm_result
            
            # Try FDA openFDA API
            fda_result = self._query_fda(drug_name)
            if fda_result['found']:
                return fda_result
            
            # Fallback to LLaMA
            llama_result = self._query_llama(drug_name)
            return llama_result
            
        except Exception as e:
            logger.error(f"Error querying drug APIs: {e}")
            return {
                'drug_name': drug_name,
                'found': False,
                'error': str(e)
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
