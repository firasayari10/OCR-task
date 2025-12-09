# PHI Filtering and Redaction

## Overview
The OCR backend now includes automatic PHI (Protected Health Information) detection and redaction using:
1. **ClinicalBERT embeddings** (optional, via Hugging Face Inference API)
2. **Named Entity Recognition (NER)** using BERT-based models (optional, via Hugging Face API)
3. **Regex-based pattern matching** for common PHI patterns (always runs)

## What Gets Redacted
The following PHI types are automatically detected and redacted:
- **Names** (after "Name:", "Patient Name:" labels)
- **Addresses** (after "Address:", "Street:", "Location:" labels)
- **Ages** (after "Age:" label)
- **Dates** (various formats: MM-DD-YY, DD/MM/YYYY, Month DD YYYY)
- **License/Registration numbers** (Lic. No., PTR No., S2 No., etc.)
- **Medical Record Numbers / IDs** (6+ digit numbers near ID/MRN/Account keywords)
- **SSNs** (format: XXX-XX-XXXX)
- **Emails**
- **Phone numbers** (various formats)
- **NER-detected entities** (PERSON, ORG, LOC) if HF API is available

## API Response
The `/api/segment` endpoint returns:
```json
{
  "success": true,
  "extracted_text": "[REDACTED_NAME] Address: [REDACTED_ADDRESS] Age: [REDACTED_AGE]...",
  "phi_summary": [
    {"label": "NAME", "sample": "John Doe", "start": 6, "end": 14},
    {"label": "ADDRESS", "sample": "123 Main St", "start": 24, "end": 35}
  ],
  "annotated_image": "data:image/png;base64,...",
  "original_image": "data:image/png;base64,...",
  "regions_detected": 4,
  "method": "azure_vision_with_segmentation"
}
```

## Configuration
Set these in your `.env` file:
```bash
HF_TOKEN=hf_your_token_here          # Required for HF Inference API (optional feature)
HF_NER_MODEL=dslim/bert-base-NER     # Optional: override default NER model
```

## Testing
Run the test script to see PHI detection in action:
```bash
cd backend
python test_phi.py
```

## Privacy Notes
- **Redacted text** is returned in the API response (PHI removed)
- **PHI summary** includes small samples of detected PHI for debugging (remove samples in production if needed)
- If HF API is unavailable, regex-based detection still works
- All PHI detection runs server-side before returning results to the frontend

## Limitations
- Regex patterns may not catch all PHI variations
- NER models may miss domain-specific medical entities
- For maximum PHI protection, review and test with your specific data formats
