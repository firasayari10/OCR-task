# 🏥 Medical Prescription OCR System with Multi-Agent Architecture

An advanced OCR system for medical prescriptions using a multi-agent architecture with privacy-first PHI filtering and automated drug information extraction.

---

## 🎯 System Overview

This system processes medical prescription images through a sophisticated multi-agent pipeline that:
1. **Extracts text** from prescription images using Azure Vision OCR
2. **Detects and redacts PHI** (Protected Health Information) automatically
3. **Identifies medications** and queries multiple drug databases for alternatives
4. **Provides drug information** from FDA, RxNorm, and AI-powered sources

### Key Features

- 🤖 **6-Agent Architecture** - Orchestrated multi-agent system for specialized tasks
- 🔒 **HIPAA-Compliant PHI Filtering** - Automatic detection and redaction of sensitive information
- 💊 **Drug Information Extraction** - Automated medication detection with database queries
- 🔄 **Multi-Database Drug Lookup** - RxNorm (NIH), FDA openFDA, and LLaMA AI fallback
- 🖼️ **Image Segmentation** - SAM2-powered region detection (with fallback)
- 📱 **Modern React UI** - Beautiful, responsive interface for image upload and results display

---

## 🏗️ Architecture

### Multi-Agent System

The system uses 6 specialized agents coordinated by an orchestrator:

```
┌─────────────────────────────────────────────────────────────┐
│                    OrchestratorAgent                         │
│              (Routes tasks to specialized agents)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        OCRAgent                              │
│           (Coordinates full pipeline workflow)               │
└─────────────────────────────────────────────────────────────┘
           ↓              ↓              ↓              ↓
  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
  │Segmentation│  │    Text    │  │    PHI     │  │    Drug    │
  │   Agent    │  │Recognition │  │   Filter   │  │Information │
  │            │  │   Agent    │  │   Agent    │  │   Agent    │
  └────────────┘  └────────────┘  └────────────┘  └────────────┘
```

### Processing Pipeline

```
Image Upload
    ↓
┌─────────────────────────────────────┐
│  STEP 1: Image Segmentation         │
│  Agent: SegmentationAgent           │
│  Tools: SAM2 / Fallback Contours    │
│  Output: Detected regions           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  STEP 2: Text Recognition           │
│  Agent: TextRecognitionAgent        │
│  Tools: Azure Vision OCR / TrOCR    │
│  Output: Extracted text             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  STEP 3: PHI Filtering              │
│  Agent: PHIFilterAgent              │
│  Tools: HuggingFace NER + Regex     │
│  Output: Redacted text, PHI list    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  STEP 4: Drug Information           │
│  Agent: DrugInformationAgent        │
│  Tools: RxNorm, FDA, LLaMA AI       │
│  Output: Medications + Alternatives │
└─────────────────────────────────────┘
    ↓
Final Response to UI
```

---

## 🤖 Agent Details

### 1. **OrchestratorAgent**
- **Purpose**: Routes incoming tasks to the appropriate specialized agent
- **Routing Rules**:
  - "prescription" → OCRAgent
  - "segment" → SegmentationAgent
  - "phi" / "hipaa" → PHIFilterAgent
  - "drug" / "medication" → DrugInformationAgent
- **Capabilities**: Task delegation, agent discovery, priority-based routing

### 2. **OCRAgent**
- **Purpose**: Coordinates the entire OCR pipeline
- **Workflow**:
  1. Delegates segmentation to SegmentationAgent
  2. Delegates text extraction to TextRecognitionAgent
  3. Delegates PHI filtering to PHIFilterAgent
  4. Delegates drug extraction to DrugInformationAgent
- **Processing Modes**:
  - `full` - Complete pipeline (default)
  - `ocr_only` - Text extraction only
  - `segment_only` - Image segmentation only

### 3. **SegmentationAgent**
- **Purpose**: Detects and extracts regions from prescription images
- **Tools**:
  - **SAM2** (Segment Anything Model 2) - Advanced AI segmentation
  - **Fallback Contour Detection** - OpenCV-based region detection
  - **Region Extraction** - Crops and preprocesses detected regions
- **Output**: List of regions with bounding boxes and cropped images

### 4. **TextRecognitionAgent**
- **Purpose**: Extracts text from images using OCR
- **Tools**:
  - **Azure Vision OCR** - Microsoft's cloud OCR API (primary)
  - **TrOCR** - Transformer-based handwriting recognition (optional)
- **Methods**:
  - `auto` - Tries all available methods
  - `azure` - Azure Vision only
  - `trocr` - TrOCR only
- **Output**: Extracted text with confidence scores

### 5. **PHIFilterAgent**
- **Purpose**: Detects and redacts Protected Health Information
- **Tools**:
  - **HuggingFace NER** - Named Entity Recognition for person names
  - **Regex Patterns** - Pattern matching for structured PHI
- **Detects**:
  - 👤 **Names** - Patient and doctor names
  - 📅 **Dates** - Birth dates, appointment dates
  - 📞 **Phone Numbers** - All formats
  - 📧 **Email Addresses**
  - 🏠 **Addresses** - Street addresses
  - 🔢 **SSN** - Social Security Numbers
  - 🆔 **Medical IDs** - MRNs, Patient IDs
  - 💳 **Insurance Numbers**
- **Output**: Redacted text (PHI replaced with `[TYPE_REDACTED]`), PHI entity list

### 6. **DrugInformationAgent**
- **Purpose**: Extracts medications and queries drug databases
- **Tools**:
  - **Medication Extractor** - Regex-based pattern matching
  - **RxNorm API** - NIH drug database (free, international)
  - **FDA openFDA API** - US FDA drug labels database
  - **LLaMA AI** - AI-powered fallback for unknown drugs
- **Extraction Patterns**:
  - `Tab. Augmentin 625mg`
  - `Cap. Amoxicillin 500mg`
  - Drug names with common suffixes (cillin, mycin, pril, etc.)
  - 100+ common medication names
- **Output**: 
  - Medication list (name + dosage)
  - Drug alternatives with generic/brand names
  - Manufacturer information
  - Indications and usage
  - AI-generated information for unknown drugs

---

## 🛠️ Technical Stack

### Backend
- **Framework**: FastAPI (Python)
- **Agent System**: Custom multi-agent framework with async/await
- **OCR**: Azure Vision API
- **Segmentation**: SAM2 (Meta AI) + OpenCV fallback
- **NLP**: HuggingFace Transformers (NER models)
- **Drug APIs**: RxNorm (NIH), openFDA, HuggingFace LLaMA
- **Image Processing**: OpenCV, NumPy, PIL

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: CSS3 with modern features
- **State Management**: React Hooks

### APIs & Services
- **Azure Vision API** - Text extraction
- **RxNorm REST API** - Drug information
- **FDA openFDA API** - US drug labels
- **HuggingFace Inference API** - NER and LLaMA
- **Meta SAM2** - Image segmentation

---

## 📁 Project Structure

```
ocr/
├── backend/                          # Backend API server
│   ├── agent_system.py              # Agent system initialization
│   ├── main_agent_api.py            # FastAPI server with agent routes
│   ├── main.py                      # Legacy API (reference)
│   ├── requirements.txt             # Python dependencies
│   ├── .env                         # Configuration (keys, tokens)
│   ├── agents/                      # Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Base agent class
│   │   ├── orchestrator.py         # Task routing agent
│   │   ├── ocr_agent.py            # Pipeline coordinator
│   │   ├── segmentation_agent.py   # Image segmentation
│   │   ├── text_recognition_agent.py # Text extraction
│   │   ├── phi_filter_agent.py     # PHI detection/redaction
│   │   ├── drug_information_agent.py # Medication extraction
│   │   └── tools.py                # Tool implementations
│   ├── checkpoints/                 # Model checkpoints
│   │   └── sam2_hiera_large.pt     # SAM2 model (900MB)
│   └── segment-anything-2/          # SAM2 library
│
├── src/                             # React frontend
│   ├── components/
│   │   ├── LandingPage.jsx         # Main UI component
│   │   └── LandingPage.css         # Styling
│   ├── App.jsx
│   └── main.jsx
│
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+**
2. **Node.js 16+**
3. **Azure Vision API** credentials
4. **HuggingFace API** token (optional, for LLaMA)

### Backend Setup

1. **Install Python dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Install SAM2** (optional, for advanced segmentation):
```bash
cd segment-anything-2
pip install -e .
```

3. **Download SAM2 checkpoint** (optional):
```bash
cd ..
python download_sam2_checkpoint.py
```

4. **Configure environment variables** (`.env` file):
```env
# Azure Vision API (Required)
AZURE_VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_VISION_KEY=your_azure_key

# HuggingFace API (Optional - for LLaMA AI fallback)
HF_TOKEN=your_huggingface_token
```

5. **Start the agent API server**:
```bash
python main_agent_api.py
```

Server will be available at `http://localhost:8000`

### Frontend Setup

1. **Install Node dependencies**:
```bash
npm install
```

2. **Start development server**:
```bash
npm run dev
```

App will be available at `http://localhost:5173`

---

## 📡 API Endpoints

### Main Endpoints

#### `POST /api/process-image`
Process an image through the full agent pipeline.

**Request**:
```bash
curl -X POST http://localhost:8000/api/process-image \
  -F "file=@prescription.jpg" \
  -F "mode=full" \
  -F "filter_phi=true"
```

**Parameters**:
- `file` - Image file (JPEG, PNG)
- `mode` - Processing mode: `full`, `ocr_only`, `segment_only`
- `filter_phi` - Enable PHI filtering (default: true)
- `include_regions` - Include region details (default: false)

**Response**:
```json
{
  "success": true,
  "mode": "full",
  "agent_used": "OCRAgent",
  "tools_used": ["azure_vision_ocr", "filter_phi", "extract_medications"],
  "regions_detected": 3,
  "extracted_text": "Tab. Augmentin 625mg...",
  "redacted_text": "[NAME_REDACTED]...",
  "phi_summary": [
    {"type": "NAME", "original": "John Doe"},
    {"type": "DATE", "original": "12-09-2025"}
  ],
  "medications": [
    {"name": "augmentin", "dosage": "625mg"}
  ],
  "drug_alternatives": [
    {
      "original_drug": {"name": "augmentin", "dosage": "625mg"},
      "drug_info": {
        "found": true,
        "alternatives": [...],
        "source": "RxNorm (NIH)"
      }
    }
  ]
}
```

#### `POST /api/phi/filter`
Filter PHI from text directly (without image).

**Request**:
```bash
curl -X POST http://localhost:8000/api/phi/filter \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient: John Doe, Age: 45, Phone: 555-1234"}'
```

#### `GET /health`
Check system health and status.

**Response**:
```json
{
  "status": "healthy",
  "agent_system_initialized": true,
  "sam2_loaded": false,
  "trocr_loaded": false
}
```

#### `GET /api/agent-status`
Get detailed agent system status.

---

## 🔧 Configuration

### Processing Modes

- **`full`** - Complete pipeline (recommended)
  - Segmentation → OCR → PHI Filtering → Drug Extraction
  
- **`ocr_only`** - Text extraction only
  - Skips segmentation, directly extracts text
  
- **`segment_only`** - Image segmentation only
  - Returns detected regions without text extraction

### Agent Configuration

Edit `agent_system.py` to:
- Enable/disable SAM2 segmentation
- Enable/disable TrOCR handwriting recognition
- Configure model checkpoints
- Adjust agent routing rules

### PHI Filtering Rules

Customize PHI detection in `agents/phi_filter_agent.py`:
- Add custom regex patterns
- Configure NER model
- Adjust redaction format

### Drug Extraction Patterns

Customize medication extraction in `agents/drug_information_agent.py`:
- Add medication name patterns
- Configure database priorities
- Adjust extraction rules

---

## 🧪 Testing

### Test with Sample Image

```bash
# Backend
cd backend
python main_agent_api.py

# In another terminal
curl -X POST http://localhost:8000/api/process-image \
  -F "file=@test_prescription.jpg" \
  -F "mode=full"
```

### Test Individual Agents

```python
from agent_system import get_agent_system
import asyncio

async def test():
    # Initialize system
    agent_system = await get_agent_system()
    
    # Test PHI filtering
    response = await agent_system.phi_filter_agent.process(
        "Filter PHI",
        {"text": "Patient: John Doe, DOB: 01/15/1980"}
    )
    print(response.data)

asyncio.run(test())
```

---

## 🔒 Privacy & Security

### PHI Protection

- All PHI is detected and redacted before drug information queries
- No PHI is sent to external APIs (RxNorm, FDA, LLaMA)
- Redacted text uses placeholder format: `[TYPE_REDACTED]`

### HIPAA Compliance Features

- ✅ Automatic PHI detection and redaction
- ✅ No PHI logging in agent responses
- ✅ Privacy-first design (PHI filtered before external API calls)
- ⚠️ **Note**: This is a demonstration system. For production HIPAA compliance, additional measures required:
  - Encrypted storage
  - Access logging
  - Audit trails
  - BAA with cloud providers

---

## 🎨 UI Features

### Image Upload
- Drag & drop support
- Format support: JPEG, PNG
- Real-time processing feedback

### Results Display
- **Agent Badge** - Shows which agent processed the image
- **Tools Badge** - Lists tools used in pipeline
- **Text Display** - Original extracted text
- **Redacted Text** - PHI-filtered version
- **PHI Summary** - Detailed list of detected PHI entities
- **Medications** - Extracted drugs with dosages
- **Drug Alternatives** - Database results for each medication

### Interactive Features
- Toggle between original/segmented image view
- Copy text to clipboard
- Download full report with metadata
- Show/hide detailed metadata
- Region statistics display

---

## 🐛 Troubleshooting

### "Agent system not initialized"
**Solution**: Wait a few seconds after server start for initialization.

### "No text detected"
**Solutions**:
- Verify Azure Vision API credentials in `.env`
- Check image quality and resolution
- Ensure prescription is clearly visible

### "SAM2 not loaded"
**Solution**: This is optional. System uses fallback segmentation. To enable:
```bash
cd backend/segment-anything-2
pip install -e .
cd ..
python download_sam2_checkpoint.py
```

### "TrOCR not loaded"
**Solution**: Optional for handwriting. To enable:
```bash
pip install "huggingface-hub>=0.24.0,<1.0"
pip install transformers --upgrade
```

---

## 📊 Performance

### Processing Times (Approximate)

- **Segmentation**: 1-3 seconds (with fallback)
- **SAM2 Segmentation**: 5-10 seconds (if enabled)
- **Azure Vision OCR**: 2-5 seconds
- **PHI Filtering**: <1 second
- **Drug Information**: 1-3 seconds per medication

**Total Pipeline**: 5-15 seconds per image

### Optimization Tips

1. Use `ocr_only` mode if segmentation not needed
2. Disable SAM2 for faster processing (fallback is sufficient)
3. Cache drug information results
4. Process multiple images in batch mode

---

## 🔄 Future Enhancements

- [ ] Multi-language support
- [ ] Batch processing UI
- [ ] Drug interaction checker
- [ ] Dosage validation
- [ ] Prescription history tracking
- [ ] Export to PDF/JSON
- [ ] Advanced handwriting recognition
- [ ] Custom agent creation API

---

## 📝 License

This project is for educational and demonstration purposes.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- **Azure Vision API** - Microsoft Cognitive Services
- **SAM2** - Meta AI Research
- **RxNorm** - National Library of Medicine (NLM)
- **openFDA** - U.S. Food & Drug Administration
- **HuggingFace** - Transformers and Inference API
- **FastAPI** - Modern Python web framework
- **React** - Facebook Open Source
