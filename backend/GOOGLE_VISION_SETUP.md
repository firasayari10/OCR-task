# Google Cloud Vision API Setup

Google Cloud Vision API provides excellent OCR capabilities for both handwritten and printed text.

## Step 1: Install the Library

```bash
pip install google-cloud-vision
```

## Step 2: Set Up Google Cloud Project

1. **Create a Google Cloud Project** (if you don't have one):
   - Go to: https://console.cloud.google.com/
   - Click "Create Project"
   - Give it a name (e.g., "prescription-ocr")

2. **Enable Vision API**:
   - In the Google Cloud Console, go to "APIs & Services" > "Library"
   - Search for "Cloud Vision API"
   - Click "Enable"

3. **Create Service Account**:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Give it a name (e.g., "vision-api-service")
   - Click "Create and Continue"
   - Grant role: "Cloud Vision API User"
   - Click "Continue" then "Done"

4. **Create and Download Credentials**:
   - Click on the service account you just created
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose "JSON"
   - Download the JSON file
   - **Rename it to `credentials.json`** and place it in the `backend/` directory

## Step 3: Set Environment Variable (Optional)

You can also set the environment variable:

**Windows (PowerShell):**
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\credentials.json"
```

**Windows (Command Prompt):**
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\credentials.json
```

**Linux/Mac:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

## Step 4: Pricing

Google Cloud Vision API has a free tier:
- **First 1,000 units per month: FREE**
- After that: $1.50 per 1,000 units
- 1 unit = 1 image (up to 4 pages)

For most use cases, the free tier is sufficient.

## Step 5: Verify Setup

After placing `credentials.json` in the backend directory, restart the server:

```bash
python main.py
```

You should see:
```
Initializing Google Cloud Vision API...
Using credentials from: credentials.json
Google Cloud Vision API initialized successfully!
```

## Troubleshooting

1. **"Could not find credentials"**:
   - Make sure `credentials.json` is in the `backend/` directory
   - Or set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable

2. **"Permission denied"**:
   - Make sure the service account has "Cloud Vision API User" role
   - Check that Vision API is enabled in your project

3. **"Billing required"**:
   - Even with free tier, you may need to add a payment method
   - You won't be charged for the first 1,000 units/month

## Alternative: Use Without Credentials (for testing)

If you want to test without setting up credentials, the system will fall back to TrOCR automatically.



