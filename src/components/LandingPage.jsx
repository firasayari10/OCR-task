import React, { useState, useEffect } from 'react'
import './LandingPage.css'

const LandingPage = () => {
  const [isScrolled, setIsScrolled] = useState(false)
  const [uploadedImage, setUploadedImage] = useState(null)
  const [ocrText, setOcrText] = useState('')
  const [extractedText, setExtractedText] = useState('') // Store just the extracted text
  const [isProcessing, setIsProcessing] = useState(false)
  const [segmentationResults, setSegmentationResults] = useState(null)
  const [overlayImage, setOverlayImage] = useState(null)
  const [phiSummary, setPhiSummary] = useState([])
  const [medications, setMedications] = useState([])
  const [fdaAlternatives, setFdaAlternatives] = useState([])
  const [apiError, setApiError] = useState(null)
  const [showOverlay, setShowOverlay] = useState(true)
  const [showMetadata, setShowMetadata] = useState(false) // Toggle for metadata

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToSection = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleImageUpload = async (e) => {
    const file = e.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setUploadedImage(reader.result)
      }
      reader.readAsDataURL(file)
      
      // Perform segmentation with SAM2
      await performSegmentation(file)
    }
  }

  const performSegmentation = async (file) => {
    setIsProcessing(true)
    setApiError(null)
    setSegmentationResults(null)
    setOverlayImage(null)
    setOcrText('')
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('mode', 'full') // full, ocr_only, or segment_only
      formData.append('filter_phi', 'true')
      formData.append('include_regions', 'true')
      
      // Use the new agent API endpoint
      const response = await fetch('http://localhost:8000/api/process-image', {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      
        if (data.success) {
        // Agent API response format
        const extractedTextValue = data.extracted_text || ''
        const redactedTextValue = data.redacted_text || extractedTextValue
        const phiSummaryValue = data.phi_summary || []
        const phiTypesValue = data.phi_types || {}
        const regionsDetected = data.regions_detected || 0
        const agentUsed = data.agent_used || 'Unknown'
        const toolsUsed = data.tools_used || []
          
        // Store the extracted and redacted text
        setExtractedText(extractedTextValue || '(No text detected)')
        // Store PHI summary for display
        setPhiSummary(phiSummaryValue)
        // Store medications and drug alternatives from agent
        setMedications(data.medications || [])
        setFdaAlternatives(data.drug_alternatives || [])
        
        // Format detailed OCR text output with metadata (for download/copy)
        let textOutput = `Agent-Based OCR Results:\n`
        textOutput += `${'='.repeat(50)}\n\n`
        textOutput += `Agent Used: ${agentUsed}\n`
        textOutput += `Tools Used: ${toolsUsed.join(', ')}\n`
        textOutput += `Processing Mode: ${data.mode || 'full'}\n`
        textOutput += `Regions Detected: ${regionsDetected}\n\n`
        textOutput += `${'='.repeat(50)}\n\n`
        textOutput += `Extracted Text:\n`
        textOutput += `${'='.repeat(50)}\n\n`
        textOutput += extractedTextValue || '(No text detected)'
        textOutput += `\n\n${'='.repeat(50)}\n\n`
        textOutput += `Redacted Text (PHI Filtered):\n`
        textOutput += `${'='.repeat(50)}\n\n`
        textOutput += redactedTextValue
        textOutput += `\n\n${'='.repeat(50)}\n`
        textOutput += `Text Length: ${extractedTextValue.length} characters\n`
        textOutput += `PHI Entities Found: ${phiSummaryValue.length}\n`
        
        // Add PHI types summary
        if (Object.keys(phiTypesValue).length > 0) {
          textOutput += `\nPHI Types Detected:\n`
          textOutput += `${'='.repeat(50)}\n`
          Object.entries(phiTypesValue).forEach(([type, count]) => {
            textOutput += `  ${type}: ${count}\n`
          })
        }
        
        setOcrText(textOutput)
        
        // Store segmentation results for compatibility
        setSegmentationResults({
          success: true,
          total_regions: regionsDetected,
          handwritten_regions: 0, // Agent API doesn't distinguish yet
          extracted_text: extractedTextValue,
          redacted_text: redactedTextValue,
          phi_summary: phiSummaryValue,
          phi_types: phiTypesValue,
          agent_used: agentUsed,
          tools_used: toolsUsed
        })
        
        // Use images from agent API
        if (data.annotated_image) {
          setOverlayImage(data.annotated_image)
        } else if (data.original_image) {
          setOverlayImage(data.original_image)
        }
      } else {
        throw new Error(data.error || 'Processing failed')
      }
    } catch (error) {
      console.error('Segmentation error:', error)
      const errorMessage = error.message || 'Failed to process image. Make sure the backend server is running.'
      setApiError(errorMessage)
      const errorText = `Error: ${errorMessage}\n\nPlease ensure:\n1. Backend server is running (http://localhost:8000)\n2. SAM2 model is loaded\n3. Image format is supported`
      setOcrText(errorText)
      setExtractedText(errorText)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleRemoveImage = () => {
    setUploadedImage(null)
    setOcrText('')
    setExtractedText('')
    setSegmentationResults(null)
    setOverlayImage(null)
    setApiError(null)
    setShowMetadata(false)
  }

  return (
    <div className="landing-page">
      {/* Navigation */}
      <nav className={`navbar ${isScrolled ? 'scrolled' : ''}`}>
        <div className="nav-container">
          <div className="logo">YourBrand</div>
          <ul className="nav-menu">
            <li><a href="#home" onClick={(e) => { e.preventDefault(); scrollToSection('home') }}>Home</a></li>
            <li><a href="#ocr" onClick={(e) => { e.preventDefault(); scrollToSection('ocr') }}>OCR Scanner</a></li>
            <li><a href="#features" onClick={(e) => { e.preventDefault(); scrollToSection('features') }}>Features</a></li>
            <li><a href="#about" onClick={(e) => { e.preventDefault(); scrollToSection('about') }}>About</a></li>
            <li><a href="#contact" onClick={(e) => { e.preventDefault(); scrollToSection('contact') }}>Contact</a></li>
          </ul>
          <button className="cta-button">Get Started</button>
        </div>
      </nav>

      {/* Hero Section */}
      <section id="home" className="hero">
        <div className="hero-content">
          <h1 className="hero-title">
            Prescription OCR
            <span className="gradient-text"> Scanner</span>
          </h1>
          <p className="hero-subtitle">
            Upload your doctor's prescription and extract text instantly with our advanced OCR technology.
            Fast, accurate, and secure.
          </p>
        </div>
      </section>

      {/* OCR Upload Section */}
      <section id="ocr" className="ocr-section">
        <div className="container">
          <div className="ocr-container">
            {/* Upload Section */}
            <div className="upload-section">
              <h2 className="section-title-small">Upload Prescription</h2>
              <div className="upload-area">
                {uploadedImage ? (
                  <div className="image-preview-container">
                    <img 
                      src={showOverlay && overlayImage ? overlayImage : uploadedImage} 
                      alt={showOverlay && overlayImage ? "Segmented prescription" : "Uploaded prescription"} 
                      className="uploaded-image" 
                    />
                    <button className="remove-image-btn" onClick={handleRemoveImage}>
                      ✕ Remove
                    </button>
                    {overlayImage && segmentationResults && (
                      <div className="view-toggle">
                        <button 
                          className={`toggle-btn ${showOverlay ? 'active' : ''}`}
                          onClick={() => setShowOverlay(true)}
                          title="Show segmentation overlay (Red=Handwritten, Green=Printed)"
                        >
                          🎨 Segmented
                        </button>
                        <button 
                          className={`toggle-btn ${!showOverlay ? 'active' : ''}`}
                          onClick={() => setShowOverlay(false)}
                          title="Show original image"
                        >
                          📷 Original
                        </button>
                      </div>
                    )}
                    {segmentationResults && (
                      <div className="segmentation-stats">
                        <div className="stat-badge" style={{background: 'rgba(102, 126, 234, 0.9)', color: 'white'}}>
                          🤖 Agent: {segmentationResults.agent_used || 'OCRAgent'}
                        </div>
                        {segmentationResults.total_regions > 0 && (
                          <div className="stat-badge" style={{background: 'rgba(76, 175, 80, 0.9)', color: 'white'}}>
                            📍 Regions: {segmentationResults.total_regions}
                          </div>
                        )}
                        <div className="stat-badge" style={{background: 'rgba(255, 152, 0, 0.9)', color: 'white'}}>
                          📝 Text: {segmentationResults.extracted_text?.length || 0} chars
                        </div>
                        {segmentationResults.tools_used && segmentationResults.tools_used.length > 0 && (
                          <div className="stat-badge" style={{background: 'rgba(156, 39, 176, 0.9)', color: 'white', marginTop: '0.5rem', fontSize: '0.75rem'}}>
                            🔧 Tools: {segmentationResults.tools_used.join(', ')}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <label htmlFor="prescription-upload" className="upload-label">
                    <div className="upload-icon">📄</div>
                    <div className="upload-text">
                      <p className="upload-title">Click to upload or drag and drop</p>
                      <p className="upload-subtitle">PNG, JPG, PDF up to 10MB</p>
                    </div>
                    <input
                      type="file"
                      id="prescription-upload"
                      accept="image/*,.pdf"
                      onChange={handleImageUpload}
                      className="upload-input"
                    />
                  </label>
                )}
              </div>
              {uploadedImage && (
                <div className="upload-actions">
                  <button 
                    className="btn btn-primary upload-again-btn" 
                    onClick={() => document.getElementById('prescription-upload').click()}
                  >
                    Upload Another Image
                  </button>
                  {apiError && (
                    <div className="error-message">
                      ⚠️ {apiError}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* OCR Results Section */}
            <div className="ocr-results-section">
              <h2 className="section-title-small">Extracted Text</h2>
              <div className="ocr-results-container">
                {isProcessing ? (
                  <div className="processing-indicator">
                    <div className="spinner"></div>
                    <p>Processing image...</p>
                  </div>
                ) : extractedText ? (
                  <div className="ocr-text-display">
                    {/* Main extracted text display */}
                    <div className="extracted-text-main">
                      <div className="extracted-text-content">
                        {extractedText}
                      </div>
                    </div>

                    {/* PHI Redaction & NER Summary */}
                    <div className="phi-redaction-section">
                      <h3>🔒 Redacted Text (PHI Filtered)</h3>
                      <div className="redacted-text">
                        {segmentationResults?.redacted_text || extractedText || '(No text)'}
                      </div>

                      <h4>⚠️ PHI Entities Detected</h4>
                      {phiSummary && phiSummary.length > 0 ? (
                        <>
                          <div className="phi-summary-stats">
                            {segmentationResults?.phi_types && Object.entries(segmentationResults.phi_types).map(([type, count]) => (
                              <span key={type} className="phi-type-badge">
                                {type}: {count}
                              </span>
                            ))}
                          </div>
                          <div className="phi-list">
                            {phiSummary.map((item, idx) => (
                              <div key={idx} className="phi-item">
                                <div className="phi-type">{item.type}</div>
                                <div className="phi-original">Removed: {item.original}</div>
                              </div>
                            ))}
                          </div>
                        </>
                      ) : (
                        <div className="no-phi">✅ No PHI detected</div>
                      )}
                    </div>

                    {/* Medications & FDA Alternatives Section */}
                    {medications && medications.length > 0 && (
                      <div className="medications-section">
                        <h3>Detected Medications</h3>
                        <div className="medications-list">
                          {medications.map((med, idx) => (
                            <div key={idx} className="medication-item">
                              <div className="med-name">{med.name}</div>
                              {med.dosage && <div className="med-dosage">Dosage: {med.dosage}</div>}
                            </div>
                          ))}
                        </div>

                        {fdaAlternatives && fdaAlternatives.length > 0 && (
                          <div className="fda-alternatives">
                            <h4>Drug Information & Alternatives</h4>
                            {fdaAlternatives.map((item, idx) => (
                              <div key={idx} className="fda-item">
                                <h5>Original: {item.original_drug.name} {item.original_drug.dosage}</h5>
                                
                                {/* Show alternatives from database APIs if available */}
                                {item.drug_info.alternatives && item.drug_info.alternatives.length > 0 && (
                                  <div className="alternatives-grid">
                                    {item.drug_info.alternatives.slice(0, 5).map((alt, altIdx) => (
                                      <div key={altIdx} className="alternative-card">
                                        <div className="alt-generic">{alt.generic_name}</div>
                                        {alt.brand_names && alt.brand_names.length > 0 && (
                                          <div className="alt-brands">Brands: {alt.brand_names.join(', ')}</div>
                                        )}
                                        <div className="alt-manufacturer">{alt.manufacturer}</div>
                                        <div className="alt-indication">{alt.indication.substring(0, 100)}...</div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                                
                                {/* Show LLM response if available */}
                                {item.drug_info.text_from_llm && (
                                  <div className="llm-response">
                                    <h6>🤖 Information from AI Assistant:</h6>
                                    <div className="llm-content">
                                      {item.drug_info.text_from_llm}
                                    </div>
                                    <div className="llm-disclaimer">
                                      ⚠️ This information is AI-generated. Always consult a healthcare professional.
                                    </div>
                                  </div>
                                )}
                                
                                <div className="source-badge">Source: {item.drug_info.source}</div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Metadata toggle and display */}
                    {segmentationResults && (
                      <div className="metadata-section">
                        <button 
                          className="metadata-toggle"
                          onClick={() => setShowMetadata(!showMetadata)}
                        >
                          {showMetadata ? '▼' : '▶'} {showMetadata ? 'Hide' : 'Show'} Details
                        </button>
                        {showMetadata && (
                          <div className="metadata-content">
                            <div className="metadata-item">
                              <strong>🤖 Agent Used:</strong> {segmentationResults.agent_used || 'Unknown'}
                            </div>
                            {segmentationResults.tools_used && segmentationResults.tools_used.length > 0 && (
                              <div className="metadata-item">
                                <strong>🔧 Tools Used:</strong> {segmentationResults.tools_used.join(', ')}
                              </div>
                            )}
                            {segmentationResults.total_regions > 0 && (
                              <div className="metadata-item">
                                <strong>📍 Total Regions:</strong> {segmentationResults.total_regions}
                              </div>
                            )}
                            <div className="metadata-item">
                              <strong>📝 Text Length:</strong> {extractedText.length} characters
                            </div>
                            {segmentationResults.phi_summary && segmentationResults.phi_summary.length > 0 && (
                              <div className="metadata-item">
                                <strong>⚠️ PHI Entities:</strong> {segmentationResults.phi_summary.length} found
                              </div>
                            )}
                            {segmentationResults.phi_types && Object.keys(segmentationResults.phi_types).length > 0 && (
                              <div className="metadata-item">
                                <strong>🔒 PHI Types:</strong>
                                <div style={{marginTop: '0.5rem'}}>
                                  {Object.entries(segmentationResults.phi_types).map(([type, count]) => (
                                    <div key={type} style={{marginLeft: '1rem'}}>
                                      {type}: {count}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Action buttons */}
                    <div className="ocr-actions">
                      <button 
                        className="btn btn-secondary" 
                        onClick={() => navigator.clipboard.writeText(extractedText)}
                        title="Copy extracted text"
                      >
                        📋 Copy Text
                      </button>
                      <button 
                        className="btn btn-secondary" 
                        onClick={() => {
                          const blob = new Blob([ocrText], { type: 'text/plain' })
                          const url = URL.createObjectURL(blob)
                          const a = document.createElement('a')
                          a.href = url
                          a.download = 'extracted-text.txt'
                          a.click()
                        }}
                        title="Download full report with metadata"
                      >
                        💾 Download Report
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="ocr-placeholder">
                    <div className="placeholder-icon">🔍</div>
                    <p>Upload a prescription image to extract text</p>
                    <p className="placeholder-subtitle">OCR results will appear here</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features">
        <div className="container">
          <h2 className="section-title">Why Choose Us</h2>
          <p className="section-subtitle">Discover what makes us different</p>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Lightning Fast</h3>
              <p>Experience blazing-fast performance with our optimized infrastructure.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🎨</div>
              <h3>Beautiful Design</h3>
              <p>Stunning, modern interfaces that your users will love.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🛡️</div>
              <h3>Secure & Safe</h3>
              <p>Enterprise-grade security to protect your data and privacy.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📱</div>
              <h3>Responsive</h3>
              <p>Works perfectly on all devices, from mobile to desktop.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🌐</div>
              <h3>Global Reach</h3>
              <p>Available worldwide with 24/7 support in multiple languages.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">💡</div>
              <h3>Innovative</h3>
              <p>Cutting-edge technology that keeps you ahead of the competition.</p>
            </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="about">
        <div className="container">
          <div className="about-content">
            <div className="about-text">
              <h2 className="section-title">About Us</h2>
              <p>
                We are a team of passionate developers, designers, and innovators
                dedicated to creating exceptional digital experiences. With years
                of experience and a commitment to excellence, we deliver solutions
                that exceed expectations.
              </p>
              <p>
                Our mission is to empower businesses and individuals with tools
                that make a real difference. We believe in the power of technology
                to transform lives and drive positive change.
              </p>
              <div className="stats">
                <div className="stat-item">
                  <div className="stat-number">10K+</div>
                  <div className="stat-label">Happy Customers</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number">50+</div>
                  <div className="stat-label">Countries</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number">99.9%</div>
                  <div className="stat-label">Uptime</div>
                </div>
              </div>
            </div>
            <div className="about-image">
              <div className="image-placeholder">
                <div className="placeholder-content">
                  <span>Your Image Here</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="contact">
        <div className="container">
          <h2 className="section-title">Get In Touch</h2>
          <p className="section-subtitle">We'd love to hear from you</p>
          <div className="contact-content">
            <form className="contact-form">
              <div className="form-group">
                <input type="text" placeholder="Your Name" required />
              </div>
              <div className="form-group">
                <input type="email" placeholder="Your Email" required />
              </div>
              <div className="form-group">
                <textarea placeholder="Your Message" rows="5" required></textarea>
              </div>
              <button type="submit" className="btn btn-primary">Send Message</button>
            </form>
            <div className="contact-info">
              <div className="info-item">
                <div className="info-icon">📧</div>
                <div>
                  <h4>Email</h4>
                  <p>contact@yourbrand.com</p>
                </div>
              </div>
              <div className="info-item">
                <div className="info-icon">📞</div>
                <div>
                  <h4>Phone</h4>
                  <p>+1 (555) 123-4567</p>
                </div>
              </div>
              <div className="info-item">
                <div className="info-icon">📍</div>
                <div>
                  <h4>Address</h4>
                  <p>123 Innovation Street<br />Tech City, TC 12345</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-section">
              <h3>YourBrand</h3>
              <p>Building the future, one innovation at a time.</p>
            </div>
            <div className="footer-section">
              <h4>Quick Links</h4>
              <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#features">Features</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
              </ul>
            </div>
            <div className="footer-section">
              <h4>Legal</h4>
              <ul>
                <li><a href="#">Privacy Policy</a></li>
                <li><a href="#">Terms of Service</a></li>
                <li><a href="#">Cookie Policy</a></li>
              </ul>
            </div>
            <div className="footer-section">
              <h4>Follow Us</h4>
              <div className="social-links">
                <a href="#" aria-label="Twitter">🐦</a>
                <a href="#" aria-label="Facebook">📘</a>
                <a href="#" aria-label="Instagram">📷</a>
                <a href="#" aria-label="LinkedIn">💼</a>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            <p>&copy; 2024 YourBrand. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default LandingPage

