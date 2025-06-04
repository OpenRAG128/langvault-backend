from flask import Flask, request, jsonify
from flask_cors import CORS
import io, os
import pdfplumber
import docx
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from generation_utils import should_skip_page, build_prompt, clean_translation
import logging
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Enable CORS for cross-origin requests from Vercel frontend
CORS(app, origins=["*"])  # Configure with your Vercel domain in production

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure logging for Azure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# In-memory cache with session management (consider Redis for production)
translation_sessions = {}

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
SUPPORTED_LANGUAGES = {
    'hindi', 'marathi', 'bengali', 'telugu', 'tamil', 'kannada', 'malayalam',
    'punjabi', 'gujarati', 'oriya', 'assamese', 'urdu', 'sanskrit', 'maithili',
    'konkani', 'manipuri', 'kashmiri', 'sindhi', 'dogri', 'bodo', 'santhali',
    'french', 'spanish', 'german', 'italian', 'portuguese', 'russian',
    'chinese', 'japanese', 'korean', 'arabic'
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_language(language):
    """Validate target language"""
    return language and language.lower() in SUPPORTED_LANGUAGES

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF file"""
    try:
        text_content = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_content.append({
                        "page": i + 1,
                        "content": text.strip()
                    })
        return text_content
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_docx(file_bytes):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text_content = []
        current_page = 1
        page_text = ""
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                page_text += paragraph.text.strip() + "\n"
                # Simple page break detection (could be improved)
                if len(page_text) > 2000:  # Approximate page break
                    text_content.append({
                        "page": current_page,
                        "content": page_text.strip()
                    })
                    current_page += 1
                    page_text = ""
        
        # Add remaining text
        if page_text.strip():
            text_content.append({
                "page": current_page,
                "content": page_text.strip()
            })
        
        return text_content
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {str(e)}")
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")

def extract_text_from_txt(file_bytes):
    """Extract text from TXT file"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        text = None
        
        for encoding in encodings:
            try:
                text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            raise ValueError("Unable to decode text file with supported encodings")
        
        # Split into chunks for better processing
        text_content = []
        chunks = text.split('\n\n')  # Split by double newlines
        current_page = 1
        page_text = ""
        
        for chunk in chunks:
            if chunk.strip():
                page_text += chunk.strip() + "\n\n"
                # Create pages of reasonable size
                if len(page_text) > 2000:
                    text_content.append({
                        "page": current_page,
                        "content": page_text.strip()
                    })
                    current_page += 1
                    page_text = ""
        
        # Add remaining text
        if page_text.strip():
            text_content.append({
                "page": current_page,
                "content": page_text.strip()
            })
        
        return text_content
    except Exception as e:
        logger.error(f"Error extracting TXT text: {str(e)}")
        raise ValueError(f"Failed to extract text from TXT: {str(e)}")

def extract_text_with_metadata(file_bytes, file_extension):
    """Extract text and metadata from different file types"""
    try:
        if file_extension == 'pdf':
            text_content = extract_text_from_pdf(file_bytes)
        elif file_extension == 'docx':
            text_content = extract_text_from_docx(file_bytes)
        elif file_extension == 'txt':
            text_content = extract_text_from_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Create metadata
        total_chars = sum(len(item['content']) for item in text_content)
        total_words = sum(len(item['content'].split()) for item in text_content)
        
        metadata = {
            "document": {
                "total_pages": len(text_content),
                "total_characters": total_chars,
                "total_words": total_words,
                "file_type": file_extension.upper(),
                "creation_date": datetime.now().isoformat(),
                "file_size": len(file_bytes)
            },
            "pages": [
                {
                    "page_number": item['page'],
                    "char_count": len(item['content']),
                    "word_count": len(item['content'].split())
                }
                for item in text_content
            ]
        }
        
        return [item['content'] for item in text_content], metadata
        
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise ValueError(f"Failed to extract text: {str(e)}")

def translate_text_batch(pages, target_language, llm):
    """Translate multiple pages efficiently"""
    translated_pages = []
    
    for i, text in enumerate(pages):
        logger.info(f"Translating page {i+1}/{len(pages)}")
        
        if should_skip_page(text):
            translated_pages.append("[Skipped: Insufficient meaningful content]")
            continue
        
        try:
            prompt = build_prompt(text, target_language)
            result = llm.predict(prompt)
            cleaned_result = clean_translation(result.strip())
            translated_pages.append(cleaned_result)
        except Exception as e:
            logger.error(f"Translation error for page {i+1}: {str(e)}")
            translated_pages.append(f"[Translation Error: {str(e)}]")
    
    return translated_pages

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for Azure deployment"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024)
    })

@app.route('/api/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages"""
    return jsonify({
        "languages": sorted(list(SUPPORTED_LANGUAGES)),
        "count": len(SUPPORTED_LANGUAGES)
    })

@app.route('/api/translate', methods=['POST'])
def translate_document():
    """Upload and translate document in one step"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        if 'language' not in request.form:
            return jsonify({"error": "Target language not specified"}), 400
        
        file = request.files['file']
        target_language = request.form['language'].lower().strip()
        
        # Validate file
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type. Supported formats: PDF, DOCX, TXT",
                "supported_formats": list(ALLOWED_EXTENSIONS)
            }), 400
        
        # Validate language
        if not validate_language(target_language):
            return jsonify({
                "error": f"Unsupported language: {target_language}",
                "supported_languages": sorted(list(SUPPORTED_LANGUAGES))
            }), 400
        
        # Get file extension
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        
        # Read and validate file
        file_bytes = file.read()
        if len(file_bytes) == 0:
            return jsonify({"error": "Empty file uploaded"}), 400
        
        if len(file_bytes) > MAX_FILE_SIZE:
            return jsonify({
                "error": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            }), 400
        
        # Extract text
        logger.info(f"Processing file: {secure_filename(file.filename)}, Language: {target_language}")
        pages, metadata = extract_text_with_metadata(file_bytes, file_extension)
        
        if not pages:
            return jsonify({"error": "No text content found in the document"}), 400
        
        # Initialize LLM
        if not GROQ_API_KEY:
            return jsonify({"error": "Translation service not configured"}), 500
        
        llm = ChatGroq(model="gemma2-9b-it", temperature=0, api_key=GROQ_API_KEY)
        
        # Translate document
        logger.info(f"Starting translation of {len(pages)} pages to {target_language}")
        translated_pages = translate_text_batch(pages, target_language, llm)
        
        # Prepare response
        translation_id = str(uuid.uuid4())
        
        # Calculate translation statistics
        total_original_chars = sum(len(page) for page in pages)
        total_translated_chars = sum(len(page) for page in translated_pages if not page.startswith('['))
        successful_translations = len([p for p in translated_pages if not p.startswith('[')])
        
        response_data = {
            "translation_id": translation_id,
            "status": "completed",
            "original_text": pages,
            "translated_text": translated_pages,
            "metadata": metadata,
            "translation_stats": {
                "total_pages": len(pages),
                "successful_translations": successful_translations,
                "failed_translations": len(pages) - successful_translations,
                "original_characters": total_original_chars,
                "translated_characters": total_translated_chars,
                "target_language": target_language.title(),
                "source_file_type": file_extension.upper()
            },
            "completed_at": datetime.now().isoformat(),
            "message": f"Successfully translated {successful_translations}/{len(pages)} pages to {target_language.title()}"
        }
        
        logger.info(f"Translation completed successfully. ID: {translation_id}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

@app.route('/api/translate-text', methods=['POST'])
def translate_text_direct():
    """Translate text directly without file upload"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get('text', '').strip()
        target_language = data.get('language', '').lower().strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if not target_language:
            return jsonify({"error": "Target language not specified"}), 400
        
        # Validate language
        if not validate_language(target_language):
            return jsonify({
                "error": f"Unsupported language: {target_language}",
                "supported_languages": sorted(list(SUPPORTED_LANGUAGES))
            }), 400
        
        # Check text length
        if len(text) > 10000:  # 10K character limit for direct text
            return jsonify({"error": "Text too long. Maximum 10,000 characters for direct translation"}), 400
        
        # Initialize LLM
        if not GROQ_API_KEY:
            return jsonify({"error": "Translation service not configured"}), 500
        
        llm = ChatGroq(model="gemma2-9b-it", temperature=0, api_key=GROQ_API_KEY)
        
        # Translate text
        logger.info(f"Translating direct text to {target_language}")
        
        try:
            prompt = build_prompt(text, target_language)
            result = llm.predict(prompt)
            translated_text = clean_translation(result.strip())
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return jsonify({"error": f"Translation failed: {str(e)}"}), 500
        
        # Prepare response
        translation_id = str(uuid.uuid4())
        
        response_data = {
            "translation_id": translation_id,
            "status": "completed",
            "original_text": text,
            "translated_text": translated_text,
            "translation_stats": {
                "original_characters": len(text),
                "translated_characters": len(translated_text),
                "target_language": target_language.title()
            },
            "completed_at": datetime.now().isoformat(),
            "message": f"Text successfully translated to {target_language.title()}"
        }
        
        logger.info(f"Direct text translation completed. ID: {translation_id}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Direct text translation failed: {str(e)}")
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

@app.route('/api/batch-translate', methods=['POST'])
def batch_translate():
    """Translate multiple texts in a single request"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        texts = data.get('texts', [])
        target_language = data.get('language', '').lower().strip()
        
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "No texts array provided"}), 400
        
        if len(texts) > 10:  # Limit batch size
            return jsonify({"error": "Maximum 10 texts per batch"}), 400
        
        if not target_language:
            return jsonify({"error": "Target language not specified"}), 400
        
        # Validate language
        if not validate_language(target_language):
            return jsonify({
                "error": f"Unsupported language: {target_language}",
                "supported_languages": sorted(list(SUPPORTED_LANGUAGES))
            }), 400
        
        # Validate text lengths
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                return jsonify({"error": f"Text {i+1} is not a string"}), 400
            if len(text) > 5000:  # 5K limit per text in batch
                return jsonify({"error": f"Text {i+1} too long. Maximum 5,000 characters per text in batch"}), 400
        
        # Initialize LLM
        if not GROQ_API_KEY:
            return jsonify({"error": "Translation service not configured"}), 500
        
        llm = ChatGroq(model="gemma2-9b-it", temperature=0, api_key=GROQ_API_KEY)
        
        # Translate texts
        logger.info(f"Batch translating {len(texts)} texts to {target_language}")
        translated_texts = []
        
        for i, text in enumerate(texts):
            try:
                if not text.strip():
                    translated_texts.append("")
                    continue
                    
                prompt = build_prompt(text, target_language)
                result = llm.predict(prompt)
                translated_text = clean_translation(result.strip())
                translated_texts.append(translated_text)
            except Exception as e:
                logger.error(f"Translation error for text {i+1}: {str(e)}")
                translated_texts.append(f"[Translation Error: {str(e)}]")
        
        # Prepare response
        batch_id = str(uuid.uuid4())
        
        response_data = {
            "batch_id": batch_id,
            "status": "completed",
            "results": [
                {
                    "index": i,
                    "original_text": texts[i],
                    "translated_text": translated_texts[i],
                    "success": not translated_texts[i].startswith("[Translation Error:")
                }
                for i in range(len(texts))
            ],
            "translation_stats": {
                "total_texts": len(texts),
                "successful_translations": len([t for t in translated_texts if not t.startswith("[Translation Error:")]),
                "failed_translations": len([t for t in translated_texts if t.startswith("[Translation Error:")]),
                "target_language": target_language.title()
            },
            "completed_at": datetime.now().isoformat(),
            "message": f"Batch translation completed for {len(texts)} texts"
        }
        
        logger.info(f"Batch translation completed. ID: {batch_id}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Batch translation failed: {str(e)}")
        return jsonify({"error": f"Batch translation failed: {str(e)}"}), 500

@app.route('/api/detect-language', methods=['POST'])
def detect_language():
    """Detect the language of provided text (basic implementation)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Simple language detection based on character patterns
        # This is a basic implementation - consider using a proper language detection library
        def simple_language_detect(text):
            # Check for common script patterns
            if any('\u0900' <= char <= '\u097F' for char in text):
                return 'hindi'
            elif any('\u0A00' <= char <= '\u0A7F' for char in text):
                return 'punjabi'
            elif any('\u0980' <= char <= '\u09FF' for char in text):
                return 'bengali'
            elif any('\u0600' <= char <= '\u06FF' for char in text):
                return 'arabic'
            elif any('\u4e00' <= char <= '\u9fff' for char in text):
                return 'chinese'
            elif any('\u3040' <= char <= '\u309f' for char in text) or any('\u30a0' <= char <= '\u30ff' for char in text):
                return 'japanese'
            elif any('\uac00' <= char <= '\ud7af' for char in text):
                return 'korean'
            else:
                return 'english'  # Default fallback
        
        detected_language = simple_language_detect(text)
        
        response_data = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "detected_language": detected_language,
            "confidence": "basic_detection",  # Placeholder since this is a simple implementation
            "message": f"Detected language: {detected_language.title()}"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        return jsonify({"error": f"Language detection failed: {str(e)}"}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    # For local development
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
