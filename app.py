from flask import Flask, render_template, request, send_file, jsonify
import io, os
import pdfplumber
import fitz  # PyMuPDF
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from generation_utils import should_skip_page, build_prompt, clean_translation
import logging
from datetime import datetime

app = Flask(__name__)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for translated content
translated_content_cache = {
    "text": "",
    "pages": [],
    "original_pages": [],
    "metadata": {}
}

def extract_text_with_metadata(pdf_bytes):
    """Extract text and metadata from PDF pages"""
    text_pages = []
    metadata = []
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            doc_info = {
                "total_pages": len(pdf.pages),
                "creation_date": datetime.now().isoformat()
            }
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                page_text = text or ""
                text_pages.append(page_text)
                
                # Extract page metadata
                page_meta = {
                    "page_number": i + 1,
                    "char_count": len(page_text),
                    "word_count": len(page_text.split()) if page_text else 0,
                    "bbox": page.bbox if hasattr(page, 'bbox') else None
                }
                metadata.append(page_meta)
                
        return text_pages, {"document": doc_info, "pages": metadata}
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise

def create_professional_pdf(original_pdf_bytes, translated_pages, original_pages, target_language):
    """Create a professional-looking bilingual PDF with proper formatting and Unicode support"""
    try:
        doc = fitz.open(stream=original_pdf_bytes, filetype="pdf")
        new_doc = fitz.open()  # Create new document for better control
        
        # Define styling constants
        MARGIN = 40
        HEADER_HEIGHT = 60
        FOOTER_HEIGHT = 40
        COLUMN_GAP = 20
        
        def safe_text_encode(text):
            """Safely encode text to handle Unicode characters"""
            if not text:
                return ""
            try:
                # Try to handle various encodings
                if isinstance(text, bytes):
                    text = text.decode('utf-8', errors='replace')
                # Replace problematic characters that might not render
                text = text.replace('\u00a0', ' ')  # Non-breaking space
                text = text.replace('\ufeff', '')   # BOM
                return text
            except Exception as e:
                logger.warning(f"Text encoding issue: {e}")
                return str(text)
        
        def get_unicode_safe_font():
            """Get fonts that can handle Unicode characters better"""
            # For Unicode support, we'll use basic fonts and handle encoding properly
            return "helv"  # Use basic Helvetica as it's most reliable
        
        def insert_text_safely(page, point_or_rect, text, fontname="helv", fontsize=10, color=(0,0,0), align=None):
            """Safely insert text with proper encoding and error handling"""
            try:
                safe_text = safe_text_encode(text)
                if isinstance(point_or_rect, fitz.Rect) and align:
                    # Use textbox for alignment
                    return page.insert_textbox(point_or_rect, safe_text, 
                                             fontname=fontname, fontsize=fontsize, 
                                             color=color, align=align)
                else:
                    # Use simple text insertion
                    if isinstance(point_or_rect, fitz.Rect):
                        point = (point_or_rect.x0 + 5, point_or_rect.y0 + 15)
                    else:
                        point = point_or_rect
                    return page.insert_text(point, safe_text, 
                                          fontname=fontname, fontsize=fontsize, color=color)
            except Exception as e:
                logger.warning(f"Text insertion failed: {e}")
                # Fallback: insert error message
                try:
                    error_text = "[Text rendering error - check character encoding]"
                    if isinstance(point_or_rect, fitz.Rect):
                        point = (point_or_rect.x0 + 5, point_or_rect.y0 + 15)
                    else:
                        point = point_or_rect
                    return page.insert_text(point, error_text, 
                                          fontname="helv", fontsize=fontsize, color=(0.8,0,0))
                except:
                    return None
        
        # Get working font
        regular_font = get_unicode_safe_font()
        bold_font = regular_font
        italic_font = regular_font
        
        # Language labels
        lang_labels = {
    "hindi": "Hindi Translation",
    "marathi": "Marathi Translation",
    "bengali": "Bengali Translation",
    "telugu": "Telugu Translation",
    "tamil": "Tamil Translation",
    "kannada": "Kannada Translation",
    "malayalam": "Malayalam Translation",
    "punjabi": "Punjabi Translation",
    "gujarati": "Gujarati Translation",
    "oriya": "Odia Translation",
    "assamese": "Assamese Translation",
    "urdu": "Urdu Translation",
    "sanskrit": "Sanskrit Translation",
    "maithili": "Maithili Translation",
    "konkani": "Konkani Translation",
    "manipuri": "Manipuri Translation",
    "kashmiri": "Kashmiri Translation",
    "sindhi": "Sindhi Translation",
    "dogri": "Dogri Translation",
    "bodo": "Bodo Translation",
    "santhali": "Santhali Translation",
    "french": "French Translation",
    "spanish": "Spanish Translation",
    "german": "German Translation",
    "italian": "Italian Translation",
    "portuguese": "Portuguese Translation",
    "russian": "Russian Translation",
    "chinese": "Chinese Translation",
    "japanese": "Japanese Translation",
    "korean": "Korean Translation",
    "arabic": "Arabic Translation"
    }

        
        original_label = "Original Text"
        translated_label = lang_labels.get(target_language.lower(), f"{target_language.title()} Translation")
        
        for page_num, (original_text, translated_text) in enumerate(zip(original_pages, translated_pages)):
            # Create new page with A4 dimensions
            new_page = new_doc.new_page(width=595, height=842)  # A4 size
            page_rect = new_page.rect
            
            # Calculate content area
            content_rect = fitz.Rect(
                MARGIN, 
                MARGIN + HEADER_HEIGHT, 
                page_rect.width - MARGIN, 
                page_rect.height - MARGIN - FOOTER_HEIGHT
            )
            
            # Draw header
            header_rect = fitz.Rect(MARGIN, MARGIN, page_rect.width - MARGIN, MARGIN + HEADER_HEIGHT - 10)
            new_page.draw_rect(header_rect, color=(0.9, 0.9, 0.9), fill=True)
            
            # Add title with safe encoding
            title_text = f"Bilingual Document Translation - Page {page_num + 1}"
            insert_text_safely(
                new_page, 
                fitz.Rect(MARGIN + 10, MARGIN + 10, page_rect.width - MARGIN - 10, MARGIN + 35),
                title_text,
                fontname=bold_font,
                fontsize=14,
                color=(0.2, 0.2, 0.2),
                align=fitz.TEXT_ALIGN_CENTER
            )
            
            # Calculate column widths
            column_width = (content_rect.width - COLUMN_GAP) / 2
            
            # Left column (Original)
            left_col_rect = fitz.Rect(
                content_rect.x0,
                content_rect.y0,
                content_rect.x0 + column_width,
                content_rect.y1
            )
            
            # Right column (Translation)
            right_col_rect = fitz.Rect(
                content_rect.x0 + column_width + COLUMN_GAP,
                content_rect.y0,
                content_rect.x1,
                content_rect.y1
            )
            
            # Draw column separators
            new_page.draw_line(
                fitz.Point(content_rect.x0 + column_width + COLUMN_GAP/2, content_rect.y0),
                fitz.Point(content_rect.x0 + column_width + COLUMN_GAP/2, content_rect.y1),
                color=(0.8, 0.8, 0.8),
                width=1
            )
            
            # Add column headers with safe encoding
            header_y = content_rect.y0 + 5
            insert_text_safely(
                new_page,
                fitz.Rect(left_col_rect.x0, header_y, left_col_rect.x1, header_y + 20),
                original_label,
                fontname=bold_font,
                fontsize=12,
                color=(0.3, 0.3, 0.3),
                align=fitz.TEXT_ALIGN_CENTER
            )
            
            insert_text_safely(
                new_page,
                fitz.Rect(right_col_rect.x0, header_y, right_col_rect.x1, header_y + 20),
                translated_label,
                fontname=bold_font,
                fontsize=12,
                color=(0.1, 0.5, 0.1),
                align=fitz.TEXT_ALIGN_CENTER
            )
            
            # Content areas (with padding)
            content_padding = 10
            left_content_rect = fitz.Rect(
                left_col_rect.x0 + content_padding,
                left_col_rect.y0 + 30,
                left_col_rect.x1 - content_padding,
                left_col_rect.y1 - content_padding
            )
            
            right_content_rect = fitz.Rect(
                right_col_rect.x0 + content_padding,
                right_col_rect.y0 + 30,
                right_col_rect.x1 - content_padding,
                right_col_rect.y1 - content_padding
            )
            
            # Insert original text with safe encoding
            if original_text and not should_skip_page(original_text):
                # Clean and format original text
                clean_original = safe_text_encode(original_text.strip()[:1800])  # Limit text length
                
                # Split text into manageable chunks for better rendering
                words = clean_original.split()
                lines = []
                current_line = ""
                max_chars_per_line = 50
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if len(test_line) <= max_chars_per_line:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                
                # Insert lines with proper spacing
                y_pos = left_content_rect.y0 + 15
                line_height = 12
                max_lines = min(len(lines), 30)  # Limit lines to prevent overflow
                
                for i in range(max_lines):
                    if y_pos + line_height > left_content_rect.y1:
                        break
                    insert_text_safely(
                        new_page,
                        (left_content_rect.x0, y_pos),
                        lines[i],
                        fontname=regular_font,
                        fontsize=9,
                        color=(0.1, 0.1, 0.1)
                    )
                    y_pos += line_height
                    
                # Add truncation notice if needed
                if len(lines) > max_lines:
                    insert_text_safely(
                        new_page,
                        (left_content_rect.x0, y_pos),
                        "[... text truncated for display ...]",
                        fontname=italic_font,
                        fontsize=8,
                        color=(0.5, 0.5, 0.5)
                    )
            else:
                insert_text_safely(
                    new_page,
                    (left_content_rect.x0 + 50, left_content_rect.y0 + 100),
                    "[No meaningful content to display]",
                    fontname=italic_font,
                    fontsize=9,
                    color=(0.6, 0.6, 0.6)
                )
            
            # Insert translated text with safe encoding and Unicode handling
            if translated_text and not translated_text.startswith("[Skipped:"):
                # Clean and format translated text with special attention to Unicode
                clean_translated = safe_text_encode(clean_translation(translated_text).strip()[:1800])
                
                # For non-Latin scripts, provide additional handling
                if target_language.lower() in ['hindi', 'arabic', 'chinese', 'japanese', 'korean', 'russian']:
                    # For complex scripts, we'll use a different approach
                    # Split by sentences rather than words for better rendering
                    sentences = clean_translated.replace('।', '।\n').replace('。', '。\n').replace('.', '.\n').split('\n')
                    lines = []
                    
                    for sentence in sentences[:20]:  # Limit sentences
                        sentence = sentence.strip()
                        if sentence:
                            # Break long sentences into chunks
                            if len(sentence) > 60:
                                words = sentence.split()
                                current_line = ""
                                for word in words:
                                    if len(current_line + " " + word) <= 60:
                                        current_line += " " + word if current_line else word
                                    else:
                                        if current_line:
                                            lines.append(current_line)
                                        current_line = word
                                if current_line:
                                    lines.append(current_line)
                            else:
                                lines.append(sentence)
                else:
                    # For Latin scripts, use word-based wrapping
                    words = clean_translated.split()
                    lines = []
                    current_line = ""
                    max_chars_per_line = 50
                    
                    for word in words:
                        test_line = current_line + " " + word if current_line else word
                        if len(test_line) <= max_chars_per_line:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                
                # Insert lines with proper spacing
                y_pos = right_content_rect.y0 + 15
                line_height = 12
                max_lines = min(len(lines), 30)  # Limit lines to prevent overflow
                
                for i in range(max_lines):
                    if y_pos + line_height > right_content_rect.y1:
                        break
                    insert_text_safely(
                        new_page,
                        (right_content_rect.x0, y_pos),
                        lines[i],
                        fontname=regular_font,
                        fontsize=9,
                        color=(0.1, 0.4, 0.1)
                    )
                    y_pos += line_height
                    
                # Add truncation notice if needed
                if len(lines) > max_lines:
                    insert_text_safely(
                        new_page,
                        (right_content_rect.x0, y_pos),
                        "[... translation truncated for display ...]",
                        fontname=italic_font,
                        fontsize=8,
                        color=(0.5, 0.5, 0.5)
                    )
                    
            else:
                skip_message = translated_text if translated_text.startswith("[Skipped:") else "[Translation not available]"
                insert_text_safely(
                    new_page,
                    (right_content_rect.x0 + 50, right_content_rect.y0 + 100),
                    skip_message,
                    fontname=italic_font,
                    fontsize=9,
                    color=(0.6, 0.6, 0.6)
                )
            
            # Add footer with safe encoding
            footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | Page {page_num + 1} of {len(original_pages)}"
            insert_text_safely(
                new_page,
                fitz.Rect(MARGIN, page_rect.height - MARGIN - 20, page_rect.width - MARGIN, page_rect.height - MARGIN),
                footer_text,
                fontname=regular_font,
                fontsize=8,
                color=(0.5, 0.5, 0.5),
                align=fitz.TEXT_ALIGN_CENTER
            )
        
        # Save to bytes
        pdf_bytes = io.BytesIO()
        new_doc.save(pdf_bytes)
        new_doc.close()
        doc.close()
        pdf_bytes.seek(0)
        
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    global translated_content_cache

    if request.method == 'POST':
        try:
            file = request.files.get('file')
            lang = request.form.get('language')
            
            if not file or not lang:
                return render_template("index.html", error="Please upload a PDF file and select a target language.")

            if not file.filename.lower().endswith('.pdf'):
                return render_template("index.html", error="Please upload a valid PDF file.")

            logger.info(f"Processing file: {file.filename}, Target language: {lang}")
            
            # Read PDF
            pdf_bytes = file.read()
            if len(pdf_bytes) == 0:
                return render_template("index.html", error="The uploaded file is empty.")
            
            # Extract text and metadata
            pages, metadata = extract_text_with_metadata(pdf_bytes)
            
            if not pages:
                return render_template("index.html", error="No text content found in the PDF.")
            
            # Initialize LLM
            llm = ChatGroq(model="gemma2-9b-it", temperature=0, api_key=GROQ_API_KEY)
            translated_pages = []
            
            logger.info(f"Translating {len(pages)} pages to {lang}")
            
            # Translate pages
            for i, text in enumerate(pages):
                logger.info(f"Processing page {i+1}/{len(pages)}")
                
                if should_skip_page(text):
                    translated_pages.append("[Skipped: Page contains insufficient meaningful content for translation]")
                    continue

                try:
                    prompt = build_prompt(text, lang)
                    result = llm.predict(prompt)
                    cleaned_result = clean_translation(result.strip())
                    translated_pages.append(cleaned_result)
                except Exception as e:
                    logger.error(f"Translation error for page {i+1}: {str(e)}")
                    translated_pages.append(f"[Translation Error: {str(e)}]")

            # Cache results
            translated_content_cache = {
                "text": '\n\n'.join(translated_pages),
                "pages": translated_pages,
                "original_pages": pages,
                "metadata": metadata,
                "target_language": lang,
                "pdf_bytes": pdf_bytes
            }
            
            logger.info("Translation completed successfully")
            
            return render_template("index.html",
                                   translated_text=translated_content_cache["text"],
                                   page_count=len(translated_pages),
                                   target_language=lang.title(),
                                   success_message=f"Successfully translated {len(pages)} pages to {lang.title()}!")

        except Exception as e:
            logger.error(f"Translation process failed: {str(e)}")
            return render_template("index.html", error=f"Translation failed: {str(e)}")

    return render_template("index.html")

@app.route('/download_pdf')
def download_pdf():
    global translated_content_cache
    
    try:
        if not translated_content_cache.get("pages"):
            return jsonify({"error": "No translated content available. Please translate a document first."}), 400
        
        logger.info("Generating PDF download")
        
        # Generate optimized PDF
        pdf_bytes = create_professional_pdf(
            translated_content_cache["pdf_bytes"],
            translated_content_cache["pages"],
            translated_content_cache["original_pages"],
            translated_content_cache["target_language"]
        )
        
        # Save to file
        output_filename = f"bilingual_translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = os.path.join(os.getcwd(), output_filename)
        
        with open(output_path, "wb") as f:
            f.write(pdf_bytes.read())
        
        logger.info(f"PDF saved as {output_filename}")
        
        return send_file(
            output_path, 
            as_attachment=True, 
            download_name=output_filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"PDF download failed: {str(e)}")
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500

@app.route('/api/translation_status')
def translation_status():
    """API endpoint to check if translation is available"""
    global translated_content_cache
    
    has_translation = bool(translated_content_cache.get("pages"))
    
    return jsonify({
        "has_translation": has_translation,
        "page_count": len(translated_content_cache.get("pages", [])),
        "target_language": translated_content_cache.get("target_language", ""),
        "metadata": translated_content_cache.get("metadata", {})
    })

@app.errorhandler(404)
def not_found_error(error):
    return render_template("index.html", error="Page not found."), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("index.html", error="An internal server error occurred."), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)