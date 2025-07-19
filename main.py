from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.cloud import translate_v2 as translate
import os
from typing import List, Tuple, Dict
import logging
import asyncio
import uuid
from contextlib import asynccontextmanager

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, environment variables should be set manually
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Manga OCR Translator", description="Translate Japanese manga text to English")

# Translation client will be initialized when needed
translate_client = None

# Track active processing requests
active_requests: Dict[str, asyncio.Task] = {}

class MangaProcessor:
    def __init__(self):
        try:
            # Import manga_ocr here to handle potential import issues
            import manga_ocr
            self.ocr = manga_ocr.MangaOcr()
            logger.info("MangaOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MangaOCR: {e}")
            self.ocr = None
    
    def extract_text_regions(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Extract text and bounding boxes from manga image"""
        if not self.ocr:
            raise Exception("OCR not initialized")
        
        # Convert numpy array to PIL Image for manga-ocr
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # manga-ocr returns full text, we need to implement text detection
        # For now, we'll use a simple approach with contour detection
        text_regions = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter small areas and very thin/wide rectangles
            if area > 100 and 0.1 < h/w < 10:
                # Extract region and run OCR
                roi = pil_image.crop((x, y, x+w, y+h))
                try:
                    text = self.ocr(roi)
                    if text and text.strip():
                        text_regions.append((text.strip(), (x, y, w, h)))
                except Exception as e:
                    logger.warning(f"OCR failed for region {x},{y},{w},{h}: {e}")
                    continue
        
        return text_regions
    
    def translate_text(self, text: str) -> str:
        """Translate Japanese text to English"""
        global translate_client
        try:
            # Initialize translation client if not already done
            if translate_client is None:
                credentials_path = "./credentials.json"  # Force local file
                if not os.path.exists(credentials_path):
                    raise FileNotFoundError(
                        f"Google credentials not found at {credentials_path}. "
                        "Please ensure 'credentials.json' is in the project folder."
                    )
                translate_client = translate.Client.from_service_account_json(credentials_path)

            
            result = translate_client.translate(text, target_language='en', source_language='ja')
            return result['translatedText']
        except Exception as e:
            logger.error(f"Translation failed for text '{text}': {e}")
            return text  # Return original text if translation fails
    
    def remove_text_from_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Remove text from specified region using inpainting"""
        x, y, w, h = bbox
        
        # Create mask for the text region
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        
        # Use inpainting to remove text
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return result
    
    def add_translated_text(self, image: np.ndarray, text: str, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Add translated text to the image at specified location"""
        x, y, w, h = bbox
        
        # Convert to PIL for text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, fallback to default if not available
        try:
            font_size = max(12, min(h-4, w//len(text)+2))
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position to center it in the bounding box
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        
        text_x = x + (w - text_w) // 2
        text_y = y + (h - text_h) // 2
        
        # Add white background rectangle
        draw.rectangle([text_x-2, text_y-2, text_x+text_w+2, text_y+text_h+2], fill='white', outline='black')
        
        # Add text
        draw.text((text_x, text_y), text, fill='black', font=font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    async def process_manga_image(self, image: np.ndarray, request_id: str = None) -> np.ndarray:
        """Complete pipeline to process manga image with cancellation support"""
        logger.info(f"Starting manga image processing (request_id: {request_id})")
        
        # Check for cancellation
        if request_id and request_id not in active_requests:
            raise asyncio.CancelledError("Request was cancelled")
        
        # Extract text regions
        text_regions = self.extract_text_regions(image)
        logger.info(f"Found {len(text_regions)} text regions")
        
        result_image = image.copy()
        
        # Process each text region
        for i, (japanese_text, bbox) in enumerate(text_regions):
            # Check for cancellation before each region
            if request_id and request_id not in active_requests:
                raise asyncio.CancelledError("Request was cancelled")
                
            logger.info(f"Processing region {i+1}/{len(text_regions)}: '{japanese_text}'")
            
            # Translate text (run in thread pool to avoid blocking)
            english_text = await asyncio.get_event_loop().run_in_executor(
                None, self.translate_text, japanese_text
            )
            logger.info(f"Translated to: '{english_text}'")
            
            # Check for cancellation after translation
            if request_id and request_id not in active_requests:
                raise asyncio.CancelledError("Request was cancelled")
            
            # Remove original text
            result_image = self.remove_text_from_region(result_image, bbox)
            
            # Add translated text
            result_image = self.add_translated_text(result_image, english_text, bbox)
        
        logger.info(f"Completed manga image processing (request_id: {request_id})")
        return result_image

# Initialize processor
processor = MangaProcessor()

@app.post("/translate-manga")
async def translate_manga(file: UploadFile = File(...)):
    """Upload manga image and get translated version"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    try:
        # Read image data
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Create and track the processing task
        task = asyncio.create_task(processor.process_manga_image(image, request_id))
        active_requests[request_id] = task
        
        try:
            # Process the image
            result_image = await task
            
            # Encode result as PNG
            _, buffer = cv2.imencode('.png', result_image)
            
            # Return as streaming response with request ID header
            return StreamingResponse(
                io.BytesIO(buffer.tobytes()),
                media_type="image/png",
                headers={
                    "Content-Disposition": "attachment; filename=translated_manga.png",
                    "X-Request-Id": request_id
                }
            )
        except asyncio.CancelledError:
            logger.info(f"Request {request_id} was cancelled")
            raise HTTPException(status_code=499, detail="Request was cancelled")
        finally:
            # Clean up the request tracking
            active_requests.pop(request_id, None)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        active_requests.pop(request_id, None)
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        active_requests.pop(request_id, None)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/cancel/{request_id}")
async def cancel_request(request_id: str):
    """Cancel an ongoing translation request"""
    if request_id not in active_requests:
        raise HTTPException(status_code=404, detail="Request not found or already completed")
    
    task = active_requests[request_id]
    task.cancel()
    active_requests.pop(request_id, None)
    
    logger.info(f"Cancelled request {request_id}")
    return {"message": f"Request {request_id} cancelled successfully"}

@app.get("/status")
async def get_status():
    """Get status of all active requests"""
    return {
        "active_requests": len(active_requests),
        "request_ids": list(active_requests.keys())
    }

@app.get("/")
async def root():
    return {
        "message": "Manga OCR Translator API", 
        "endpoints": [
            "/translate-manga", 
            "/cancel/{request_id}", 
            "/status", 
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    # Check if translation credentials are available
    translation_available = False
    try:
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', './credentials.json')
        test_client = translate.Client.from_service_account_json(credentials_path)
        translation_available = True
    except Exception:
        translation_available = False
    
    return {
        "status": "healthy", 
        "ocr_available": processor.ocr is not None,
        "translation_available": translation_available
    }

if __name__ == "__main__":
    # Print credentials information
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', './credentials.json')
    print(f"Looking for Google Cloud credentials at: {credentials_path}")
    print(f"Credentials file exists: {os.path.exists(credentials_path)}")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)