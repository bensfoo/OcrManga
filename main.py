from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    def extract_text_regions(self, image: np.ndarray, request_id: str = None) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Extract text and bounding boxes using manga-ocr with sliding window"""
        logger.info("Starting text region extraction with manga-ocr sliding window")
        
        if not self.ocr:
            raise Exception("OCR not initialized")
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        logger.info(f"Image size: {pil_image.size}")
        
        text_regions = []
        h, w = image.shape[:2]
        
        # Use sliding window with multiple sizes optimized for manga text bubbles
        window_sizes = [
            (120, 60),   # Small text bubbles
            (180, 90),   # Medium text bubbles  
            (240, 120),  # Large text bubbles
            (300, 150),  # Very large text bubbles
            (160, 40),   # Wide horizontal text
            (80, 120),   # Tall vertical text
        ]
        
        step_size = 60  # Larger step size for faster processing
        logger.info(f"Using {len(window_sizes)} window sizes with step size {step_size}")
        
        # Calculate total iterations for progress tracking
        total_windows = 0
        for window_w, window_h in window_sizes:
            y_steps = max(1, (h - window_h) // step_size + 1)
            x_steps = max(1, (w - window_w) // step_size + 1)
            total_windows += y_steps * x_steps
        
        logger.info(f"Total windows to process: {total_windows}")
        processed_windows = 0
        
        for window_idx, (window_w, window_h) in enumerate(window_sizes):
            logger.info(f"Processing window size {window_idx+1}/{len(window_sizes)}: {window_w}x{window_h}")
            
            y_steps = max(1, (h - window_h) // step_size + 1)
            x_steps = max(1, (w - window_w) // step_size + 1)
            window_size_total = y_steps * x_steps
            window_size_processed = 0
            
            for y in range(0, h - window_h + 1, step_size):
                for x in range(0, w - window_w + 1, step_size):
                    processed_windows += 1
                    window_size_processed += 1
                    
                    # Log progress every 50 windows
                    if processed_windows % 50 == 0:
                        progress = (processed_windows / total_windows) * 100
                        logger.info(f"Progress: {processed_windows}/{total_windows} ({progress:.1f}%) - Found {len(text_regions)} text regions so far")
                    
                    # Check for cancellation periodically
                    if processed_windows % 25 == 0:
                        if request_id and request_id not in active_requests:
                            logger.info(f"Text extraction cancelled at {processed_windows}/{total_windows}")
                            raise asyncio.CancelledError("Request was cancelled")
                    # Extract region
                    roi = pil_image.crop((x, y, x + window_w, y + window_h))
                    
                    try:
                        text = self.ocr(roi)
                        if text and text.strip() and len(text.strip()) > 1:
                            # Check if this region overlaps significantly with existing regions
                            overlaps = False
                            for existing_text, (ex, ey, ew, eh) in text_regions:
                                # Calculate overlap
                                overlap_x = max(0, min(x + window_w, ex + ew) - max(x, ex))
                                overlap_y = max(0, min(y + window_h, ey + eh) - max(y, ey))
                                overlap_area = overlap_x * overlap_y
                                current_area = window_w * window_h
                                existing_area = ew * eh
                                
                                # Consider overlap if more than 30% of either region overlaps
                                if (overlap_area > current_area * 0.3 or 
                                    overlap_area > existing_area * 0.3):
                                    overlaps = True
                                    # Keep the one with more text or larger area
                                    if len(text.strip()) > len(existing_text.strip()) or current_area > existing_area:
                                        logger.info(f"Replacing overlapping region: '{existing_text}' -> '{text.strip()}'")
                                        text_regions.remove((existing_text, (ex, ey, ew, eh)))
                                        overlaps = False  # Add the new one
                                    break
                            
                            if not overlaps:
                                text_regions.append((text.strip(), (x, y, window_w, window_h)))
                                logger.info(f"Found text: '{text.strip()}' at ({x},{y},{window_w},{window_h})")
                    except Exception as e:
                        # OCR failed, continue to next region
                        continue
        
        # Sort regions by area (largest first) to prioritize main text areas
        text_regions.sort(key=lambda x: x[1][2] * x[1][3], reverse=True)
        
        # Limit to top 12 regions to avoid too many translations
        text_regions = text_regions[:12]
        logger.info(f"Final text regions count: {len(text_regions)}")
        
        return text_regions
    
    async def translate_text(self, text: str, request_id: str = None) -> str:
        """Translate Japanese text to English with cancellation support"""
        global translate_client
        logger.info(f"Starting translation for: '{text}'")
        
        try:
            # Check for cancellation before starting translation
            if request_id and request_id not in active_requests:
                logger.info(f"Translation cancelled before starting for: '{text}'")
                raise asyncio.CancelledError("Request was cancelled")
                
            # Initialize translation client if not already done
            if translate_client is None:
                logger.info("Initializing Google Translate client...")
                credentials_path = "./credentials.json"  # Force local file
                if not os.path.exists(credentials_path):
                    raise FileNotFoundError(
                        f"Google credentials not found at {credentials_path}. "
                        "Please ensure 'credentials.json' is in the project folder."
                    )
                translate_client = translate.Client.from_service_account_json(credentials_path)
                logger.info("Google Translate client initialized successfully")

            # Run translation in executor but with timeout to allow cancellation checking
            def _translate():
                logger.debug(f"Calling Google Translate API for: '{text}'")
                result = translate_client.translate(text, target_language='en', source_language='ja')
                logger.debug(f"Google Translate API returned: {result}")
                return result
            
            # Use asyncio.wait_for with a short timeout to allow periodic cancellation checks
            try:
                logger.info(f"Sending to Google Translate (5s timeout): '{text}'")
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, _translate), 
                    timeout=5.0
                )
                translated_text = result['translatedText']
                logger.info(f"Translation successful: '{text}' -> '{translated_text}'")
                return translated_text
            except asyncio.TimeoutError:
                logger.warning(f"Translation timeout for: '{text}', checking cancellation...")
                # If translation takes too long, check for cancellation and retry
                if request_id and request_id not in active_requests:
                    logger.info(f"Translation cancelled during timeout for: '{text}'")
                    raise asyncio.CancelledError("Request was cancelled")
                # If not cancelled, run without timeout
                logger.info(f"Retrying translation without timeout for: '{text}'")
                result = await asyncio.get_event_loop().run_in_executor(None, _translate)
                translated_text = result['translatedText']
                logger.info(f"Translation successful (retry): '{text}' -> '{translated_text}'")
                return translated_text
                
        except asyncio.CancelledError:
            logger.info(f"Translation cancelled for: '{text}'")
            raise
        except Exception as e:
            logger.error(f"Translation failed for text '{text}': {e}")
            return text  # Return original text if translation fails
    
    def remove_text_from_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Remove text from specified region by filling with white"""
        x, y, w, h = bbox
        result = image.copy()
        
        # Fill the text region with white background
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255), -1)
        
        return result
    
    def add_translated_text(self, image: np.ndarray, text: str, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Add translated text to the image at specified location"""
        x, y, w, h = bbox
        
        # Convert to PIL for text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Calculate appropriate font size based on bounding box
        font_size = max(8, min(h-4, w//max(1, len(text)//2)))
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        
        # If text is too wide, wrap it or use smaller font
        if text_w > w - 4:
            font_size = max(6, int(font_size * (w - 4) / text_w))
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            bbox_text = draw.textbbox((0, 0), text, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        
        # Center text in the bounding box
        text_x = x + (w - text_w) // 2
        text_y = y + (h - text_h) // 2
        
        # Ensure text stays within bounds
        text_x = max(x, min(text_x, x + w - text_w))
        text_y = max(y, min(text_y, y + h - text_h))
        
        # Add text directly on white background (already cleared)
        draw.text((text_x, text_y), text, fill='black', font=font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def add_red_text_overlay(self, image: np.ndarray, text: str, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Add red translated text overlay on the original image"""
        x, y, w, h = bbox
        logger.info(f"Creating red overlay - Text: '{text}', Box: ({x},{y},{w},{h})")
        
        # Convert to PIL for text rendering
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        logger.info(f"Converted to PIL image size: {pil_image.size}")
        
        # Calculate appropriate font size based on bounding box - much larger
        font_size = max(160, min(h*2, w//max(1, len(text)//4)))  # 20x bigger base size
        logger.info(f"Calculated font size: {font_size}")
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            logger.info("Successfully loaded Arial font")
        except Exception as e:
            logger.warning(f"Failed to load Arial font: {e}, using default")
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        logger.info(f"Text dimensions: {text_w}x{text_h}")
        
        # If text is too wide, use smaller font
        if text_w > w - 4:
            old_font_size = font_size
            font_size = max(120, int(font_size * (w - 4) / text_w))  # Keep minimum 120 instead of 6
            logger.info(f"Text too wide, reducing font size from {old_font_size} to {font_size}")
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            bbox_text = draw.textbbox((0, 0), text, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
            logger.info(f"New text dimensions: {text_w}x{text_h}")
        
        # Center text in the bounding box
        text_x = x + (w - text_w) // 2
        text_y = y + (h - text_h) // 2
        logger.info(f"Centered text position: ({text_x},{text_y})")
        
        # Ensure text stays within bounds
        text_x = max(x, min(text_x, x + w - text_w))
        text_y = max(y, min(text_y, y + h - text_h))
        logger.info(f"Bounded text position: ({text_x},{text_y})")
        
        # Draw a semi-transparent red rectangle background
        logger.info(f"Drawing red background rectangle at ({x},{y}) to ({x+w},{y+h})")
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x, y, x+w, y+h], fill=(255, 0, 0, 100))  # Semi-transparent red
        
        # Composite the overlay
        pil_image = pil_image.convert('RGBA')
        pil_image = Image.alpha_composite(pil_image, overlay)
        pil_image = pil_image.convert('RGB')
        draw = ImageDraw.Draw(pil_image)
        logger.info("Applied red background overlay")
        
        # Add white text with black outline for visibility
        logger.info(f"Drawing text '{text}' with black outline at ({text_x},{text_y})")
        # Black outline
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((text_x + dx, text_y + dy), text, fill='black', font=font)
        # White text
        draw.text((text_x, text_y), text, fill='white', font=font)
        logger.info("Finished drawing text with outline")
        
        # Convert back to OpenCV format
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        logger.info("Converted back to OpenCV format and returning")
        return result
    
    async def process_manga_image(self, image: np.ndarray, request_id: str = None) -> np.ndarray:
        """Complete pipeline to process manga image with cancellation support"""
        logger.info(f"Starting manga image processing (request_id: {request_id})")
        logger.info(f"Input image shape: {image.shape}")
        
        # Check for cancellation
        if request_id and request_id not in active_requests:
            logger.info(f"Request {request_id} was cancelled before starting")
            raise asyncio.CancelledError("Request was cancelled")
        
        # Extract text regions
        logger.info("=" * 50)
        logger.info("STARTING TEXT REGION EXTRACTION")
        logger.info("=" * 50)
        text_regions = self.extract_text_regions(image, request_id)
        logger.info("=" * 50)
        logger.info(f"TEXT REGION EXTRACTION COMPLETE - Found {len(text_regions)} regions")
        logger.info("=" * 50)
        
        # Start with original image
        result_image = image.copy()
        logger.info(f"Created copy of original image for processing")
        
        # Process each text region and add red translated text overlay
        for i, (japanese_text, bbox) in enumerate(text_regions):
            # Check for cancellation before each region
            if request_id and request_id not in active_requests:
                logger.info(f"Request {request_id} was cancelled before region {i+1}")
                raise asyncio.CancelledError("Request was cancelled")
                
            logger.info("-" * 30)
            logger.info(f"PROCESSING REGION {i+1}/{len(text_regions)}")
            logger.info(f"Japanese text: '{japanese_text}'")
            logger.info(f"Bounding box: {bbox}")
            logger.info("-" * 30)
            
            # Translate text with cancellation support
            english_text = await self.translate_text(japanese_text, request_id)
            logger.info(f"Final translated text: '{english_text}'")
            
            # Check for cancellation after translation
            if request_id and request_id not in active_requests:
                logger.info(f"Request {request_id} was cancelled after translating region {i+1}")
                raise asyncio.CancelledError("Request was cancelled")
            
            # Add red translated text overlay (don't remove original)
            logger.info(f"Adding red overlay for: '{english_text}' at {bbox}")
            result_image = self.add_red_text_overlay(result_image, english_text, bbox)
            logger.info(f"Successfully added overlay for region {i+1}")
        
        logger.info("=" * 50)
        logger.info(f"MANGA IMAGE PROCESSING COMPLETE (request_id: {request_id})")
        logger.info(f"Processed {len(text_regions)} regions successfully")
        logger.info("=" * 50)
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
            
            # Save the translated image to output folder
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"translated_{file.filename or 'manga'}.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, result_image)
            logger.info(f"Saved translated image to: {output_path}")
            
            # Encode result as PNG
            _, buffer = cv2.imencode('.png', result_image)
            
            # Return as streaming response with request ID header
            return StreamingResponse(
                io.BytesIO(buffer.tobytes()),
                media_type="image/png",
                headers={
                    "Content-Disposition": f"attachment; filename={output_filename}",
                    "X-Request-Id": request_id,
                    "X-Output-Path": output_path
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