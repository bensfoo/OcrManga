# Manga OCR Translator

A lightweight OCR application that reads Japanese text from manga images and translates it to English, preserving the original layout and replacing text in-place.

## Features

- **Manga-specific OCR**: Uses specialized OCR designed for manga text recognition
- **Japanese to English Translation**: Leverages Google Cloud Translation API
- **Layout Preservation**: Maintains original image frames and layout
- **Text Replacement**: Intelligently removes Japanese text and replaces with English
- **RESTful API**: Easy to integrate with other applications

## Technology Stack

- **OCR**: Lightweight MangaOCR (designed specifically for manga)
- **Translation**: Google Cloud Translation API
- **Backend**: FastAPI (Python)
- **Image Processing**: OpenCV + PIL
- **Text Detection**: Contour-based detection with OCR validation

## Setup

### Prerequisites

1. **Python 3.13+**
2. **Poetry** - Python dependency management tool
   ```bash
   # Install Poetry (if not already installed)
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. **Google Cloud Translation API credentials**

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd OcrManga
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Set up Google Cloud Translation API:**
   - Create a Google Cloud Project
   - Enable the Translation API
   - Create a service account and download the JSON key
   - Set the environment variable:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
     ```

### Running the Application

1. **Start the API server:**
   ```bash
   poetry run python main.py
   ```
   The API will be available at `http://localhost:8000`

2. **Check API health:**
   ```bash
   curl http://localhost:8000/health
   ```

## Usage

### API Endpoints

- `POST /translate-manga` - Upload manga image for translation
- `GET /health` - Check API status
- `GET /` - API information

### Using the Test Client

```bash
# Test with an image
poetry run python test_client.py your_manga_image.jpg output_translated.png

# Check API status
poetry run python test_client.py
```

### Using curl

```bash
curl -X POST "http://localhost:8000/translate-manga" \
     -H "accept: image/png" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_manga_image.jpg" \
     --output translated_manga.png
```

## How It Works

1. **Image Upload**: Accepts manga image in common formats (JPG, PNG, etc.)
2. **Text Detection**: Uses computer vision to detect text regions
3. **OCR Processing**: Extracts Japanese text using manga-specialized OCR
4. **Translation**: Translates detected text to English via Google Translate
5. **Text Removal**: Uses inpainting to remove original Japanese text
6. **Text Replacement**: Places English translation in the same locations
7. **Output**: Returns processed image with English text

## Limitations

- Requires internet connection for translation API
- Works best with clear, readable manga text
- Complex layouts or artistic text may need manual adjustment
- Currently supports Japanese to English only

## Configuration

The application uses environment variables for configuration:

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud credentials JSON
- `PORT`: API server port (default: 8000)

## Contributing

1. Ensure your changes maintain the lightweight nature of the application
2. Test with various manga images
3. Follow existing code style and structure

## License

This project is for educational and personal use. Please respect copyright when processing manga images.