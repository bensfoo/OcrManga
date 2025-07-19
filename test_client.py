import requests
import sys
from pathlib import Path

def test_manga_translation(image_path: str, output_path: str = "translated_output.png"):
    """Test the manga translation API"""
    
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found")
        return
    
    # API endpoint
    url = "http://localhost:8000/translate-manga"
    
    try:
        # Open and send image file
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            # Save translated image
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Success! Translated image saved as '{output_path}'")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print(f"API Status: {data['status']}")
            print(f"OCR Available: {data['ocr_available']}")
        else:
            print(f"API Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("API is not running. Start it with: python main.py")
    except Exception as e:
        print(f"Error checking API: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <image_path> [output_path]")
        print("Example: python test_client.py manga_page.jpg translated.png")
        print()
        print("Checking API health...")
        check_api_health()
    else:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "translated_output.png"
        test_manga_translation(image_path, output_path)