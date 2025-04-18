from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pdfplumber
import os
import uuid
import requests
from dotenv import load_dotenv
import shutil
from pathlib import Path
import json
from typing import Optional
import google.generativeai as genai

# Load environment variables
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create absolute paths for file storage
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "audio_files"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# Mount static files directory
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Pydantic models for request validation
class TTSRequest(BaseModel):
    file_id: Optional[str] = None
    page_number: Optional[int] = None
    text: Optional[str] = None

class SummarizeRequest(BaseModel):
    file_id: Optional[str] = None
    page_number: Optional[int] = None
    text: Optional[str] = None

# In-memory storage
pdfs = {}  # Store PDF file paths by file ID
texts = {}  # Store extracted text by file ID and page number
audios = {}  # Store audio file paths by audio ID

def extract_text_from_pdf(file_path, page_number=None):
    try:
        with pdfplumber.open(file_path) as pdf:
            if page_number is not None:
                if page_number < 0 or page_number >= len(pdf.pages):
                    raise HTTPException(status_code=400, detail="Invalid page number")
                return pdf.pages[page_number].extract_text() or ""
            else:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")

def text_to_speech(text):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")
    
    url = "https://api.elevenlabs.io/v1/text-to-speech/UEKYgullGqaF0keqT8Bu"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    data = {"text": text, "model_id": "eleven_monolingual_v1", "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.content
        else:
            raise HTTPException(status_code=response.status_code, detail=f"ElevenLabs API error: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling ElevenLabs API: {str(e)}")

async def generate_summary_with_gemini(text):
    """Generate a summary using Google's Gemini AI model."""
    if not GEMINI_API_KEY:
        # Fallback if no Gemini API key
        return f"Summary: {text[:200]}..." if len(text) > 200 else text
    
    try:
        # Configure the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Create the prompt for summarization
        prompt = f"""
        Please provide a concise summary of the following text. Extract the key points, 
        main ideas, and important details. Keep the summary brief and focused on the most 
        relevant information:

        {text}
        """
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Return the generated summary
        return response.text.strip()
    except Exception as e:
        # Fallback in case of any errors
        print(f"Error using Gemini for summarization: {str(e)}")
        return f"Summary: {text[:200]}..." if len(text) > 200 else text

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.pdf"
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Store the file path
        pdfs[file_id] = str(file_path)
        
        # Extract text from all pages
        with pdfplumber.open(file_path) as pdf:
            texts[file_id] = {
                "total_pages": len(pdf.pages),
                "pages": {}
            }
            for i, page in enumerate(pdf.pages):
                texts[file_id]["pages"][i] = page.extract_text() or ""
        
        return {
            "file_id": file_id,
            "message": "File uploaded successfully",
            "total_pages": texts[file_id]["total_pages"]
        }
    except Exception as e:
        # Clean up the file if something went wrong
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdf/{file_id}")
def get_pdf(file_id: str):
    file_path = pdfs.get(file_id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="application/pdf")

@app.get("/text/{file_id}/{page_number}")
def get_text(file_id: str, page_number: int):
    if file_id not in texts:
        raise HTTPException(status_code=404, detail="File not found")
    
    if page_number < 0 or page_number >= texts[file_id]["total_pages"]:
        raise HTTPException(status_code=400, detail="Invalid page number")
    
    return {"text": texts[file_id]["pages"][page_number]}

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    # Case 1: Reading a page from a PDF
    if request.file_id and request.page_number is not None:
        if request.file_id not in texts:
            raise HTTPException(status_code=404, detail="File not found")
        
        if request.page_number < 0 or request.page_number >= texts[request.file_id]["total_pages"]:
            raise HTTPException(status_code=400, detail="Invalid page number")
        
        text = texts[request.file_id]["pages"][request.page_number]
    
    # Case 2: Direct text input (for reading summaries)
    elif request.text:
        text = request.text
    else:
        raise HTTPException(status_code=400, detail="Either file_id and page_number, or text must be provided")
    
    try:
        audio_data = text_to_speech(text)
        audio_id = str(uuid.uuid4())
        audio_path = AUDIO_DIR / f"{audio_id}.mp3"
        
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        
        audios[audio_id] = str(audio_path)
        return {"audio_url": f"/audio/{audio_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{audio_id}")
def get_audio(audio_id: str):
    audio_path = audios.get(audio_id)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(audio_path, media_type="audio/mpeg")

@app.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    # Case 1: Summarizing a page from a PDF
    if request.file_id and request.page_number is not None:
        if request.file_id not in texts:
            raise HTTPException(status_code=404, detail="File not found")
        
        if request.page_number < 0 or request.page_number >= texts[request.file_id]["total_pages"]:
            raise HTTPException(status_code=400, detail="Invalid page number")
        
        text = texts[request.file_id]["pages"][request.page_number]
    
    # Case 2: Direct text input
    elif request.text:
        text = request.text
    else:
        raise HTTPException(status_code=400, detail="Either file_id and page_number, or text must be provided")
    
    # Generate summary using Gemini AI
    summary = await generate_summary_with_gemini(text)
    return {"summary": summary}

# Cleanup function to remove old files
@app.on_event("shutdown")
async def cleanup():
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    shutil.rmtree(AUDIO_DIR, ignore_errors=True)