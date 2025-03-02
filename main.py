from fastapi import FastAPI, Depends, HTTPException, status, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import logging
import openai
import cv2
import numpy as np
import io
import torch
import librosa
from pydub import AudioSegment
from datetime import datetime, timedelta
from jose import JWTError
from deepface import DeepFace
# For gaze tracking, we use a placeholder. Replace with your DeepGaze integration.
# from deepgaze.pytorch_implementation import GazeTracking
# For WebRTC, we'll include a placeholder endpoint.
import requests
import whisper

# Local imports
from database import SessionLocal, init_db
from models import User, Interview
from auth import hash_password, verify_password, create_access_token, decode_access_token
from config import OPENAI_API_KEY, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# 🔹 Define Judge0 API URL (Change this if using a private instance)
JUDGE0_API_URL = "https://ce.judge0.com/submissions/?base64_encoded=false&wait=true"

# 🔹 Define headers (Add API key if required)
JUDGE0_HEADERS = {
    "Content-Type": "application/json"
    # "X-Auth-Token": "your_api_key"  # Uncomment if your instance requires authentication
}

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Interview Assistant with Full Multimodal Features")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize the database
init_db()

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

# Dependency: get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Models ---
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    email: EmailStr
    username: str

class InterviewRequest(BaseModel):
    question: str
    answer: str

class CodeSubmission(BaseModel):
    language_id: int
    source_code: str
    input_data: str

# --- Authentication Dependency ---
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_access_token(token)
    user_email = payload.get("sub") if payload else None
    if not user_email:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# --- Serve Frontend Pages ---
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup")
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/start-interview")
async def start_interview_page(request: Request):
    return templates.TemplateResponse("start-interview.html", {"request": request})

@app.get("/about")
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact")
async def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.get("/history")
async def history_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("history.html", {"request": request, "user": current_user})

@app.get("/achievements")
async def achievements_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("achievements.html", {"request": request, "user": current_user})

@app.get("/improvements")
async def improvements_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("improvements.html", {"request": request, "user": current_user})

@app.get("/profile")
async def profile_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("profile.html", {"request": request, "user": current_user})

# --- User Authentication Endpoints ---
@app.post("/api/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter((User.email == user.email) | (User.username == user.username)).first():
        raise HTTPException(status_code=400, detail="Email or username already taken")
    hashed_pwd = hash_password(user.password)
    new_user = User(email=user.email, username=user.username, hashed_password=hashed_pwd)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully"}

@app.post("/api/login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    user_record = db.query(User).filter(User.email == user.email).first()
    if not user_record or not verify_password(user.password, user_record.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/profile")
def get_user_profile(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username, "email": current_user.email, "created_at": current_user.created_at}

@app.put("/api/profile")
def update_user_profile(update_data: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if db.query(User).filter(User.email == update_data.email, User.id != current_user.id).first():
        raise HTTPException(status_code=400, detail="Email already taken")
    if db.query(User).filter(User.username == update_data.username, User.id != current_user.id).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    current_user.username = update_data.username
    current_user.email = update_data.email
    db.commit()
    db.refresh(current_user)
    return {"message": "Profile updated successfully"}

# --- AI Interview Feedback ---
@app.post("/api/interview")
def interview_simulation(interview: InterviewRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI that provides interview feedback."},
                {"role": "user", "content": f"Provide feedback for the following answer: {interview.answer}"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        feedback = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail="AI service unavailable")
    new_interview = Interview(user_id=current_user.id, question=interview.question, answer=interview.answer, feedback=feedback)
    db.add(new_interview)
    db.commit()
    return {"question": interview.question, "answer": interview.answer, "feedback": feedback}

# --- Facial Analysis using DeepFace ---
@app.post("/api/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        analysis = DeepFace.analyze(image, actions=["emotion"], enforce_detection=False)
        return {"emotion": analysis[0]["dominant_emotion"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Gaze Tracking (DeepGaze Placeholder) ---
@app.post("/api/analyze-gaze")
async def analyze_gaze(file: UploadFile = File(...)):
    try:
        # Read image and simulate gaze tracking; replace this with DeepGaze integration.
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # Placeholder: in a real implementation, you would use DeepGaze or Dlib-based landmark detection.
        gaze_direction = "Center"  # Simulated result
        return {"gaze_direction": gaze_direction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Speech Analysis using Whisper and Librosa ---
@app.post("/api/analyze-voice")
async def analyze_voice(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # 🔹 Convert M4A to WAV using pydub
        audio = AudioSegment.from_file(io.BytesIO(contents), format="m4a")
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")  
        wav_buffer.seek(0)

        # 🔹 Load Whisper model for transcription
        model = whisper.load_model("base")
        transcript = model.transcribe(wav_buffer)["text"]

        # 🔹 Load audio as NumPy array for analysis
        wav_buffer.seek(0)  # Reset buffer before reading again
        y, sr = librosa.load(wav_buffer, sr=None)  # Librosa loads as np.ndarray

        # 🔹 Extract pitch and energy
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
        energy = np.mean(np.abs(y))

        return {"transcript": transcript, "average_pitch": avg_pitch, "energy": energy}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/submit-code")
async def submit_code(submission: CodeSubmission):
    try:
        payload = {
            "language_id": submission.language_id,
            "source_code": submission.source_code,
            "stdin": submission.input_data,
            "cpu_time_limit": "2"
        }
        response = requests.post(JUDGE0_API_URL, json=payload, headers=JUDGE0_HEADERS)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- WebRTC Video Streaming Offer (Placeholder) ---
@app.post("/api/webrtc-offer")
async def webrtc_offer(offer: dict):
    # This is a simplified placeholder for WebRTC signaling.
    # In a full implementation, you'd integrate aiortc for real-time communication.
    from aiortc import RTCPeerConnection, RTCSessionDescription
    peer_connection = RTCPeerConnection()
    await peer_connection.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)
    return {"sdp": peer_connection.localDescription.sdp, "type": peer_connection.localDescription.type}

# --- End of Endpoints ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
