from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import uuid
import hashlib
from sqlalchemy.orm import Session
from database import get_db, User, init_db

router = APIRouter()

# Initialize Tables
init_db()

# --- Models ---
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

# --- Helpers ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# --- Routes ---

@router.post("/session/register")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    print(f"--> REGISTER ATTEMPT: {req.email}")
    
    email_key = req.email.lower().strip()
    
    # Check if user exists
    existing = db.query(User).filter(User.email == email_key).first()
    if existing:
        print(f"    FAILED: User {email_key} already exists")
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create simple token
    token = str(uuid.uuid4())
    safe_password = hash_password(req.password.strip())
    
    new_user = User(
        email=email_key,
        username=req.username,
        password_hash=safe_password,
        token=token,
        params={}
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    print(f"    SUCCESS: User {email_key} registered with token {token}")
    
    return {
        "status": "success",
        "token": token,
        "user": {
            "username": req.username,
            "email": req.email,
            "id": token
        }
    }

@router.post("/session/create")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    print(f"--> LOGIN ATTEMPT: {req.email}")
    
    email_key = req.email.lower().strip()
    
    # 1. Check DB
    user = db.query(User).filter(User.email == email_key).first()
    
    # 2. SEED DEFAULT ADMIN IF MISSING (For instant local access)
    if not user and email_key == "admin@drone.com":
        print("    Targeting Default Admin - creating if missing...")
        safe_pass = hash_password("password")
        admin = User(
            email="admin@drone.com",
            username="Admin",
            password_hash=safe_pass,
            token="admin-token-permanent",
            params={}
        )
        db.add(admin)
        db.commit()
        user = admin
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Verify Hash
    received_pass = hash_password(req.password.strip())
    
    if user.password_hash != received_pass:
        print(f"    FAILED: Password mismatch for {email_key}")
        raise HTTPException(status_code=401, detail="Invalid password")
        
    print(f"    SUCCESS: User {email_key} logged in")
    return {
        "status": "success",
        "token": user.token,
        "user": {
            "username": user.username,
            "email": req.email,
            "id": user.token
        }
    }
