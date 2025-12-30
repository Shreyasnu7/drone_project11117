from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import os
import uuid

from utils import BASE_DATA_DIR

router = APIRouter()
DB_FILE = os.path.join(BASE_DATA_DIR, "users.json")

# --- Models ---
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

# --- Helper Functions ---
def load_db():
    if not os.path.exists(DB_FILE):
        # 🛡️ PERSISTENCE FIX: Seed Default User
        # On Render Free Tier, files are wiped on restart.
        # We re-create the admin user every time to ensure login works.
        default_db = {}
        # Admin User
        token = "admin-token-permanent"
        default_db["admin@drone.com"] = {
            "username": "Admin",
            "password": hash_password("password"), # "password"
            "token": token,
            "params": {}
        }
        return default_db
        
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_db(db):
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(db, f)
    except Exception as e:
        print(f"❌ DB SAVE ERROR: {e}")

# --- Routes ---
# --- Routes ---
import hashlib
import time

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

@router.post("/session/register")
async def register(req: RegisterRequest):
    print(f"--> REGISTER ATTEMPT: {req.email}")
    db = load_db()
    
    email_key = req.email.lower().strip()
    
    # Check if user exists
    if email_key in db:
        print(f"    FAILED: User {email_key} already exists")
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create simple token
    token = str(uuid.uuid4())
    
    # Hash Password
    safe_password = hash_password(req.password.strip())
    
    # Save user
    user_data = {
        "username": req.username,
        "password": safe_password, 
        "token": token,
        "params": {} 
    }
    db[email_key] = user_data
    save_db(db)
    
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
async def login(req: LoginRequest):
    print(f"--> LOGIN ATTEMPT: {req.email}")
    db = load_db()
    
    email_key = req.email.lower().strip()
    user = db.get(email_key)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Verify Hash
    stored_pass = user.get("password", "")
    received_pass = hash_password(req.password.strip())
    
    if stored_pass != received_pass:
        print(f"    FAILED: Password mismatch for {email_key}")
        raise HTTPException(status_code=401, detail="Invalid password")
        
    print(f"    SUCCESS: User {email_key} logged in")
    return {
        "status": "success",
        "token": user["token"],
        "user": {
            "username": user["username"],
            "email": req.email,
            "id": user["token"]
        }
    }
