import os
import re
import io
import base64
import hashlib
import sqlite3
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
from pypdf import PdfReader

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except Exception:
    DDGS_AVAILABLE = False

try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

load_dotenv()

APP_NAME = "जुगनू AI"
DB_FILE = "jugnu_data.db"
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
WHISPER_MODEL = "whisper-large-v3-turbo"
GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

app = FastAPI(title="Jugnu AI Shared Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DATABASE SETUP (100% COMPATIBLE WITH OLD JUGNU)
# =========================================================

def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            name TEXT,
            plan TEXT DEFAULT 'FREE',
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS auth_users (
            username TEXT PRIMARY KEY,
            password_hash TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            title TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            image_data TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            title TEXT,
            content TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            memory TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            username TEXT PRIMARY KEY,
            language TEXT DEFAULT 'Hindi',
            bot_mode TEXT DEFAULT 'दोस्ताना',
            voice_speed TEXT DEFAULT 'सामान्य'
        )
    """)

    # Default Arvind VIP PRO Account
    username = "arvind"
    if cur.execute("SELECT username FROM users WHERE username=?", (username,)).fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, name, plan, created_at) VALUES (?, ?, ?, ?)",
            (username, "अरविंद सिंह", "VIP PRO", datetime.now().isoformat())
        )
    if cur.execute("SELECT username FROM auth_users WHERE username=?", (username,)).fetchone() is None:
        cur.execute(
            "INSERT INTO auth_users (username, password_hash) VALUES (?, ?)",
            (username, hash_password("Jugnu@123"))
        )

    conn.commit()
    conn.close()

init_db()

# =========================================================
# AI CLIENTS & HELPERS
# =========================================================

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def creator_answer():
    return (
        "मुझे अरविंद सिंह ने बनाया है। मेरे निर्माता अरविंद सिंह हैं। "
        "मेरे निर्माता के पिता का नाम Mr Rewant Singh है। "
        "उनका गाँव Doojasar है और वे वर्तमान में Shri Mohangarh में हैं।"
    )

def is_creator_question(prompt: str) -> bool:
    p = prompt.lower().strip()
    phrases = [
        "creator", "who created you", "who made you", "who built you", "maker", "owner",
        "creator कौन", "creator kaun", "maker कौन", "owner कौन", "किसने बनाया",
        "निर्माता कौन", "मालिक कौन", "बनाने वाला कौन", "अरविंद सिंह"
    ]
    return any(phrase in p for phrase in phrases)

def is_image_request(prompt: str) -> bool:
    p = prompt.lower()
    keywords = [
        "image", "photo", "picture", "draw", "generate image", "create image",
        "तस्वीर", "फोटो", "चित्र", "इमेज", "तस्वीर बनाओ", "फोटो बनाओ", "इमेज बनाओ"
    ]
    return any(word in p for word in keywords)

def generate_gemini_image(prompt: str):
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return None, "Gemini API उपलब्ध नहीं है।"
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.interactions.create(
            model=GEMINI_IMAGE_MODEL,
            input=prompt,
            response_format={
                "type": "image",
                "mime_type": "image/jpeg",
                "aspect_ratio": "1:1",
                "image_size": "1K"
            }
        )
        if hasattr(response, "outputs"):
            for output in response.outputs:
                if hasattr(output, "image") and hasattr(output.image, "data"):
                    data = output.image.data
                    if isinstance(data, str):
                        return base64.b64decode(data), None
                    return data, None
        return None, "Gemini ने image data वापस नहीं दिया।"
    except Exception as e:
        return None, str(e)

def web_search(query: str, max_results: int = 5) -> str:
    if not DDGS_AVAILABLE:
        return ""
    try:
        results = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append(item)
        if not results:
            return ""
        text = "\n\nWEB SEARCH RESULTS:\n"
        for i, r in enumerate(results, 1):
            text += f"\n{i}. {r.get('title', '')}\n{r.get('body', '')}\n{r.get('href', '')}\n"
        return text
    except Exception:
        return ""

def make_tts_base64(text: str, language: str = "Hindi", voice_speed: str = "सामान्य") -> Optional[str]:
    if not text:
        return None
    try:
        lang = "en" if language == "English" else "hi"
        slow = (voice_speed == "धीमी")
        tts = gTTS(text=text, lang=lang, slow=slow)
        audio = io.BytesIO()
        tts.write_to_fp(audio)
        audio.seek(0)
        return base64.b64encode(audio.getvalue()).decode("utf-8")
    except Exception:
        return None

def build_system_prompt(language: str, bot_mode: str, memories: list) -> str:
    modes = {
        "दोस्ताना": "तुम जुगनू AI हो। तुम बहुत friendly, सरल और मददगार तरीके से बात करती हो।",
        "शिक्षक": "तुम जुगनू AI हो। तुम एक अच्छे शिक्षक की तरह सरल भाषा में उदाहरण और step-by-step तरीके से समझाती हो।",
        "कहानीकार": "तुम जुगनू AI हो। तुम कहानीकार की तरह रोचक, भावनात्मक और कल्पनाशील अंदाज में जवाब देती हो।",
        "मारवाड़ी / राजस्थानी": "तुम जुगनू AI हो। मारवाड़ी/राजस्थानी अंदाज में सरल और अपनापन भरी भाषा में बात करती हो।"
    }
    personality = modes.get(bot_mode, modes["दोस्ताना"])

    if language == "Hindi":
        lang_inst = "\nहमेशा मुख्य रूप से हिंदी में जवाब दो।\n"
    elif language == "English":
        lang_inst = "\nहमेशा मुख्य रूप से English में जवाब दो।\n"
    else:
        lang_inst = "\nHindi + English दोनों का natural mixture इस्तेमाल करो।\n"

    mem_text = ""
    if memories:
        mem_text = "\nUser Memories:\n" + "\n".join(f"- {m['memory']}" for m in memories)

    creator_inst = """
IMPORTANT CREATOR INFORMATION:
अगर user पूछे कि तुम्हें किसने बनाया या तुम्हारा creator कौन है, तो हमेशा बताना कि तुम्हें अरविंद सिंह ने बनाया है।
अरविंद सिंह जुगनू AI के निर्माता हैं। उनके पिता का नाम Mr Rewant Singh है। उनका गाँव Doojasar है और वे वर्तमान में Shri Mohangarh में हैं।
"""
    return personality + lang_inst + creator_inst + mem_text

# =========================================================
# REQUEST MODELS
# =========================================================

class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    username: str
    name: str
    password: str

class PasswordChangeRequest(BaseModel):
    username: str
    old_password: str
    new_password: str

class SettingsRequest(BaseModel):
    username: str
    language: str
    bot_mode: str
    voice_speed: str

class ChatRequest(BaseModel):
    username: str
    conversation_id: Optional[int] = None
    prompt: str
    web_search_enabled: bool = False
    voice_response: bool = True
    file_context: str = ""

class NoteRequest(BaseModel):
    username: str
    title: str
    content: str

# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/")
def home():
    return {"status": "online", "app": APP_NAME, "message": "जुगनू AI Backend तैयार है!"}

@app.post("/api/login")
def login(req: LoginRequest):
    conn = get_db()
    auth = conn.execute("SELECT password_hash FROM auth_users WHERE username=?", (req.username,)).fetchone()
    if not auth or auth["password_hash"] != hash_password(req.password):
        conn.close()
        raise HTTPException(status_code=401, detail="Username या Password गलत है।")
    user = dict(conn.execute("SELECT * FROM users WHERE username=?", (req.username,)).fetchone())
    conn.close()
    return {"success": True, "user": user}

@app.post("/api/signup")
def signup(req: SignupRequest):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, name, plan, created_at) VALUES (?, ?, ?, ?)",
            (req.username, req.name, "FREE", datetime.now().isoformat())
        )
        conn.execute(
            "INSERT INTO auth_users (username, password_hash) VALUES (?, ?)",
            (req.username, hash_password(req.password))
        )
        conn.commit()
        return {"success": True, "message": "Account बन गया।"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="यह Username पहले से मौजूद है।")
    finally:
        conn.close()

@app.post("/api/guest")
def guest_login():
    guest_username = "guest_" + datetime.now().strftime("%Y%m%d%H%M%S")
    conn = get_db()
    conn.execute(
        "INSERT INTO users (username, name, plan, created_at) VALUES (?, ?, ?, ?)",
        (guest_username, "Guest User", "FREE", datetime.now().isoformat())
    )
    conn.execute(
        "INSERT INTO auth_users (username, password_hash) VALUES (?, ?)",
        (guest_username, hash_password(os.urandom(16).hex()))
    )
    conn.commit()
    user = dict(conn.execute("SELECT * FROM users WHERE username=?", (guest_username,)).fetchone())
    conn.close()
    return {"success": True, "user": user}

@app.post("/api/change-password")
def change_password(req: PasswordChangeRequest):
    conn = get_db()
    auth = conn.execute("SELECT password_hash FROM auth_users WHERE username=?", (req.username,)).fetchone()
    if not auth or auth["password_hash"] != hash_password(req.old_password):
        conn.close()
        raise HTTPException(status_code=400, detail="पुराना Password गलत है।")
    conn.execute("UPDATE auth_users SET password_hash=? WHERE username=?", (hash_password(req.new_password), req.username))
    conn.commit()
    conn.close()
    return {"success": True, "message": "Password बदल गया।"}

@app.get("/api/settings/{username}")
def get_settings(username: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM app_settings WHERE username=?", (username,)).fetchone()
    if not row:
        conn.execute("INSERT INTO app_settings (username) VALUES (?)", (username,))
        conn.commit()
        row = conn.execute("SELECT * FROM app_settings WHERE username=?", (username,)).fetchone()
    data = dict(row)
    conn.close()
    return data

@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO app_settings (username, language, bot_mode, voice_speed) VALUES (?, ?, ?, ?)",
        (req.username, req.language, req.bot_mode, req.voice_speed)
    )
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/conversations/{username}")
def get_conversations(username: str):
    conn = get_db()
    rows = conn.execute("SELECT * FROM conversations WHERE username=? ORDER BY id DESC", (username,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/conversations/{username}")
def new_conversation(username: str, title: str = "नई चैट"):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO conversations (username, title, created_at) VALUES (?, ?, ?)", (username, title, datetime.now().isoformat()))
    conv_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"id": conv_id, "title": title}

@app.delete("/api/conversations/{username}/{conversation_id}")
def delete_conversation(username: str, conversation_id: int):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE conversation_id=? AND username=?", (conversation_id, username))
    conn.execute("DELETE FROM conversations WHERE id=? AND username=?", (conversation_id, username))
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/messages/{username}/{conversation_id}")
def get_messages(username: str, conversation_id: int):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM messages WHERE username=? AND conversation_id=? ORDER BY id ASC",
        (username, conversation_id)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/memories/{username}")
def get_memories(username: str):
    conn = get_db()
    rows = conn.execute("SELECT * FROM user_memories WHERE username=? ORDER BY id DESC LIMIT 20", (username,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.delete("/api/memories/{username}")
def clear_memories(username: str):
    conn = get_db()
    conn.execute("DELETE FROM user_memories WHERE username=?", (username,))
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/notes/{username}")
def get_notes(username: str):
    conn = get_db()
    rows = conn.execute("SELECT * FROM notes WHERE username=? ORDER BY id DESC", (username,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/notes")
def add_note(req: NoteRequest):
    conn = get_db()
    conn.execute(
        "INSERT INTO notes (username, title, content, created_at) VALUES (?, ?, ?, ?)",
        (req.username, req.title or "Note", req.content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return {"success": True}

@app.post("/api/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    content = await file.read()
    if file.filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        try:
            text = content.decode("utf-8")
        except Exception:
            text = content.decode("latin-1", errors="ignore")
    return {"filename": file.filename, "text": text[:30000]}

@app.post("/api/transcribe")
async def transcribe_voice(file: UploadFile = File(...)):
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY नहीं मिली।")
    audio_bytes = await file.read()
    temp_name = f"temp_{datetime.now().timestamp()}.wav"
    with open(temp_name, "wb") as f:
        f.write(audio_bytes)
    try:
        with open(temp_name, "rb") as f:
            res = groq_client.audio.transcriptions.create(file=f, model=WHISPER_MODEL)
        return {"text": res.text}
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)

@app.post("/api/chat")
def chat(req: ChatRequest):
    conn = get_db()
    conv_id = req.conversation_id
    if not conv_id:
        cur = conn.cursor()
        cur.execute("INSERT INTO conversations (username, title, created_at) VALUES (?, ?, ?)", (req.username, req.prompt[:50], datetime.now().isoformat()))
        conv_id = cur.lastrowid

    conn.execute(
        "INSERT INTO messages (username, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (req.username, conv_id, "user", req.prompt, datetime.now().isoformat())
    )
    conn.commit()

    settings = conn.execute("SELECT * FROM app_settings WHERE username=?", (req.username,)).fetchone()
    language = settings["language"] if settings else "Hindi"
    bot_mode = settings["bot_mode"] if settings else "दोस्ताना"
    voice_speed = settings["voice_speed"] if settings else "सामान्य"

    reply_text = ""
    image_b64 = None

    if is_creator_question(req.prompt):
        reply_text = creator_answer()
    elif is_image_request(req.prompt):
        img_bytes, err = generate_gemini_image(req.prompt)
        if img_bytes:
            image_b64 = base64.b64encode(img_bytes).decode("utf-8")
            reply_text = "✨ मैंने आपके लिए image तैयार कर दी है।"
        else:
            reply_text = f"Image बनाने में समस्या आई: {err}"
    elif re.search(r"(याद रखना|याद रखो|remember that|remember)", req.prompt, re.IGNORECASE):
        mem = re.sub(r"(याद रखना|याद रखो|remember that|remember)", "", req.prompt, flags=re.IGNORECASE).strip()
        if mem:
            conn.execute("INSERT INTO user_memories (username, memory, created_at) VALUES (?, ?, ?)", (req.username, mem[:1000], datetime.now().isoformat()))
            reply_text = "ठीक है 😊 मैंने इसे याद रखने के लिए save कर लिया।"
    else:
        if not groq_client:
            reply_text = "GROQ_API_KEY उपलब्ध नहीं है।"
        else:
            memories = conn.execute("SELECT memory FROM user_memories WHERE username=? ORDER BY id DESC LIMIT 20", (req.username,)).fetchall()
            sys_prompt = build_system_prompt(language, bot_mode, memories)
            extra = ""
            if req.web_search_enabled:
                extra += web_search(req.prompt)
            if req.file_context:
                extra += "\n\nUploaded File Content:\n" + req.file_context[:30000]

            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": req.prompt + extra}
            ]
            try:
                res = groq_client.chat.completions.create(model=DEFAULT_MODEL, messages=msgs, temperature=0.7)
                reply_text = res.choices[0].message.content
            except Exception:
                try:
                    res = groq_client.chat.completions.create(model=FALLBACK_MODEL, messages=msgs, temperature=0.7)
                    reply_text = res.choices[0].message.content
                except Exception as e2:
                    reply_text = f"AI response में समस्या आई: {e2}"

    conn.execute(
        "INSERT INTO messages (username, conversation_id, role, content, image_data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (req.username, conv_id, "assistant", reply_text, image_b64, datetime.now().isoformat())
    )

    msg_count = conn.execute("SELECT COUNT(*) as c FROM messages WHERE conversation_id=? AND role='user'", (conv_id,)).fetchone()["c"]
    if msg_count == 1:
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (req.prompt[:50], conv_id))

    conn.commit()
    conn.close()

    audio_b64 = make_tts_base64(reply_text, language, voice_speed) if req.voice_response else None

    return {
        "conversation_id": conv_id,
        "reply": reply_text,
        "image_base64": image_b64,
        "audio_base64": audio_b64
    }