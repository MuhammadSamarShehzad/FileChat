import sqlite3
import json
from typing import List, Dict, Optional


class FileChatDB:
    def __init__(self, db_path: str = "chatbot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pdfs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    chunks TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_threads (
                    id TEXT PRIMARY KEY,
                    pdf_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (thread_id) REFERENCES chat_threads (id)
                )
            """)
            conn.commit()
        finally:
            conn.close()
    
    def store_pdf(self, filename: str, content_hash: str, chunks: List[str]) -> int:
        """Store PDF metadata and chunks, return PDF ID. 
        If PDF with same content already exists, return existing ID instead of overwriting."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Check if PDF with same content hash already exists
            existing_pdf = conn.execute("SELECT id FROM pdfs WHERE content_hash = ?", (content_hash,)).fetchone()
            
            if existing_pdf:
                # PDF already exists, return existing ID (don't overwrite)
                return existing_pdf[0]
            else:
                # New PDF, insert it
                cursor = conn.execute(
                    "INSERT INTO pdfs (filename, content_hash, chunks) VALUES (?, ?, ?)",
                    (filename, content_hash, json.dumps(chunks))
                )
                conn.commit()
                return cursor.lastrowid
        finally:
            conn.close()
    
    def get_pdf_chunks(self, pdf_id: int) -> List[str]:
        """Retrieve PDF chunks by ID"""
        conn = sqlite3.connect(self.db_path)
        try:
            result = conn.execute("SELECT chunks FROM pdfs WHERE id = ?", (pdf_id,)).fetchone()
            return json.loads(result[0]) if result else []
        finally:
            conn.close()
    
    def create_chat_thread(self, thread_id: str, pdf_id: Optional[int] = None):
        """Create a new chat thread or update existing one with PDF.
        If thread already has a PDF, don't overwrite it unless explicitly requested."""
        conn = sqlite3.connect(self.db_path)
        try:
            existing = conn.execute("SELECT id, pdf_id FROM chat_threads WHERE id = ?", (thread_id,)).fetchone()
            
            if existing:
                existing_pdf_id = existing[1]
                if pdf_id is not None and existing_pdf_id is None:
                    # Thread exists but has no PDF, add the PDF
                    conn.execute("UPDATE chat_threads SET pdf_id = ? WHERE id = ?", (pdf_id, thread_id))
                # If thread already has a PDF, don't overwrite it (preserve existing connection)
            else:
                # Create new thread
                conn.execute("INSERT INTO chat_threads (id, pdf_id) VALUES (?, ?)", (thread_id, pdf_id))
            
            conn.commit()
        finally:
            conn.close()
    
    def add_message(self, thread_id: str, role: str, content: str):
        """Add a message to a chat thread"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO chat_messages (thread_id, role, content) VALUES (?, ?, ?)",
                (thread_id, role, content)
            )
            conn.commit()
        finally:
            conn.close()
    
    def get_chat_history(self, thread_id: str) -> List[Dict]:
        """Get chat history for a thread"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT role, content, timestamp FROM chat_messages WHERE thread_id = ? ORDER BY timestamp",
                (thread_id,)
            )
            return [
                {"role": row[0], "content": row[1], "timestamp": row[2]}
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()
    
    def get_all_threads(self) -> List[Dict]:
        """Get all chat threads with PDF info"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT t.id, t.created_at, p.filename, 
                       (SELECT COUNT(*) FROM chat_messages WHERE thread_id = t.id) as message_count
                FROM chat_threads t
                LEFT JOIN pdfs p ON t.pdf_id = p.id
                ORDER BY t.created_at DESC
            """)
            return [
                {
                    "id": row[0],
                    "created_at": row[1],
                    "pdf_name": row[2] or "No PDF",
                    "message_count": row[3]
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()
    
    def get_thread_pdf_id(self, thread_id: str) -> Optional[int]:
        """Get PDF ID associated with a thread"""
        conn = sqlite3.connect(self.db_path)
        try:
            result = conn.execute("SELECT pdf_id FROM chat_threads WHERE id = ?", (thread_id,)).fetchone()
            return result[0] if result else None
        finally:
            conn.close()
    
    def get_pdf_threads(self, pdf_id: int) -> List[Dict]:
        """Get all threads that are linked to a specific PDF"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT t.id, t.created_at, 
                       (SELECT COUNT(*) FROM chat_messages WHERE thread_id = t.id) as message_count
                FROM chat_threads t
                WHERE t.pdf_id = ?
                ORDER BY t.created_at DESC
            """, (pdf_id,))
            return [
                {
                    "thread_id": row[0],
                    "created_at": row[1],
                    "message_count": row[2]
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()
    
    def cleanup_empty_threads(self):
        """Remove threads that have no messages and no PDF content"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Delete threads with no messages and no PDF
            conn.execute("""
                DELETE FROM chat_threads 
                WHERE id NOT IN (
                    SELECT DISTINCT thread_id FROM chat_messages
                ) 
                AND pdf_id IS NULL
            """)
            conn.commit()
        finally:
            conn.close()
    
    def get_langgraph_connection(self):
        """Get SQLite connection for LangGraph checkpointer"""
        return sqlite3.connect(self.db_path, check_same_thread=False)


# Global database instance
db = FileChatDB() 