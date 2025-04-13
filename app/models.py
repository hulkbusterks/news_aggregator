from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any, List
from uuid import uuid4

class FeedCreate(BaseModel):
    name: str
    url: HttpUrl
    
class FeedUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[HttpUrl] = None
    
class Feed(BaseModel):
    id: str
    name: str
    url: HttpUrl
    last_updated: str
    
class FeedContent(BaseModel):
    id: str
    content: Dict[str, Any]
    
class ErrorResponse(BaseModel):
    detail: str 

class GroqRequest(BaseModel):
    text: str
    operation: str  # "generate", "summarize", "explain", "translate"
    mode: Optional[str] = "default"
    target_language: Optional[str] = "en"
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50

class ExtractBlogRequest(BaseModel):
    url: str
class ChatRequest(BaseModel):
    messages: list
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50