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