from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from .models import FeedCreate, FeedUpdate, Feed, FeedContent, ErrorResponse
from .database import FeedDatabase
from .services import parse_xml_feed, generate_feed_id
from typing import List
from pydantic import HttpUrl, BaseModel
import trafilatura

router = APIRouter(prefix="/feeds", tags=["feeds"])

@router.post("/", response_model=Feed)
async def create_feed(feed_data: FeedCreate, background_tasks: BackgroundTasks):
    # Generate ID
    feed_id = generate_feed_id(feed_data.name)
    
    try:
        # Parse XML
        content = parse_xml_feed(str(feed_data.url))
        
        # Save to database
        FeedDatabase.save_feed(feed_id, feed_data.name, str(feed_data.url), content)
        
        # Return feed metadata
        return FeedDatabase.get_feed_metadata(feed_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[Feed])
async def list_feeds():
    return FeedDatabase.list_feeds()

@router.get("/{feed_id}", response_model=Feed)
async def get_feed(feed_id: str):
    feed = FeedDatabase.get_feed_metadata(feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    return feed

@router.get("/{feed_id}/content", response_model=FeedContent)
async def get_feed_content(feed_id: str):
    feed = FeedDatabase.get_feed_metadata(feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    content = FeedDatabase.get_feed_content(feed_id)
    return {"id": feed_id, "content": content}

@router.put("/{feed_id}", response_model=Feed)
async def update_feed(feed_id: str, feed_data: FeedUpdate):
    feed = FeedDatabase.get_feed_metadata(feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    # Update fields
    name = feed_data.name if feed_data.name is not None else feed["name"]
    url = str(feed_data.url) if feed_data.url is not None else feed["url"]
    
    try:
        # Parse XML
        content = parse_xml_feed(url)
        
        # Save updated feed
        FeedDatabase.save_feed(feed_id, name, url, content)
        
        # Return updated metadata
        return FeedDatabase.get_feed_metadata(feed_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{feed_id}", response_model=dict)
async def delete_feed(feed_id: str):
    if not FeedDatabase.delete_feed(feed_id):
        raise HTTPException(status_code=404, detail="Feed not found")
    return {"status": "success", "message": f"Feed {feed_id} deleted"}

@router.post("/{feed_id}/refresh", response_model=Feed)
async def refresh_feed(feed_id: str):
    feed = FeedDatabase.get_feed_metadata(feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    try:
        # Parse XML
        content = parse_xml_feed(feed["url"])
        
        # Save updated feed
        FeedDatabase.save_feed(feed_id, feed["name"], feed["url"], content)
        
        # Return updated metadata
        return FeedDatabase.get_feed_metadata(feed_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Add this model for the extract blog request
class ExtractBlogRequest(BaseModel):
    url: str

@router.post("/extractblog", response_model=dict)
async def extract_blog(request: ExtractBlogRequest):
    try:
        downloaded = trafilatura.fetch_url(request.url)
        content = trafilatura.extract(downloaded)
        
        if not content:
            raise HTTPException(status_code=400, detail="Could not extract content from the provided URL")
        
        return {"url": request.url, "content": content}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 
    

