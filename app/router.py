from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from .models import FeedCreate, FeedUpdate, Feed, FeedItem, ErrorResponse, GroqRequest, ExtractBlogRequest, ChatRequest
from .database import FeedDatabase
from .services import parse_xml_feed, generate_feed_id
from typing import List, Optional
from pydantic import HttpUrl, BaseModel
import trafilatura
from .grok_model import GroqAgent

router = APIRouter(prefix="/feeds", tags=["feeds"])

# Initialize the GroqAgent
groq_agent = GroqAgent()

@router.post("/", response_model=Feed)
async def create_feed(feed_data: FeedCreate, background_tasks: BackgroundTasks):
    # Generate ID
    feed_id = generate_feed_id(feed_data.name)
    
    try:
        # Parse XML
        items = parse_xml_feed(str(feed_data.url))
        
        # Save to database
        FeedDatabase.save_feed(feed_id, feed_data.name, str(feed_data.url), feed_data.category, items)
        
        # Return feed
        return FeedDatabase.get_feed(feed_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[Feed])
async def list_feeds():
    return FeedDatabase.list_feeds()

@router.get("/category/{category}", response_model=List[Feed])
async def get_feeds_by_category(category: str):
    """Get all feeds in a specific category"""
    feeds = FeedDatabase.get_feeds_by_category(category)
    if not feeds:
        raise HTTPException(
            status_code=404,
            detail=f"No feeds found in category '{category}'"
        )
    return feeds

@router.get("/{feed_id}", response_model=Feed)
async def get_feed(feed_id: str):
    feed = FeedDatabase.get_feed(feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    return feed

@router.get("/{feed_id}/items", response_model=List[FeedItem])
async def get_feed_items(feed_id: str):
    items = FeedDatabase.get_feed_items(feed_id)
    if items is None:
        raise HTTPException(status_code=404, detail="Feed not found")
    return items

@router.put("/{feed_id}", response_model=Feed)
async def update_feed(feed_id: str, feed_data: FeedUpdate):
    feed = FeedDatabase.get_feed(feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    # Update fields
    name = feed_data.name if feed_data.name is not None else feed["name"]
    url = str(feed_data.url) if feed_data.url is not None else feed["url"]
    category = feed_data.category if feed_data.category is not None else feed["category"]
    
    try:
        # Parse XML
        items = parse_xml_feed(url)
        
        # Save updated feed
        FeedDatabase.save_feed(feed_id, name, url, category, items)
        
        # Return updated feed
        return FeedDatabase.get_feed(feed_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{feed_id}", response_model=dict)
async def delete_feed(feed_id: str):
    if not FeedDatabase.delete_feed(feed_id):
        raise HTTPException(status_code=404, detail="Feed not found")
    return {"status": "success", "message": f"Feed {feed_id} deleted"}

@router.post("/{feed_id}/refresh", response_model=Feed)
async def refresh_feed(feed_id: str):
    feed = FeedDatabase.get_feed(feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    try:
        # Parse XML
        items = parse_xml_feed(feed["url"])
        
        # Save updated feed
        FeedDatabase.save_feed(feed_id, feed["name"], feed["url"], feed["category"], items)
        
        # Return updated feed
        return FeedDatabase.get_feed(feed_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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

@router.post("/groq", response_model=dict)
async def process_with_groq(request: GroqRequest):
    try:
        # Validate input
        if not request.text and not request.feed_id:
            raise HTTPException(
                status_code=400,
                detail="Either text or feed_id must be provided"
            )
        
        # Get content based on input type
        content = request.text
        if request.feed_id:
            feed = FeedDatabase.get_feed(request.feed_id)
            if not feed:
                raise HTTPException(status_code=404, detail="Feed not found")
            
            if request.item_id:
                # Find specific item in feed
                item = next((item for item in feed["items"] if item["id"] == request.item_id), None)
                if not item:
                    raise HTTPException(status_code=404, detail="Item not found in feed")
                content = item["description"]
            else:
                # Use feed URL to extract content
                blog_response = await extract_blog(ExtractBlogRequest(url=feed["url"]))
                content = blog_response["content"]
        
        # Process content based on operation
        if request.operation == "generate":
            result = groq_agent.generate_text(
                prompt=content,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
            return {"response": result}
        
        elif request.operation == "summarize":
            result = groq_agent.summarize(
                text=content, 
                mode=request.mode
            )
            return {"summary": result}
        
        elif request.operation == "explain":
            result = groq_agent.explain(
                text=content, 
                mode=request.mode
            )
            return {"explanation": result}
        
        elif request.operation == "translate":
            result = groq_agent.translate(
                text=content, 
                target_language=request.target_language
            )
            return {"translation": result}
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid operation: {request.operation}. Must be one of: generate, summarize, explain, translate"
            )
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/groq/chat", response_model=dict)
async def chat_with_groq(request: ChatRequest):
    try:
        result = groq_agent.chat(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class URLProcessRequest(BaseModel):
    url: str
    operation: str  # "summarize", "explain", "translate"
    mode: Optional[str] = "default"
    target_language: Optional[str] = "hi"

@router.post("/process-url", response_model=dict)
async def process_url_content(request: URLProcessRequest):
    """Extract content from URL and perform the requested operation"""
    try:
        # Extract content from URL
        blog_response = await extract_blog(ExtractBlogRequest(url=request.url))
        content = blog_response["content"]
        
        # Process content based on operation
        if request.operation == "summarize":
            result = groq_agent.summarize(
                text=content, 
                mode=request.mode
            )
            return {
                "url": request.url,
                "operation": "summarize",
                "content": content,
                "result": result
            }
        
        elif request.operation == "explain":
            result = groq_agent.explain(
                text=content, 
                mode=request.mode
            )
            return {
                "url": request.url,
                "operation": "explain",
                "content": content,
                "result": result
            }
        
        elif request.operation == "translate":
            result = groq_agent.translate(
                text=content, 
                target_language=request.target_language
            )
            return {
                "url": request.url,
                "operation": "translate",
                "content": content,
                "result": result
            }
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid operation: {request.operation}. Must be one of: summarize, explain, translate"
            )
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 
    

