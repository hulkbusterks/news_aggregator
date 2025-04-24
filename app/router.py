from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from .models import FeedCreate, FeedUpdate, Feed, FeedItem, ErrorResponse, GroqRequest, ExtractBlogRequest, ChatRequest, MediaItem
from .database import FeedDatabase
from .services import parse_xml_feed, generate_feed_id
from typing import List, Optional
from pydantic import HttpUrl, BaseModel
import trafilatura
from .groq_model import GroqAgent
import sqlite3
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from pydantic import BaseModel, Field

router = APIRouter(prefix="/feeds", tags=["feeds"])

# Initialize the GroqAgent
groq_agent = GroqAgent()

# FAISS configuration
FAISS_INDEX_PATH = "faiss_index"
INDEX_NAME = "combined_embeddings"
SENTENCE_SIMILARITY_MODEL = 'all-MiniLM-L6-v2'

# Initialize database
try:
    FeedDatabase.initialize_database()
except sqlite3.Error as e:
    print(f"Error initializing database: {e}")
    raise

@router.get("/health")
async def health_check():
    """Check database connectivity and health"""
    try:
        with FeedDatabase.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            return {"status": "healthy", "database": "connected"}
    except sqlite3.Error as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

@router.post("/", response_model=Feed)
async def create_feed(feed_data: FeedCreate, background_tasks: BackgroundTasks):
    try:
        # Generate ID
        feed_id = generate_feed_id(feed_data.name)
        
        # Parse XML
        items = parse_xml_feed(str(feed_data.url))
        
        # Save to database
        FeedDatabase.save_feed(feed_id, feed_data.name, str(feed_data.url), feed_data.category, items)
        
        # Return feed
        feed = FeedDatabase.get_feed(feed_id)
        if not feed:
            raise HTTPException(status_code=500, detail="Failed to retrieve created feed")
        return feed
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[Feed])
async def list_feeds():
    try:
        return FeedDatabase.list_feeds()
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/category/{category}", response_model=List[Feed])
async def get_feeds_by_category(category: str):
    """Get all feeds in a specific category"""
    try:
        feeds = FeedDatabase.get_feeds_by_category(category)
        if not feeds:
            raise HTTPException(
                status_code=404,
                detail=f"No feeds found in category '{category}'"
            )
        return feeds
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/{feed_id}", response_model=Feed)
async def get_feed(feed_id: str):
    try:
        feed = FeedDatabase.get_feed(feed_id)
        if not feed:
            raise HTTPException(status_code=404, detail="Feed not found")
        return feed
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/{feed_id}/items", response_model=List[FeedItem])
async def get_feed_items(feed_id: str):
    """Get items for a specific feed"""
    try:
        feed = FeedDatabase.get_feed(feed_id)
        if not feed:
            raise HTTPException(status_code=404, detail="Feed not found")
        return feed["items"]
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/{feed_id}/items/{item_id}/media", response_model=List[MediaItem])
async def get_item_media(feed_id: str, item_id: str):
    """Get media items for a specific feed item"""
    feed = FeedDatabase.get_feed(feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    item = next((item for item in feed["items"] if item["id"] == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return item.get("media", [])

@router.get("/{feed_id}/items/{item_id}/media/{media_type}", response_model=List[MediaItem])
async def get_item_media_by_type(feed_id: str, item_id: str, media_type: str):
    """Get media items of a specific type for a feed item"""
    feed = FeedDatabase.get_feed(feed_id)
    if not feed:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    item = next((item for item in feed["items"] if item["id"] == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    media_items = [media for media in item.get("media", []) if media["type"].lower() == media_type.lower()]
    return media_items

@router.put("/{feed_id}", response_model=Feed)
async def update_feed(feed_id: str, feed_data: FeedUpdate):
    try:
        feed = FeedDatabase.get_feed(feed_id)
        if not feed:
            raise HTTPException(status_code=404, detail="Feed not found")
        
        # Update fields
        name = feed_data.name if feed_data.name is not None else feed["name"]
        url = str(feed_data.url) if feed_data.url is not None else feed["url"]
        category = feed_data.category if feed_data.category is not None else feed["category"]
        
        # Parse XML
        items = parse_xml_feed(url)
        
        # Save updated feed
        FeedDatabase.save_feed(feed_id, name, url, category, items)
        
        # Return updated feed
        updated_feed = FeedDatabase.get_feed(feed_id)
        if not updated_feed:
            raise HTTPException(status_code=500, detail="Failed to retrieve updated feed")
        return updated_feed
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{feed_id}", response_model=dict)
async def delete_feed(feed_id: str):
    try:
        if not FeedDatabase.delete_feed(feed_id):
            raise HTTPException(status_code=404, detail="Feed not found")
        return {"status": "success", "message": f"Feed {feed_id} deleted"}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

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

@router.post("/initialize-faiss")
async def initialize_faiss():
    """Initialize FAISS index with empty data"""
    try:
        # Create directory if it doesn't exist
        Path(FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_SIMILARITY_MODEL)
        
        # Create empty FAISS index
        docsearch = FAISS.from_texts(
            texts=["Initial document"],
            embedding=embedding_function,
            metadatas=[{"source": "initial"}]
        )
        
        # Save the index
        docsearch.save_local(FAISS_INDEX_PATH, index_name=INDEX_NAME)
        
        return {
            "status": "success",
            "message": "FAISS index initialized successfully",
            "index_path": FAISS_INDEX_PATH
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing FAISS: {str(e)}")

@router.post("/load-feeds-to-faiss")
async def load_feeds_to_faiss():
    """Load all feed data into FAISS index"""
    try:
        # Get all feeds
        feeds = FeedDatabase.list_feeds()
        if not feeds:
            raise HTTPException(status_code=404, detail="No feeds found in database")
        
        # Prepare documents for FAISS
        documents = []
        for feed in feeds:
            for item in feed.get("items", []):
                content = item.get("description", "") or item.get("title", "")
                if content:
                    documents.append({
                        "page_content": content,
                        "metadata": {
                            "feed_id": feed["id"],
                            "item_id": item["id"],
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "pub_date": item.get("pub_date", "")
                        }
                    })
        
        if not documents:
            raise HTTPException(status_code=404, detail="No valid content found in feeds")
        
        # Initialize embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_SIMILARITY_MODEL)
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        # Split documents
        texts = text_splitter.create_documents(
            [doc["page_content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )
        
        # Create or load existing FAISS index
        if os.path.exists(FAISS_INDEX_PATH):
            docsearch = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embedding_function, 
                index_name=INDEX_NAME,
                allow_dangerous_deserialization=True  # Safe since we're loading our own local index
            )
            # Add new documents to existing index
            docsearch.add_texts(
                texts=[t.page_content for t in texts],
                metadatas=[t.metadata for t in texts]
            )
        else:
            # Create new index
            docsearch = FAISS.from_texts(
                texts=[t.page_content for t in texts],
                embedding=embedding_function,
                metadatas=[t.metadata for t in texts]
            )
        
        # Save the index
        docsearch.save_local(FAISS_INDEX_PATH, index_name=INDEX_NAME)
        
        return {
            "status": "success",
            "message": "Feeds loaded into FAISS successfully",
            "documents_processed": len(documents),
            "index_path": FAISS_INDEX_PATH
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading feeds to FAISS: {str(e)}")

class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query to find relevant feed items")
    k: Optional[int] = Field(default=5, description="Number of results to return")

@router.post("/search")
async def search_feeds(search_query: SearchQuery):
    """Search for relevant feed items using semantic search"""
    try:
        # Check if FAISS index exists
        if not os.path.exists(FAISS_INDEX_PATH):
            raise HTTPException(
                status_code=404,
                detail="FAISS index not found. Please initialize and load feeds first."
            )
        
        # Initialize embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_SIMILARITY_MODEL)
        
        # Load FAISS index
        docsearch = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embedding_function, 
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True  # Safe since we're loading our own local index
        )
        
        # Perform similarity search
        docs = docsearch.similarity_search_with_score(
            search_query.query,
            k=search_query.k
        )
        
        # Process results
        results = []
        for doc, score in docs:
            metadata = doc.metadata
            feed_id = metadata.get("feed_id")
            item_id = metadata.get("item_id")
            
            # Get feed and item details
            feed = FeedDatabase.get_feed(feed_id)
            if not feed:
                continue
                
            item = next((item for item in feed.get("items", []) if item["id"] == item_id), None)
            if not item:
                continue
            # Get media items for this feed item
            media_items = item.get("media", [])
            image_links = [media["url"] for media in media_items if media.get("type", "").lower() in ["image", "img", "thumbnail"]]

            results.append({
                "feed": {
                    "id": feed_id,
                    "name": feed["name"],
                    "category": feed["category"]
                },
                "item": {
                    "id": item_id,
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "pub_date": item.get("pub_date", ""),
                    "description": item.get("description", "")
                },
                "media": {
                    "images": image_links,
                    "all_media": media_items
                },
                "relevance_score": float(score),
                "matched_content": doc.page_content
            })
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant feed items found for the given query"
            )
        
        return {
            "status": "success",
            "query": search_query.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")


class UserInterest(BaseModel):
    interest: str = Field(..., description="User interest topic")

class UserInterests(BaseModel):
    interests: List[UserInterest] = Field(..., description="List of user interests", max_items=9)

# Replace the global variable with file-based storage
import json
import os

# File path for storing user interests
INTERESTS_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "user_interests.json")

# Ensure the data directory exists
os.makedirs(os.path.dirname(INTERESTS_FILE_PATH), exist_ok=True)

# Function to load interests from file
def load_interests():
    if not os.path.exists(INTERESTS_FILE_PATH):
        return []
    try:
        with open(INTERESTS_FILE_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

# Function to save interests to file
def save_interests(interests):
    # Ensure directory exists
    os.makedirs(os.path.dirname(INTERESTS_FILE_PATH), exist_ok=True)
    with open(INTERESTS_FILE_PATH, 'w') as f:
        json.dump(interests, f)

@router.post("/interests")
async def add_interests(interests: UserInterests):
    """Add or update user interests (maximum 9)"""
    try:
        # Get existing interests
        existing_interests = load_interests()
        
        # Add new interests (avoiding duplicates)
        new_interests = [interest.interest for interest in interests.interests]
        
        # Combine existing and new interests, remove duplicates, and limit to 9
        combined_interests = list(set(existing_interests + new_interests))[:9]
        
        # Save to file
        save_interests(combined_interests)
        
        return {
            "status": "success",
            "message": f"Successfully added interests. Total: {len(combined_interests)}",
            "interests": combined_interests
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error adding interests: {str(e)}")

@router.delete("/interests")
async def reset_interests():
    """Reset all user interests"""
    try:
        # Clear all interests
        save_interests([])
        
        return {
            "status": "success",
            "message": "Successfully reset all interests",
            "interests": []
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error resetting interests: {str(e)}")

# @router.get("/interests", response_model=dict)
# async def get_interests():
#     """Get current user interests"""
#     try:
#         # Check if file exists
#         if not os.path.exists(INTERESTS_FILE_PATH):
#             return {
#                 "status": "success",
#                 "interests": [],
#                 "note": "No interests file found"
#             }
        
#         # Try to load interests with explicit error handling
#         try:
#             with open(INTERESTS_FILE_PATH, 'r') as f:
#                 user_interests = json.load(f)
#         except json.JSONDecodeError:
#             # File exists but contains invalid JSON
#             return {
#                 "status": "success",
#                 "interests": [],
#                 "note": "Interest file contains invalid JSON, resetting"
#             }
#         except Exception as e:
#             # Other file reading errors
#             return {
#                 "status": "success",
#                 "interests": [],
#                 "note": f"Error reading interests file: {str(e)}"
#             }
        
#         return {
#             "status": "success",
#             "interests": user_interests
#         }
#     except Exception as e:
#         # Catch-all for any other errors
#         return {
#             "status": "success",
#             "interests": [],
#             "error": str(e)
#         }

@router.post("/recommended-feeds")
async def get_recommended_feeds():
    """Get feed recommendations based on user interests"""
    try:
        # Load interests from file
        user_interests = load_interests()
        
        if not user_interests:
            raise HTTPException(
                status_code=400,
                detail="No interests found. Please add interests first."
            )
        
        # Get all feeds for random selection
        all_feeds = FeedDatabase.list_feeds()
        if not all_feeds:
            raise HTTPException(status_code=404, detail="No feeds found in database")
        
        import random
        
        # Track all results to avoid duplicates
        all_results = []
        seen_items = set()  # Track seen items by their IDs
        
        # Get more results for each interest (5 per interest instead of 3)
        for interest in user_interests:
            # Create a search query for this interest with more results
            search_query = SearchQuery(query=interest, k=5)
            
            try:
                # Use the existing search function
                search_results = await search_feeds(search_query)
                
                # Add non-duplicate results
                for result in search_results.get("results", []):
                    item_id = f"{result['feed']['id']}:{result['item']['id']}"
                    if item_id not in seen_items:
                        all_results.append(result)
                        seen_items.add(item_id)
            except HTTPException:
                # Continue if no results for this interest
                continue
        
        # Add random feeds that aren't already included
        # Keep adding until we reach 30 total or run out of attempts
        random_feeds = []
        attempts = 0
        max_attempts = 50  # Increased from 20 to 50
        
        while len(all_results) + len(random_feeds) < 30 and attempts < max_attempts:
            attempts += 1
            random_feed = random.choice(all_feeds)
            
            if not random_feed.get("items"):
                continue
                
            random_item = random.choice(random_feed.get("items"))
            item_id = f"{random_feed['id']}:{random_item['id']}"
            
            if item_id not in seen_items:
                # Get media items for this feed item
                media_items = random_item.get("media", [])
                image_links = [media["url"] for media in media_items 
                              if media.get("type", "").lower() in ["image", "img", "thumbnail"]]
                
                random_feeds.append({
                    "feed": {
                        "id": random_feed["id"],
                        "name": random_feed["name"],
                        "category": random_feed["category"]
                    },
                    "item": {
                        "id": random_item["id"],
                        "title": random_item.get("title", ""),
                        "link": random_item.get("link", ""),
                        "pub_date": random_item.get("pub_date", ""),
                        "description": random_item.get("description", "")
                    },
                    "media": {
                        "images": image_links,
                        "all_media": media_items
                    },
                    "relevance_score": 0.0,  # No relevance score for random items
                    "matched_content": random_item.get("description", "") or random_item.get("title", "")
                })
                seen_items.add(item_id)
        
        # Combine interest-based and random results
        combined_results = all_results + random_feeds
        
        # Limit to 30 results maximum
        combined_results = combined_results[:30]
        
        if not combined_results:
            raise HTTPException(
                status_code=404,
                detail="No relevant feed items found for your interests"
            )
        
        return {
            "status": "success",
            "interests": user_interests,
            "results_count": len(combined_results),
            "results": combined_results
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@router.post("/similar-interests")
async def get_similar_interests():
    """Generate similar interests based on existing user interests"""
    try:
        # Load existing interests
        user_interests = load_interests()
        
        if not user_interests:
            raise HTTPException(
                status_code=400,
                detail="No interests found. Please add interests first."
            )
        
        # Format interests for the prompt
        interests_text = ", ".join(user_interests)
        
        # Create system message with instructions
        system_message = f"""
        You are an interest recommendation system. Based on the user's existing interests, 
        suggest 5 additional related interests they might enjoy. Each interest should be 
        a short phrase (1-3 words). Provide a brief explanation of why you selected these interests.
        
        Format your response as a JSON object with the following structure:
        {{
            "similar_interests": ["interest1", "interest2", "interest3", "interest4", "interest5"],
            "reasoning": "Brief explanation of why these interests were selected"
        }}
        """
        
        # Create user message with the interests
        user_message = f"My current interests are: {interests_text}. Please suggest similar interests I might enjoy."
        
        # Use Groq to generate similar interests without tool calling
        result = groq_agent.chat(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        try:
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                try:
                    parsed_result = json.loads(json_match.group(0))
                    similar_interests = parsed_result.get("similar_interests", [])
                    reasoning = parsed_result.get("reasoning", "")
                except:
                    lines = result.strip().split('\n')
                    similar_interests = [line.strip() for line in lines if line.strip() and not line.startswith('{') and not line.startswith('}')][:5]
                    reasoning = "Generated based on your existing interests."
            else:
                lines = result.strip().split('\n')
                similar_interests = [line.strip() for line in lines if line.strip() and not line.startswith('{') and not line.startswith('}')][:5]
                reasoning = "Generated based on your existing interests."
            
            similar_interests = similar_interests[:5]
            while len(similar_interests) < 5:
                similar_interests.append(f"Interest {len(similar_interests) + 1}")
            
            return {
                "status": "success",
                "original_interests": user_interests,
                "similar_interests": similar_interests,
            }
        except Exception as parsing_error:
            # Fallback response if parsing fails
            return {
                "status": "partial_success",
                "original_interests": user_interests,
                "similar_interests": ["Technology", "Science", "Arts", "Sports", "Travel"],
                "error": str(parsing_error),
                "raw_response": str(result)
            }
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error generating similar interests: {str(e)}")