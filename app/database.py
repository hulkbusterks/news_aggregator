import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FEEDS_FILE = DATA_DIR / "feeds.json"

class FeedDatabase:
    @staticmethod
    def _load_feeds() -> Dict[str, Any]:
        """Load all feeds from the JSON file"""
        if not FEEDS_FILE.exists():
            return {"feeds": {}}
        
        with open(FEEDS_FILE, "r") as f:
            return json.load(f)
    
    @staticmethod
    def _save_feeds(feeds: Dict[str, Any]) -> None:
        """Save all feeds to the JSON file"""
        with open(FEEDS_FILE, "w") as f:
            json.dump(feeds, f, indent=2)
    
    @staticmethod
    def save_feed(feed_id: str, name: str, url: str, category: str, items: List[Dict[str, Any]]) -> None:
        """Save or update a feed in the feeds file"""
        feeds_data = FeedDatabase._load_feeds()
        
        # Add unique IDs to items if they don't have one
        for item in items:
            if "id" not in item:
                item["id"] = str(uuid4())
        
        feed_data = {
            "id": feed_id,
            "name": name,
            "url": url,
            "category": category,
            "last_updated": datetime.now().isoformat(),
            "items": items
        }
        
        feeds_data["feeds"][feed_id] = feed_data
        FeedDatabase._save_feeds(feeds_data)
    
    @staticmethod
    def get_feed(feed_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific feed"""
        feeds_data = FeedDatabase._load_feeds()
        return feeds_data["feeds"].get(feed_id)
    
    @staticmethod
    def list_feeds() -> List[Dict[str, Any]]:
        """List all feeds"""
        feeds_data = FeedDatabase._load_feeds()
        return list(feeds_data["feeds"].values())
    
    @staticmethod
    def get_feeds_by_category(category: str) -> List[Dict[str, Any]]:
        """Get all feeds in a specific category"""
        feeds_data = FeedDatabase._load_feeds()
        return [
            feed for feed in feeds_data["feeds"].values()
            if feed["category"].lower() == category.lower()
        ]
    
    @staticmethod
    def delete_feed(feed_id: str) -> bool:
        """Delete a feed"""
        feeds_data = FeedDatabase._load_feeds()
        if feed_id not in feeds_data["feeds"]:
            return False
        
        del feeds_data["feeds"][feed_id]
        FeedDatabase._save_feeds(feeds_data)
        return True
    
    @staticmethod
    def get_feed_items(feed_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get items for a specific feed"""
        feed = FeedDatabase.get_feed(feed_id)
        return feed["items"] if feed else None 