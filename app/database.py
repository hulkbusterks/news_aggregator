import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Create data directory if it doesn't exist
DATA_DIR = Path("data/feeds")
DATA_DIR.mkdir(parents=True, exist_ok=True)

class FeedDatabase:
    @staticmethod
    def save_feed(feed_id: str, name: str, url: str, content: Dict[str, Any]) -> None:
        """Save feed metadata and content to JSON files"""
        # Save metadata
        metadata = {
            "id": feed_id,
            "name": name,
            "url": url,
            "last_updated": datetime.now().isoformat()
        }
        
        metadata_path = DATA_DIR / f"{feed_id}_meta.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save content
        content_path = DATA_DIR / f"{feed_id}_content.json"
        with open(content_path, "w") as f:
            json.dump(content, f, indent=2)
    
    @staticmethod
    def get_feed_metadata(feed_id: str) -> Optional[Dict[str, Any]]:
        """Get feed metadata"""
        metadata_path = DATA_DIR / f"{feed_id}_meta.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    @staticmethod
    def get_feed_content(feed_id: str) -> Optional[Dict[str, Any]]:
        """Get feed content"""
        content_path = DATA_DIR / f"{feed_id}_content.json"
        if not content_path.exists():
            return None
        
        with open(content_path, "r") as f:
            return json.load(f)
    
    @staticmethod
    def list_feeds() -> List[Dict[str, Any]]:
        """List all feeds"""
        feeds = []
        for file in DATA_DIR.glob("*_meta.json"):
            with open(file, "r") as f:
                feeds.append(json.load(f))
        return feeds
    
    @staticmethod
    def delete_feed(feed_id: str) -> bool:
        """Delete a feed"""
        metadata_path = DATA_DIR / f"{feed_id}_meta.json"
        content_path = DATA_DIR / f"{feed_id}_content.json"
        
        if not metadata_path.exists():
            return False
        
        if metadata_path.exists():
            os.remove(metadata_path)
        if content_path.exists():
            os.remove(content_path)
        
        return True 