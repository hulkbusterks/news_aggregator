import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4
from contextlib import contextmanager

# Database configuration
DB_PATH = "feeds.db"

class FeedDatabase:
    @staticmethod
    @contextmanager
    def get_connection():
        """Context manager for database connections"""
        conn = sqlite3.connect(DB_PATH)
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def initialize_database():
        """Initialize the SQLite database and create necessary tables"""
        with FeedDatabase.get_connection() as conn:
            # Create feeds table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feeds (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                );
            """)
            
            # Create feed items table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feed_items (
                    id TEXT PRIMARY KEY,
                    feed_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    link TEXT UNIQUE NOT NULL,
                    description TEXT,
                    pub_date TEXT,
                    FOREIGN KEY (feed_id) REFERENCES feeds(id) ON DELETE CASCADE
                );
            """)
            
            # Create table for media items
            conn.execute("""
                CREATE TABLE IF NOT EXISTS media_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    url TEXT NOT NULL,
                    width TEXT,
                    height TEXT,
                    medium TEXT,
                    description TEXT,
                    length TEXT,
                    FOREIGN KEY (item_id) REFERENCES feed_items(id) ON DELETE CASCADE
                );
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feed_items_feed_id ON feed_items(feed_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_items_item_id ON media_items(item_id);")
            conn.commit()

    @staticmethod
    def save_feed(feed_id: str, name: str, url: str, category: str, items: List[Dict[str, Any]]) -> None:
        """Save or update a feed with its items"""
        with FeedDatabase.get_connection() as conn:
            cursor = conn.cursor()
            current_time = datetime.now().isoformat()
            
            try:
                # Check if feed exists
                cursor.execute("SELECT id FROM feeds WHERE id = ?", (feed_id,))
                feed_exists = cursor.fetchone() is not None
                
                if feed_exists:
                    # Update existing feed
                    cursor.execute(
                        "UPDATE feeds SET name = ?, url = ?, category = ?, last_updated = ? WHERE id = ?",
                        (name, url, category, current_time, feed_id)
                    )
                else:
                    # Insert new feed
                    cursor.execute(
                        "INSERT INTO feeds (id, name, url, category, last_updated) VALUES (?, ?, ?, ?, ?)",
                        (feed_id, name, url, category, current_time)
                    )
                
                # Process items
                for item in items:
                    item_id = item.get('id') or str(uuid4())
                    
                    # Check if item exists
                    cursor.execute("SELECT id FROM feed_items WHERE id = ?", (item_id,))
                    item_exists = cursor.fetchone() is not None
                    
                    if item_exists:
                        # Update existing item
                        cursor.execute(
                            """
                            UPDATE feed_items 
                            SET feed_id = ?, title = ?, link = ?, description = ?, pub_date = ?
                            WHERE id = ?
                            """,
                            (
                                feed_id, 
                                item.get('title', 'No Title'), 
                                item.get('link', ''), 
                                item.get('description', ''),
                                item.get('pub_date', ''), 
                                item_id
                            )
                        )
                        
                        # Delete existing media items for this feed item
                        cursor.execute("DELETE FROM media_items WHERE item_id = ?", (item_id,))
                    else:
                        # Insert new item
                        cursor.execute(
                            """
                            INSERT INTO feed_items 
                            (id, feed_id, title, link, description, pub_date) 
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                item_id, 
                                feed_id, 
                                item.get('title', 'No Title'), 
                                item.get('link', ''), 
                                item.get('description', ''),
                                item.get('pub_date', '')
                            )
                        )
                    
                    # Insert media items
                    for media in item.get('media', []):
                        cursor.execute(
                            """
                            INSERT INTO media_items 
                            (item_id, type, url, width, height, medium, description, length) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                item_id,
                                media.get('type', ''),
                                media.get('url', ''),
                                media.get('width'),
                                media.get('height'),
                                media.get('medium'),
                                media.get('description'),
                                media.get('length')
                            )
                        )
                
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e

    @staticmethod
    def get_feed(feed_id: str) -> Optional[Dict[str, Any]]:
        """Get a feed by ID with all its items"""
        with FeedDatabase.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get feed data
            cursor.execute("SELECT * FROM feeds WHERE id = ?", (feed_id,))
            feed_row = cursor.fetchone()
            
            if not feed_row:
                return None
            
            # Get feed items
            cursor.execute("""
                SELECT * FROM feed_items 
                WHERE feed_id = ? 
                ORDER BY pub_date DESC
            """, (feed_id,))
            items = []
            
            for item_row in cursor.fetchall():
                item = dict(item_row)
                
                # Get media items for this feed item
                cursor.execute("""
                    SELECT * FROM media_items 
                    WHERE item_id = ?
                """, (item['id'],))
                
                item['media'] = [dict(media_row) for media_row in cursor.fetchall()]
                items.append(item)
            
            feed = dict(feed_row)
            feed['items'] = items
            return feed

    @staticmethod
    def list_feeds() -> List[Dict[str, Any]]:
        """List all feeds"""
        with FeedDatabase.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM feeds")
            feed_ids = [row['id'] for row in cursor.fetchall()]
            
            return [FeedDatabase.get_feed(feed_id) for feed_id in feed_ids]

    @staticmethod
    def get_feeds_by_category(category: str) -> List[Dict[str, Any]]:
        """Get all feeds in a specific category"""
        with FeedDatabase.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM feeds WHERE category = ?", (category,))
            feed_ids = [row['id'] for row in cursor.fetchall()]
            
            return [FeedDatabase.get_feed(feed_id) for feed_id in feed_ids]

    @staticmethod
    def delete_feed(feed_id: str) -> bool:
        """Delete a feed"""
        with FeedDatabase.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if feed exists
            cursor.execute("SELECT id FROM feeds WHERE id = ?", (feed_id,))
            if not cursor.fetchone():
                return False
            
            # Delete feed (cascade will handle items and media)
            cursor.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))
            conn.commit()
            return True

    @staticmethod
    def get_feed_items(feed_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get items for a specific feed"""
        feed = FeedDatabase.get_feed(feed_id)
        return feed["items"] if feed else None 