import requests
import xml.etree.ElementTree as ET
from typing import Dict, Any
import uuid
import json
from datetime import datetime

def parse_xml_feed(url: str) -> Dict[str, Any]:
    """Fetch and parse XML feed from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.text)
        
        # Convert XML to dictionary
        # This is a simplified version - you may need to adapt
        # this based on the specific XML structure of your feeds
        result = {
            "title": find_element_text(root, ".//title") or "Unknown",
            "description": find_element_text(root, ".//description") or "",
            "items": []
        }
        
        # Find items/entries
        items = root.findall(".//item") or root.findall(".//entry")
        
        for item in items:
            item_dict = {
                "title": find_element_text(item, "./title") or "No Title",
                "link": find_element_text(item, "./link") or "",
                "description": find_element_text(item, "./description") or 
                               find_element_text(item, "./content") or "",
                "pub_date": find_element_text(item, "./pubDate") or 
                            find_element_text(item, "./published") or "",
            }
            result["items"].append(item_dict)
        
        return result
    
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch XML: {str(e)}")
    except ET.ParseError as e:
        raise Exception(f"Failed to parse XML: {str(e)}")

def find_element_text(element, xpath):
    """Helper to find element text or None"""
    found = element.find(xpath)
    return found.text if found is not None else None

def generate_feed_id(name: str) -> str:
    """Generate a unique ID for a feed based on name"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_name = name.lower().replace(" ", "_")
    return f"{safe_name}_{timestamp}" 