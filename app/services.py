import requests
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
import uuid
import json
from datetime import datetime

def parse_xml_feed(url: str) -> List[Dict[str, Any]]:
    """Fetch and parse XML feed from URL, returning items with unique IDs"""
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/xml, application/rss+xml, application/atom+xml, application/rdf+xml, text/xml'
        }
        
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        
        # Try to detect content type and handle different encodings
        content_type = response.headers.get('content-type', '').lower()
        if 'charset=' in content_type:
            encoding = content_type.split('charset=')[-1]
            response.encoding = encoding
        
        # Parse XML with error recovery
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            # Try to fix common XML issues
            fixed_xml = response.text.replace('&', '&amp;')
            root = ET.fromstring(fixed_xml)
        
        # Define comprehensive namespaces for different feed types
        namespaces = {
            # RSS namespaces
            'rss': 'http://purl.org/rss/1.0/',
            'rss2': 'http://purl.org/rss/1.0/',
            'content': 'http://purl.org/rss/1.0/modules/content/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'sy': 'http://purl.org/rss/1.0/modules/syndication/',
            'admin': 'http://webns.net/mvcb/',
            'cc': 'http://web.resource.org/cc/',
            
            # Media namespaces
            'media': 'http://search.yahoo.com/mrss/',
            'itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd',
            'geo': 'http://www.w3.org/2003/01/geo/wgs84_pos#',
            
            # Atom namespaces
            'atom': 'http://www.w3.org/2005/Atom',
            'atom2': 'http://www.w3.org/2005/Atom',
            
            # RDF namespaces
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdf2': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            
            # Additional namespaces
            'slash': 'http://purl.org/rss/1.0/modules/slash/',
            'trackback': 'http://madskills.com/public/xml/rss/module/trackback/',
            'wfw': 'http://wellformedweb.org/CommentAPI/',
            'wp': 'http://wordpress.org/export/1.2/',
            'excerpt': 'http://wordpress.org/export/1.2/excerpt/',
            'comments': 'http://purl.org/rss/1.0/modules/comments/'
        }
        
        # Find items/entries - handle multiple feed formats
        items = []
        if root.tag.endswith('RDF'):
            # RDF format (like DW feed)
            items = root.findall(".//rss:item", namespaces=namespaces) or \
                   root.findall(".//item", namespaces=namespaces)
        elif root.tag.endswith('feed'):
            # Atom format
            items = root.findall(".//atom:entry", namespaces=namespaces) or \
                   root.findall(".//entry", namespaces=namespaces)
        else:
            # Standard RSS format
            items = root.findall(".//item") or \
                   root.findall(".//entry") or \
                   root.findall(".//rss:item", namespaces=namespaces)
        
        parsed_items = []
        for item in items:
            try:
                # Extract basic item information with fallbacks
                item_dict = {
                    "id": str(uuid.uuid4()),
                    "title": find_element_text(item, "./rss:title", namespaces=namespaces) or 
                            find_element_text(item, "./title") or 
                            find_element_text(item, "./atom:title", namespaces=namespaces) or "No Title",
                    "link": find_element_text(item, "./rss:link", namespaces=namespaces) or 
                           find_element_text(item, "./link") or 
                           find_element_text(item, "./atom:link", namespaces=namespaces) or "",
                    "description": find_element_text(item, "./rss:description", namespaces=namespaces) or 
                                 find_element_text(item, "./description") or 
                                 find_element_text(item, "./content", namespaces=namespaces) or 
                                 find_element_text(item, "./content:encoded", namespaces=namespaces) or "",
                    "pub_date": find_element_text(item, "./dc:date", namespaces=namespaces) or 
                               find_element_text(item, "./pubDate") or 
                               find_element_text(item, "./published") or 
                               find_element_text(item, "./atom:published", namespaces=namespaces) or "",
                    "media": [],
                    "metadata": {}  # For additional metadata
                }
                
                # Extract media:content elements
                for media in item.findall(".//media:content", namespaces=namespaces):
                    try:
                        media_item = {
                            "type": media.get("type", ""),
                            "url": media.get("url", ""),
                            "width": media.get("width", ""),
                            "height": media.get("height", ""),
                            "medium": media.get("medium", ""),
                            "description": find_element_text(media, "./media:description", namespaces=namespaces) or ""
                        }
                        if media_item["url"]:
                            item_dict["media"].append(media_item)
                    except Exception as e:
                        print(f"Error parsing media content: {str(e)}")
                        continue
                
                # Extract media:thumbnail elements
                for thumbnail in item.findall(".//media:thumbnail", namespaces=namespaces):
                    try:
                        media_item = {
                            "type": "image",
                            "url": thumbnail.get("url", ""),
                            "width": thumbnail.get("width", ""),
                            "height": thumbnail.get("height", ""),
                            "medium": "thumbnail"
                        }
                        if media_item["url"]:
                            item_dict["media"].append(media_item)
                    except Exception as e:
                        print(f"Error parsing thumbnail: {str(e)}")
                        continue
                
                # Extract enclosure elements
                for enclosure in item.findall("./enclosure"):
                    try:
                        media_item = {
                            "type": enclosure.get("type", ""),
                            "url": enclosure.get("url", ""),
                            "length": enclosure.get("length", ""),
                            "medium": "enclosure"
                        }
                        if media_item["url"]:
                            item_dict["media"].append(media_item)
                    except Exception as e:
                        print(f"Error parsing enclosure: {str(e)}")
                        continue
                
                # Extract additional metadata
                try:
                    # Dublin Core metadata
                    item_dict["metadata"]["language"] = find_element_text(item, "./dc:language", namespaces=namespaces)
                    item_dict["metadata"]["category"] = find_element_text(item, "./dc:subject", namespaces=namespaces)
                    item_dict["metadata"]["creator"] = find_element_text(item, "./dc:creator", namespaces=namespaces)
                    
                    # WordPress metadata
                    item_dict["metadata"]["post_id"] = find_element_text(item, "./wp:post_id", namespaces=namespaces)
                    item_dict["metadata"]["status"] = find_element_text(item, "./wp:status", namespaces=namespaces)
                    
                    # Comments and trackbacks
                    item_dict["metadata"]["comments"] = find_element_text(item, "./comments", namespaces=namespaces)
                    item_dict["metadata"]["trackbacks"] = find_element_text(item, "./trackback:ping", namespaces=namespaces)
                    
                    # Geo location
                    item_dict["metadata"]["latitude"] = find_element_text(item, "./geo:lat", namespaces=namespaces)
                    item_dict["metadata"]["longitude"] = find_element_text(item, "./geo:long", namespaces=namespaces)
                except Exception as e:
                    print(f"Error extracting metadata: {str(e)}")
                
                parsed_items.append(item_dict)
                
            except Exception as e:
                print(f"Error parsing item: {str(e)}")
                continue
        
        return parsed_items
    
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch XML: {str(e)}")
    except ET.ParseError as e:
        raise Exception(f"Failed to parse XML: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

def find_element_text(element, xpath, namespaces=None):
    """Helper to find element text or None"""
    try:
        found = element.find(xpath, namespaces=namespaces)
        if found is not None:
            return found.text.strip() if found.text else None
        return None
    except Exception:
        return None

def generate_feed_id(name: str) -> str:
    """Generate a unique ID for a feed based on name"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_name = name.lower().replace(" ", "_")
    return f"{safe_name}_{timestamp}" 

