from groq import Groq
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import feedparser
from typing import List, Dict
import json
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer, util
from groq_model import GroqAgent

DATABASE_PATH = "blog_data.db"
FAISS_INDEX_PATH = "faiss_index"
INDEX_NAME = "combined_embeddings"
SENTENCE_SIMILARITY_MODEL = 'all-MiniLM-L6-v2' 

def load_and_process_rss_feed(rss_url: str) -> List[Dict]:
    """Loads and processes an RSS feed, extracting content and metadata."""
    try:
        feed = feedparser.parse(rss_url)
        if feed.get('entries'):
            documents = []
            for entry in feed.entries:
                content = entry.get('summary') or entry.get('content', [{}])[0].get('value')
                if content:
                    metadata = {'title': entry.get('title'), 'link': entry.get('link')}
                    documents.append({'page_content': content, 'metadata': metadata})
            return documents
        else:
            print(f"No entries found in the RSS feed: {rss_url}")
            return []
    except Exception as e:
        print(f"Error parsing RSS feed '{rss_url}': {e}")
        return []

def initialize_sqlite_database():
    """Initializes the SQLite database and table if they don't exist."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blog_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            link TEXT UNIQUE,
            content TEXT
        );
    """)
    conn.commit()
    conn.close()

def reset_sqlite_database():
    """Resets the SQLite database by deleting all data from the blog_data table."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM blog_data")
    conn.commit()
    conn.close()
    print("SQLite database reset.")

def reset_faiss_index():
    """Resets the FAISS index by deleting the saved index files."""
    import shutil
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
        print(f"FAISS index files deleted from {FAISS_INDEX_PATH}")
    else:
        print(f"FAISS index directory {FAISS_INDEX_PATH} does not exist.")


def add_documents_to_store(documents: List[Dict]):
    """Adds documents and metadata to SQLite."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    added_count = 0
    for doc in documents:
        try:
            cursor.execute(
                "INSERT INTO blog_data (title, link, content) VALUES (?, ?, ?)",
                (doc['metadata'].get('title'), doc['metadata'].get('link'), doc['page_content']),
            )
            conn.commit()
            added_count += 1
        except sqlite3.IntegrityError:
            pass
    conn.close()
    print(f"Added {added_count} new entries to SQLite.")



def create_faiss_index(documents: List[Dict], embedding_function) -> FAISS:
    """Creates a FAISS vector store from a list of documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.create_documents([doc['page_content'] for doc in documents],
                                           metadatas=[doc['metadata'] for doc in documents])
    embeddings = embedding_function.embed_documents([t.page_content for t in texts])
    docsearch = FAISS.from_texts(texts=[t.page_content for t in texts], embedding=embedding_function, metadatas=[doc['metadata'] for doc in documents])
    return docsearch

def load_faiss_index(embedding_function):
    """Loads the FAISS index from disk."""
    try:
        docsearch = FAISS.load_local(FAISS_INDEX_PATH, embedding_function, index_name=INDEX_NAME)
        print(f"FAISS index loaded from {FAISS_INDEX_PATH}")
        return docsearch
    except Exception as e:
        print(f"Error loading FAISS index from {FAISS_INDEX_PATH}: {e}")
        return None


def save_faiss_index(docsearch: FAISS):
    """Saves the FAISS index to disk."""
    if docsearch:
        docsearch.save_local(FAISS_INDEX_PATH, index_name=INDEX_NAME)
        print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    else:
        print("No FAISS index to save.")

def get_relevant_blog_details(user_preferences: str, docsearch: FAISS, top_k: int = 5) -> List[Dict]:
    """Retrieves relevant blog details using FAISS for search and SQLite for details."""
    if docsearch is None:
        print("Error: FAISS index is not initialized.")
        return []

    relevant_docs = docsearch.similarity_search(user_preferences, k=top_k)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    relevant_blogs = []
    sentence_model = SentenceTransformer(SENTENCE_SIMILARITY_MODEL)
    query_embedding = sentence_model.encode(user_preferences)

    for doc in relevant_docs:
        link = doc.metadata.get('link')
        if link:
            cursor.execute("SELECT title, link, content FROM blog_data WHERE link = ?", (link,))
            result = cursor.fetchone()
            if result:
                title, link, content = result
                sentences = content.split(".")
                sentence_embeddings = sentence_model.encode(sentences)
                cosine_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
                most_relevant_sentence_indices = np.argsort(-cosine_scores)[:2]
                relevant_sentences = [sentences[i].strip() for i in most_relevant_sentence_indices if
                                      cosine_scores[i] > 0.3]
                relevant_blogs.append({"title": title, "link": link, "relevant_sentences": relevant_sentences})
            else:
                print(f"Warning: Could not find blog with link '{link}' in SQLite.")
    conn.close()
    return relevant_blogs


def blog_recommendation_with_preferences_json(client: Groq, model: str, user_preferences: str,
                                             relevant_blogs: List[Dict]) -> Dict:
    """Generates a JSON response recommending relevant blogs using tool output."""
    system_prompt = """
    You are a blog recommendation system. You have access to a tool that can retrieve relevant blog details based on user preferences.
    Use this tool to find relevant blogs and then respond to the user in a JSON format.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_relevant_blog_details",
                "description": "Retrieves relevant blog details based on user preferences.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_preferences": {"type": "string", "description": "The user's interests."},
                    },
                    "required": ["user_preferences"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": user_preferences}]
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "get_relevant_blog_details"}},  # Force the tool call
        )
        if hasattr(chat_completion, 'tool_calls') and chat_completion.tool_calls:
            tool_calls = chat_completion.tool_calls
            function_name = tool_calls[0].function.name
            function_args_str = tool_calls[0].function.arguments
            function_args = json.loads(function_args_str)
            if function_name == "get_relevant_blog_details":
                tool_output = {"relevant_blogs": relevant_blogs}
                second_response = client.chat.completions.create(
                    messages=messages + [
                        tool_calls[0],
                        {"role": "tool", "tool_call_id": tool_calls[0].id, "content": json.dumps(tool_output)}
                    ],
                    model=model,
                    response_format={"type": "json_object"}
                )
                return json.loads(second_response.choices[0].message.content)
        else:
            return {"relevant_blogs": relevant_blogs}
    except Exception as e:
        print(f"Error in blog_recommendation_with_preferences_json: {e}")
        return {"error": f"An error occurred: {e}"}


def main():
    agent = GroqAgent()
    initialize_sqlite_database()
    docsearch = None 
    sentence_similarity_model = SentenceTransformer(
        SENTENCE_SIMILARITY_MODEL)  
    print("Multi-Feed Blog Recommendation System (FAISS for Search, SQLite for Data)")
    all_documents = [] 
    while True:
        action = input(
            "Enter action ('add' for RSS feed, 'query' for search, 'reset' to clear database, 'faiss', 'q' to quit): ").strip().lower() #added faiss option
        if action == 'q':
            break
        if action == 'reset':
            reset_sqlite_database()
            reset_faiss_index() 
            docsearch = None
            all_documents = [] 
            continue
        if action == 'add':
            rss_url_input = input("Enter RSS feed URL: ").strip()
            new_documents = load_and_process_rss_feed(rss_url_input)
            if new_documents:
                add_documents_to_store(new_documents)
                all_documents.extend(new_documents) 
                print(f"Added {len(new_documents)} documents.  Total documents in queue: {len(all_documents)}")
        elif action == 'faiss':
            if not all_documents:
                print("No documents available to create FAISS index. Please add some RSS feeds first.")
            elif docsearch is None:
                print("Creating FAISS index from accumulated data...")
                docsearch = create_faiss_index(all_documents, embedding_function)
                save_faiss_index(docsearch)
                print("FAISS index created and saved.")
            else:
                print("FAISS index already exists.")

        if action == 'query':
            user_preferences = input("Ask a question about the blogs: ").strip()
            if user_preferences:
                if docsearch is None and all_documents:
                    print("Creating FAISS index...")
                    docsearch = create_faiss_index(all_documents, embedding_function)
                    save_faiss_index(docsearch)
                elif docsearch is None:
                    print("No documents added yet or FAISS index not created. Please add some RSS feeds and create the index.")
                    continue 
                relevant_blogs = get_relevant_blog_details(user_preferences, docsearch)
                response_json = blog_recommendation_with_preferences_json(agent.client, agent.model,
                                                                           user_preferences, relevant_blogs)
                print("\nRecommended Blogs (JSON):\n")
                print(json.dumps(response_json, indent=2))
                print("\n------------------------------------\n")
    if docsearch:
        save_faiss_index(docsearch) 
    print("Exiting.")

if __name__ == "__main__":
    main()
