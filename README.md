![LensNews Logo](logo.png)

# 🚀 LensNews
Personalized news aggregator powered by AI to deliver content that matters to you.

## 📌 Problem Statement
Problem Statement 1 - Weave AI magic with Groq

## 🎯 Objective
LensNews solves the problem of information overload by providing a personalized news experience. Users can define their interests and receive tailored news recommendations, search across multiple sources with a single query, and discover related topics through AI-generated suggestions. This serves anyone who wants to stay informed without being overwhelmed by irrelevant content.

## 🧠 Team & Approach
**Team Name:** Hack Overflow

**Team Members:**
- Kritik Sharma (GitHub: [hulkbusterks](https://github.com/hulkbusterks) / LinkdIn: https://www.linkedin.com/in/kritik-sharma-qwerty /Role: Backend Developer)
- Honey Bansal (GitHub: [honeybansal](https://github.com/honeybansal2968) / LinkdIn: https://www.linkedin.com/in/honey-bansal-430a46194 /Role: Frontend Developer)
- Mohit Raj Sinha (GitHub: [mohitrajsinha](https://github.com/mohitrajsinha)/LinkdIn: https://www.linkedin.com/in/mohit-raj-sinha /Role: Frontend Developer)
- Kritik Sharma (GitHub: [VanshGupta1905](https://github.com/VanshGupta1905) / LinkdIn: https://www.linkedin.com/in/vansh-gupta-24128b188 /Role: Backend Developer)

**Our Approach:**
We chose this problem because information overload is a growing challenge in the digital age. Key challenges we addressed include implementing efficient vector search across thousands of articles and creating an AI system that could suggest relevant interests based on user preferences. Our breakthrough came when we integrated Groq's ultra-fast LLM API for generating personalized interest recommendations.

## 🛠️ Tech Stack
**Core Technologies Used:**
- **Backend:** FastAPI, Python, Uvicorn
- **Search:** Vector Search with FAISS, Sentence Transformers
- **Database:** File-based storage for user interests
- **APIs:** RSS feeds, Groq API
- **Hosting:** Netlify

**Sponsor Technologies Used:**
- ✅ **Groq:** Used for generating similar interest recommendations with ultra-low latency

## ✨ Key Features
- ✅ **Personalized Feed:** Define your interests and receive tailored news recommendations
- ✅ **Unified Search:** Search across multiple news sources with a single query
- ✅ **Interest Discovery:** Discover related topics through AI-generated interest suggestions
- ✅ **Clean Reading Experience:** Access a distraction-free reading interface
- ✅ **Vector Search:** Find semantically relevant articles, not just keyword matches

## 📽️ Demo & Deliverables
**Live Demo:** [https://lensnews.netlify.app/](https://lensnews.netlify.app/)

**Demo Video Link:** [Coming Soon]

**Pitch Deck Link:** [Coming Soon]

## 🧪 How to Run the Project
**Requirements:**
- Python 3.11 or higher
- Groq API Key
- Docker (optional)

**Local Setup:**
```bash
# Clone the repo
git clone https://github.com/kritiksharma/news_aggregator
cd news_aggregator

# Install dependencies
pip install -r requirements.txt

# Set environment variables
# For Windows
set GROQ_API_KEY=your_api_key_here

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 7860
 ```

Docker Setup:

```bash
# Build the Docker image
docker build -t news_aggregator .

# Run the container
docker run -p 7860:7860 -e GROQ_API_KEY=your_api_key_here news_aggregator
```

## 🧬 Future Scope
- 📈 User Profiles: Save preferences across sessions
- 🛡️ Content Filtering: Add options to filter out specific topics or sources
- 🌐 Multi-language Support: Expand to support news in multiple languages
- 📊 Analytics Dashboard: Provide insights on reading habits and interests
- 🔄 Real-time Updates: Implement real-time news updates for breaking stories

## 📎 Resources / Credits
- Powered by Groq's ultra-fast LLM API
- Built with FastAPI and Python
- Vector search implemented with FAISS and Sentence Transformers

## 🏁 Final Words
Building LensNews during this hackathon was an exciting journey of exploring how AI can transform the way we consume information. The biggest challenge was creating a system that truly understands user interests and delivers relevant content. We learned that combining vector search with LLM-powered recommendations creates a powerful synergy for personalized content discovery. We're excited to continue developing this platform to help people stay informed in an increasingly complex information landscape.
