from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .router import router
import uvicorn

app = FastAPI(
    title="News Feed API",
    description="API for fetching, parsing and storing XML feeds as JSON",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to the News Feed API! Go to /docs for documentation."}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 