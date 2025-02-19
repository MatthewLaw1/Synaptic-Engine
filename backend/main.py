"""
Main entry point for the EEG thought-to-video pipeline.
"""

import os
from ml import storage_manager
from lumaAPI import create_video
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/latest-thought")
async def get_latest_thought():
    """Get the latest thought and its interpretation."""
    try:
        thought_data = storage_manager.get_latest_thought()
        if not thought_data or not thought_data.get('thought'):
            raise HTTPException(status_code=404, detail="No thought data available")
        return thought_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-video")
async def generate_video():
    """Generate a video from the latest thought."""
    try:
        # Get latest thought from storage
        thought_data = storage_manager.get_latest_thought()
        if not thought_data or not thought_data.get('thought'):
            raise HTTPException(status_code=404, detail="No thought data available")
            
        # Create video from thought
        thought = thought_data['thought']
        print(f"\nCreating video for thought: {thought}")
        video_url = create_video(thought)
        
        if not video_url:
            raise HTTPException(status_code=500, detail="Failed to create video")
            
        return {"video_url": video_url}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point for CLI usage."""
    try:
        # Get latest thought from storage
        thought_data = storage_manager.get_latest_thought()
        if not thought_data or not thought_data.get('thought'):
            print("No thought data available")
            return
            
        # Create video from thought
        thought = thought_data['thought']
        print(f"\nCreating video for thought: {thought}")
        video_url = create_video(thought)
        
        if video_url:
            print("\nSuccess! Video is available at the URL above.")
        else:
            print("\nFailed to create video.")
            
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    import uvicorn
    # If run directly, start the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
