from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from API import forcast_controller, cluster_controller, churn_controller, sentiment_controller,chat_controller

# Initialize FastAPI app
app = FastAPI(
    title="CORTEX AI SYSTEM",
    description="API for profit forecasting, customer segmentation, churn prediction, and sentiment analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(forcast_controller.router)
app.include_router(churn_controller.router)
app.include_router(cluster_controller.router)
app.include_router(sentiment_controller.router)
app.include_router(chat_controller.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Business Analytics ML API"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)