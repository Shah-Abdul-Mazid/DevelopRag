from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "success"}

@app.get("/deploy")
async def deploy():
    return {"message": "Render Deployment"}
