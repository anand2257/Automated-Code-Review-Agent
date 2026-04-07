import uvicorn
from fastapi import FastAPI
from environment import CodeReviewEnv, Action

app = FastAPI()
env = CodeReviewEnv()

@app.get("/")
def ping():
    return {"status": "200 OK"}

@app.get("/state")
def state():
    return env.state().model_dump()

@app.get("/reset")
@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

# --- MANDATORY MAIN FUNCTION FOR VALIDATOR ---
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
