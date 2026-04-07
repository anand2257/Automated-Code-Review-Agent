import os
import json
from typing import List, Optional
from openai import OpenAI
from environment import CodeReviewEnv, Action

# Setup variables exactly as requested by the new spec
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "dummy_key")
BENCHMARK = "CodeReviewEnv"

# --- Mandatory Logging Functions ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def run_inference():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
        
        env = CodeReviewEnv(task_level=task)
        obs = env.reset()
        
        done = False
        step_count = 0
        rewards: List[float] = []
        error = None
        
        while not done:
            step_count += 1
            error = None
            
            # Request the action from the LLM
            prompt = f"Review the first code snippet in this queue: {obs.pull_requests}. Reply ONLY with JSON format: {{\"pr_id\": \"id_here\", \"decision\": \"APPROVE, REJECT_BUG, or REJECT_SECURITY\"}}"
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={ "type": "json_object" }
                )
                action_data = json.loads(response.choices[0].message.content)
                action = Action(**action_data)
                action_str = action.decision
            except Exception as e:
                # Fallback logic if LLM formats incorrectly
                error = "json_parse_error"
                if len(obs.pull_requests) > 0:
                     action = Action(pr_id=obs.pull_requests[0]["id"], decision="APPROVE")
                     action_str = "APPROVE"
                else:
                     break
                     
            # Execute step
            obs, reward, done, info = env.step(action)
            
            # Store and format reward
            reward_val = reward.score
            rewards.append(reward_val)
            
            log_step(step=step_count, action=action_str, reward=reward_val, done=done, error=error)

        # Calculate final metrics
        final_score = info.get('total_score', 0.0)
        final_score = min(max(final_score, 0.0), 1.0) # Clamp between 0 and 1
        success = final_score > 0.0
        
        log_end(success=success, steps=step_count, score=final_score, rewards=rewards)

if __name__ == "__main__":
    run_inference()
