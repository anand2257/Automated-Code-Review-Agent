from pydantic import BaseModel
from typing import List, Dict, Any, Tuple

# --- 1. OpenEnv Typed Models ---
class Observation(BaseModel):
    pull_requests: List[Dict[str, str]]
    prs_left: int

class Action(BaseModel):
    pr_id: str
    decision: str # Must be "APPROVE", "REJECT_BUG", or "REJECT_SECURITY"

class Reward(BaseModel):
    score: float

# --- 2. The Environment Class ---
class CodeReviewEnv:
    def __init__(self, task_level: str = "easy"):
        self.task_level = task_level
        self.dataset = self._load_task(task_level)
        self.current_state = []
        self.correct_answers = {}
        # total_reward must be defined in __init__ to avoid AttributeErrors
        self.total_reward = 0.0 
        self.steps_taken = 0
        self.reset()

    def _load_task(self, level: str):
        # 3 Tasks with increasing difficulty
        tasks = {
            "easy": [
                {"id": "pr1", "code": 'print("Hello World)', "true_label": "REJECT_BUG"} # Missing closing quote
            ],
            "medium": [
                {"id": "m1", "code": 'def add(a, b):\n    return a - b', "true_label": "REJECT_BUG"}, # Logic error
                {"id": "m2", "code": 'def greet(name):\n    return f"Hello {name}"', "true_label": "APPROVE"}
            ],
            "hard": [
                {"id": "h1", "code": 'query = "SELECT * FROM users WHERE id = " + user_input\nexecute(query)', "true_label": "REJECT_SECURITY"}, # SQL Injection
                {"id": "h2", "code": 'import os\nos.system("rm -rf /" + user_dir)', "true_label": "REJECT_SECURITY"}, # Command Injection
                {"id": "h3", "code": 'def is_even(num):\n    return num % 2 == 0', "true_label": "APPROVE"}
            ]
        }
        return tasks.get(level, tasks["easy"])

    def reset(self) -> Observation:
        self.current_state = [{"id": d["id"], "code": d["code"]} for d in self.dataset]
        self.correct_answers = {d["id"]: d["true_label"] for d in self.dataset}
        self.total_reward = 0.0
        self.steps_taken = 0
        return self.state()

    def state(self) -> Observation:
        return Observation(pull_requests=self.current_state, prs_left=len(self.current_state))

    def step(self, action: Action):
        reward_value = 0.0
        
        # Grading logic
        if self.task_level == "easy":
            if action.pr_id == "pr1" and action.decision == "REJECT_BUG":
                reward_value = 1.0
        
        elif self.task_level == "medium":
            if action.pr_id == "m1" and action.decision == "REJECT_BUG":
                reward_value = 0.5
            elif action.pr_id == "m2" and action.decision == "APPROVE":
                reward_value = 0.5
                
        elif self.task_level == "hard":
            if action.pr_id in ["h1", "h2"] and action.decision == "REJECT_SECURITY":
                reward_value = 0.333
            elif action.pr_id == "h3" and action.decision == "APPROVE":
                reward_value = 0.334

        self.total_reward += reward_value
        
        # Remove the reviewed PR from state
        self.current_state = [pr for pr in self.current_state if pr["id"] != action.pr_id]
        done = len(self.current_state) == 0
        
        # --- PHASE 2 FIX: STRICTLY BETWEEN 0 AND 1 ---
        # We use 0.95 and 0.05 to ensure we are never exactly 0 or 1
        final_score = self.total_reward
        if final_score >= 1.0:
            final_score = 0.95
        elif final_score <= 0.0:
            final_score = 0.05
            
        reward = Reward(score=final_score)
        
        return self.state(), reward, done, {"total_score": final_score}
