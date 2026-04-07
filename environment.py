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
        self.score = 0.0
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
        self.score = 0.0
        self.steps_taken = 0
        return self.state()

    def state(self) -> Observation:
        return Observation(pull_requests=self.current_state, prs_left=len(self.current_state))

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.steps_taken += 1
        reward_val = 0.0
        
        # Grader Logic: Did the AI make the correct review decision?
        if action.pr_id in self.correct_answers:
            if action.decision == self.correct_answers[action.pr_id]:
                reward_val = 1.0 / len(self.dataset) # Partial progress scoring (0.0 to 1.0)
                self.score += reward_val
            
            # Remove processed PR from the queue
            self.current_state = [pr for pr in self.current_state if pr["id"] != action.pr_id]

        # Episode ends when queue is empty or taking too many steps
        done = len(self.current_state) == 0 or self.steps_taken >= len(self.dataset) + 2
        
        return self.state(), Reward(score=reward_val), done, {"total_score": self.score}
