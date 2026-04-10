from pydantic import BaseModel
from typing import List, Dict, Any
import random

# --- 1. OpenEnv Typed Models ---
class Observation(BaseModel):
    pull_requests: List[Dict[str, str]]
    prs_left: int
    system_load: float       # New metadata
    queue_priority: str      # New metadata

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
        self.total_prs = len(self.dataset)
        self.current_state = []
        self.correct_answers = {}
        self.total_reward = 0.0 
        self.steps_taken = 0
        self.reset()

    def _load_task(self, level: str):
        # Expanded dataset with realistic vulnerabilities and clean code noise
        tasks = {
            "easy": [
                {"id": "e1", "code": 'print("Hello World)', "true_label": "REJECT_BUG", "desc": "Syntax Error"},
                {"id": "e2", "code": 'x = [1, 2, 3]\nprint(x[5])', "true_label": "REJECT_BUG", "desc": "Index Out of Bounds"},
                {"id": "e3", "code": 'def add(a, b):\n    return a + b', "true_label": "APPROVE", "desc": "Clean Code"}
            ],
            "medium": [
                {"id": "m1", "code": 'def factorial(n):\n    return n * factorial(n)', "true_label": "REJECT_BUG", "desc": "Infinite Recursion"},
                {"id": "m2", "code": 'import time\ndef wait():\n    time.sleep("10")', "true_label": "REJECT_BUG", "desc": "Type Error"},
                {"id": "m3", "code": 'API_KEY = "12345-ABCDE-67890"', "true_label": "REJECT_SECURITY", "desc": "Hardcoded Credential"},
                {"id": "m4", "code": 'class User:\n    def __init__(self, name):\n        self.name = name', "true_label": "APPROVE", "desc": "Clean Class"}
            ],
            "hard": [
                {"id": "h1", "code": 'eval(user_input)', "true_label": "REJECT_SECURITY", "desc": "Arbitrary Code Execution"},
                {"id": "h2", "code": 'import hashlib\ndef hash_pw(pw):\n    return hashlib.md5(pw.encode()).hexdigest()', "true_label": "REJECT_SECURITY", "desc": "Weak Cryptography"},
                {"id": "h3", "code": 'def get_user(id):\n    return db.execute(f"SELECT * FROM users WHERE id={id}")', "true_label": "REJECT_SECURITY", "desc": "SQL Injection"},
                {"id": "h4", "code": 'def process_list(items):\n    return sorted([i for i in items if i > 0])', "true_label": "APPROVE", "desc": "Clean Logic"}
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
        return Observation(
            pull_requests=self.current_state, 
            prs_left=len(self.current_state),
            system_load=round(random.uniform(0.1, 0.9), 2),
            queue_priority=self.task_level.upper()
        )

    def step(self, action: Action):
        reward_value = 0.0
        step_info = {"status": "processed"}
        
        # Pro-Level Dynamic Grading Logic
        if action.pr_id in self.correct_answers:
            expected = self.correct_answers[action.pr_id]
            if action.decision == expected:
                # Proportional reward based on dataset size
                reward_value = 1.0 / self.total_prs
                step_info["feedback"] = f"Correct! PR {action.pr_id} properly handled."
            else:
                step_info["feedback"] = f"Incorrect. Agent chose {action.decision}, but expected {expected}."
                step_info["error_flag"] = True

        self.total_reward += reward_value
        self.steps_taken += 1
        
        # Remove the reviewed PR from state
        self.current_state = [pr for pr in self.current_state if pr["id"] != action.pr_id]
        done = len(self.current_state) == 0
        
        # --- PHASE 2 FIX: STRICTLY BETWEEN 0 AND 1 ---
        final_score = self.total_reward
        if final_score >= 1.0:
            final_score = 0.95
        elif final_score <= 0.0:
            final_score = 0.05
            
        reward = Reward(score=final_score)
        step_info["total_score"] = final_score
        
        return self.state(), reward, done, step_info
