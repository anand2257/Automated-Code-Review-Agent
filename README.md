# Automated Code Review Agent - OpenEnv

This repository contains an official submission for the OpenEnv Hackathon (Round 1). 

The **Automated Code Review Agent** is a fully containerized, real-world Reinforcement Learning environment built to simulate a Senior Developer triaging a Pull Request (PR) queue. The environment evaluates an agent's ability to classify code snippets by identifying clean code, functional bugs, and critical security vulnerabilities.

## 🌍 Real-World Utility
Automated code review is a high-priority enterprise problem. Rather than a "toy" game, this environment models a genuine CI/CD pipeline task. Agents trained in this environment must learn to parse syntax, understand logic, and detect malicious injections, triaging them appropriately to save human reviewer time.

## ⚙️ Environment Specification

### Observation Space
The state is represented as a JSON object containing the current queue of pending pull requests and the total count remaining.
* `pull_requests`: List of dictionaries containing `id` and `code` strings.
* `prs_left`: Integer representing the remaining queue size.

### Action Space
The agent must select a PR from the queue and make a moderation decision.
* `pr_id`: The ID of the pull request being reviewed.
* `decision`: Must be strictly one of the following strings:
  * `APPROVE` (Code is clean and safe)
  * `REJECT_BUG` (Code contains syntax or logic errors)
  * `REJECT_SECURITY` (Code contains severe vulnerabilities)

### Reward Function
The environment utilizes a partial-progress reward system. For every correctly triaged PR, the agent receives a fractional reward `1.0 / N` (where N is the total number of PRs in the episode). The episode score ranges from `0.0` to `1.0`.

## 📈 Task Difficulties
The environment contains three distinct difficulty levels with programmatic graders:

1. **Easy:** The agent must identify a glaring syntax error (e.g., a missing closing quotation mark in a basic print statement) and reject it as a bug.
2. **Medium:** The agent is presented with a mix of clean code and subtle functional logic errors (e.g., an addition function that subtracts).
3. **Hard:** The agent must parse ambiguous code and identify critical security flaws, specifically an SQL Injection and an OS Command Injection, separating them from benign code.

## 📁 Repository Structure
* `environment.py`: The core OpenEnv logic, state management, and grading mechanics.
* `inference.py`: The baseline inference script adhering strictly to the hackathon's `[START]`, `[STEP]`, and `[END]` stdout formatting requirements.
* `openenv.yaml`: Metadata configuration for automated validation.
* `app.py`: A FastAPI server to maintain constant uptime and respond to ping/reset requests for evaluation.
* `Dockerfile`: Containerization instructions exposing port 7860.
* `requirements.txt`: Python dependencies.

## 🚀 Setup & Execution

### Running via Docker (Recommended)
This environment is designed to be run as a Docker container, seamlessly matching the Hugging Face Spaces deployment.

1. **Build the image:**
   ```bash
   docker build -t code-review-env .
