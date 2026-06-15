# C.H.A.R.T
## Constitutional Heuristics and Agentic Refinement for Translating Visual Data for Persons with Visual Impairments

![Status](https://img.shields.io/badge/status-Active-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/framework-FastAPI-darkgreen)

---

## Overview

**C.H.A.R.T** is an AI-powered accessibility platform that generates high-quality, context-aware textual descriptions of visual data (charts, graphs, diagrams) specifically tailored for persons with visual impairments. Using a multi-agent LangGraph workflow combined with Constitutional AI principles, the system iteratively refines descriptions to meet strict accessibility standards while respecting user preferences and reading ability.

### Key Innovation
The platform implements a **Constitutional AI framework** with 10 empirically-driven accessibility principles (P1–P10) to ensure descriptions are:
- **Accurate**: Factual, objective reporting of visual data
- **Accessible**: Clear, well-structured, screen-reader friendly
- **Personalized**: Adapted to user language, vision type, and familiarity level
- **Iterative**: Automatically refined through critique and revision cycles

---

## Features

### 🎯 Core Capabilities
- **Intelligent Image Description**: Vision-Language Models (NVIDIA NIM) analyze charts/graphs and generate initial descriptions
- **Constitutional Auditing**: Automated critique against 10 accessibility principles with violation detection
- **Iterative Refinement**: Up to 3 auto-correction cycles to maximize compliance score
- **User Personalization**: Descriptions adapt to language, vision type (e.g., blind, low vision), familiarity level, and use case
- **Learning from Feedback**: User ratings and comments inform future prompt generation
- **Multi-turn Q&A**: Chat interface for follow-up questions about image descriptions

### 📊 Quality Metrics
- **Compliance Scoring**: Automated assessment of adherence to 10 Constitutional principles
- **Readability Analysis**: Flesch-Kincaid Grade Level and Flesch Reading Ease metrics
- **Iteration Tracking**: Logs all revisions and compliance improvements across iterations

### 💾 Data & Analytics
- **User Profiles**: Track language, vision type, familiarity, and primary use case
- **Complete Audit Trail**: CSV-based logging of all interactions, descriptions, and metrics
- **Rating & Feedback**: Users rate summaries and provide comments to improve future sessions

### 🌍 Internationalization
- Multi-language support (English and beyond)
- Language-aware prompt generation and description refinement

---

## Architecture

### System Design

```
User Upload
    ↓
[Profiler Node]  ← Analyzes user history & preferences
    ↓
[Vision Node]    ← NVIDIA NIM generates initial draft
    ↓
[Auditor Node]   ← Checks against 10 Constitutional principles
    ↓
    ├─ Score = 1.0 OR Iter ≥ 3? → END
    │
    └─ Violations found? → [Reviser Node]
                             ↓
                          (Fix violations)
                             ↓
                          (Loop back to Auditor)
    ↓
Final Description + Metrics
```

### Components

| Component | Role |
|-----------|------|
| **Profiler** | Generates optimized prompts based on user profile and past interactions |
| **Vision** | Calls NVIDIA NIM to generate initial image descriptions |
| **Auditor** | Evaluates descriptions against 10 Constitutional AI principles |
| **Reviser** | Addresses specific accessibility violations and improves compliance |
| **Router** | Conditional logic: advance to next iteration or terminate |
| **LangGraph** | Orchestrates multi-agent workflow with state management |

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **API Framework** | FastAPI |
| **Workflow Orchestration** | LangGraph |
| **Vision-Language Model** | NVIDIA NIM (Gemma 3 27B) |
| **Image Processing** | Pillow (PIL) |
| **Metrics** | textstat (Flesch-Kincaid, FRE) |
| **Data Storage** | CSV (users.csv, user_logs/{user_id}.csv) |
| **CORS Support** | FastAPI Middleware |

---

## Constitutional AI Principles (P1–P10)

The system enforces 10 principles for accessible chart/graph descriptions:

| Principle | Description |
|-----------|-------------|
| **P1: Chart Identification** | Explicitly state the visualization type (e.g., bar chart, line graph) and title |
| **P2: Structural Elements** | Define core structure: X/Y axes, units, or layout/categories for non-graph visuals |
| **P3: High-Level Summary** | Provide 1–2 sentence overview of primary trend or insight |
| **P4: Key Data Points** | Identify critical values (max, min, outliers) or process steps in diagrams |
| **P5: No Decorative Clutter** | Omit background colors, grid lines, fonts unless they encode data |
| **P6: No Inference** | Report only what the image shows; avoid assumptions or external context |
| **P7: Data Generalization** | For dense data, describe clusters/trajectories instead of listing every point |
| **P8: Semantic Structure** | Follow predictable flow: Title → Structure → Summary → Data/Steps |
| **P9: Readability** | Use clear, precise language; avoid filler phrases ("Here is a chart...") |
| **P10: Objective Tone** | Use factual, neutral language; avoid emotional or promotional words |

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA NIM API key (for Gemma 3 model access)
- CORS origin(s) configured for your frontend

### Step 1: Clone & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/chart-accessibility.git
cd chart-accessibility

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pillow pandas langgraph textstat requests werkzeug python-multipart
```

### Step 2: Configure Environment

Create a `.env` file or set environment variables:

```bash
# NVIDIA NIM API Configuration
NIM_API_KEY="your-nvidia-nim-api-key-here"
NIM_INVOKE_URL="https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL="google/gemma-3-27b-it"

# CORS Origins (update for your frontend)
CORS_ORIGINS=["http://127.0.0.1:5500", "http://localhost:5500"]
```

**Note**: The API key is currently hardcoded in the source. For production, use environment variables:

```python
import os
NIM_API_KEY = os.getenv("NIM_API_KEY", "default-key")
```

### Step 3: Run the API Server

```bash
# Start the FastAPI server
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# API will be available at: http://127.0.0.1:8000
# Interactive docs: http://127.0.0.1:8000/docs
```

---

## API Endpoints

### 1. **POST /login**
Authenticate a user or create a new profile.

**Request (Existing User):**
```json
{
  "user_id": "1"
}
```

**Request (New User):**
```json
{
  "language": "English",
  "vision_type": "blind",
  "familiarity": "intermediate",
  "primary_use": "data_analysis"
}
```

**Response:**
```json
{
  "user_id": "1",
  "is_new": false,
  "language": "English",
  "vision_type": "blind",
  "familiarity": "intermediate",
  "primary_use": "data_analysis"
}
```

---

### 2. **POST /upload/**
Upload an image and generate an accessible description.

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/upload/ \
  -F "file=@chart.png" \
  -F "user_id=1"
```

**Response:**
```json
{
  "description": "Bar chart titled 'Q4 Sales by Region' with...",
  "image_id": "abc12345",
  "vlm_prompt": "You are an expert...",
  "compliance_score": 0.9,
  "cai_iterations": 2,
  "critique_report": {
    "P1": {"status": "satisfied", "reason": ""},
    "P2": {"status": "satisfied", "reason": ""},
    ...
  }
}
```

---

### 3. **POST /rate**
Submit user feedback (rating, comments, tone/length preferences).

**Request:**
```json
{
  "user_id": "1",
  "image_id": "abc12345",
  "iteration": 1,
  "generated_summary": "Bar chart titled...",
  "rating": 5,
  "comment": "Great, very clear!",
  "tone": "neutral",
  "summary_type": "medium"
}
```

**Response:**
```json
{
  "status": "ok",
  "fre": 62.5,
  "fkgl": 8.2
}
```

---

### 4. **POST /regenerate**
Regenerate a description with different tone or length preferences.

**Request:**
```json
{
  "user_id": "1",
  "image_id": "abc12345",
  "iteration": 2,
  "tone": "simple",
  "summary_type": "long",
  "previous_summary": "Bar chart titled..."
}
```

**Response:**
```json
{
  "description": "Rewritten description...",
  "compliance_score": 0.95,
  "cai_iterations": 1,
  "critique_report": { ... }
}
```

---

### 5. **POST /ask/**
Ask follow-up questions about an image description (multi-turn chat).

**Request:**
```json
{
  "user_id": "1",
  "image_id": "abc12345",
  "question": "What was the highest sales figure?",
  "context": "Bar chart titled 'Q4 Sales by Region'...",
  "language": "English"
}
```

**Response:**
```json
{
  "answer": "The highest sales figure was $250,000 in the North region."
}
```

---

### 6. **GET /**
Health check endpoint.

**Response:**
```json
{
  "message": "VisuAlly API — LangGraph Multi-Agent + Constitutional AI (NVIDIA NIM integration)"
}
```

---

## Data Storage & Logging

### User Profiles (`users.csv`)
Stores user metadata:
```csv
user_id,language,vision_type,familiarity,primary_use,created_at
1,English,blind,intermediate,data_analysis,2024-01-15 10:30:00
2,Hindi,low_vision,beginner,education,2024-01-15 11:00:00
```

### User Logs (`logs/{user_id}.csv`)
Complete audit trail for each user:
```csv
timestamp,user_id,image_id,iteration,image_name,vlm_prompt,raw_description,fkgl_raw,fre_raw,principle_status,regenerate_desc_1,regenerate_desc_2,regenerate_desc_3,final_description,fkgl_final,fre_final,compliance_score,cai_iterations,rating,comment,tone,summary_type
```

---

## Configuration Options

### User Profile Fields
- **language**: Language code (e.g., "English", "Hindi", "Spanish")
- **vision_type**: Type of visual impairment (e.g., "blind", "low_vision", "colorblind")
- **familiarity**: User's data literacy level ("beginner", "intermediate", "advanced")
- **primary_use**: Primary use case ("data_analysis", "education", "reporting", "general")

### Tone Preferences
- **simple**: Very clear, basic language
- **detailed**: Rich detail while remaining clear
- **neutral**: Balanced, factual language

### Summary Types
- **short**: Concise but complete
- **medium**: Moderate length with good detail
- **long**: Comprehensive and thorough

---

## Usage Examples

### Example 1: Create New User & Upload Image

```bash
#!/bin/bash

# Step 1: Create new user
RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/login \
  -F "language=English" \
  -F "vision_type=blind" \
  -F "familiarity=intermediate" \
  -F "primary_use=data_analysis")

USER_ID=$(echo $RESPONSE | jq -r '.user_id')
echo "Created user: $USER_ID"

# Step 2: Upload image
UPLOAD=$(curl -s -X POST http://127.0.0.1:8000/upload/ \
  -F "file=@my_chart.png" \
  -F "user_id=$USER_ID")

IMAGE_ID=$(echo $UPLOAD | jq -r '.image_id')
SCORE=$(echo $UPLOAD | jq -r '.compliance_score')
echo "Image uploaded. ID: $IMAGE_ID, Compliance: $SCORE"

# Step 3: Rate the description
curl -s -X POST http://127.0.0.1:8000/rate \
  -F "user_id=$USER_ID" \
  -F "image_id=$IMAGE_ID" \
  -F "iteration=1" \
  -F "generated_summary=$(echo $UPLOAD | jq -r '.description')" \
  -F "rating=5" \
  -F "comment=Perfect!" \
  -F "tone=neutral" \
  -F "summary_type=medium"
```

### Example 2: Multi-turn Q&A

```python
import requests

base_url = "http://127.0.0.1:8000"
user_id = "1"
image_id = "abc12345"
context = "Bar chart showing Q4 sales: North $250K, South $180K, East $220K"

# Question 1
q1 = requests.post(f"{base_url}/ask/", data={
    "user_id": user_id,
    "image_id": image_id,
    "question": "What is the highest value?",
    "context": context,
    "language": "English"
})
print("A1:", q1.json()["answer"])

# Question 2 (follow-up, uses chat history)
q2 = requests.post(f"{base_url}/ask/", data={
    "user_id": user_id,
    "image_id": image_id,
    "question": "Which region had the lowest sales?",
    "context": context,
    "language": "English"
})
print("A2:", q2.json()["answer"])
```

---

## Error Handling

The API returns detailed error responses:

### Example Error Response
```json
{
  "error": "Unknown user_id."
}
```

### Common Status Codes
| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (missing fields, unknown user) |
| 404 | User not found |
| 429 | NVIDIA NIM rate limit (auto-retry with backoff) |
| 500 | Server error (check logs) |

---

## Performance & Limitations

### Rate Limiting
- NVIDIA NIM API: 40 requests per minute
- Automatic exponential backoff: 45s, 90s, 135s, etc.
- Max 8 retry attempts per request

### Constraints
- Max iterations per description: 3
- Max chat history length: Unbounded (consider trimming for production)
- Temp file cleanup: Automatic on server shutdown
- Max image size: Limited by FastAPI (default 25 MB)

### Latency
- Initial draft generation: ~15–30s (NVIDIA NIM + retries)
- Per-iteration audit & revision: ~20–40s
- Total end-to-end (3 iterations): ~90–180s

---

## Extending the System

### Adding a New Constitutional Principle

```python
CONSTITUTION = {
    ...
    "P11": "Your new principle — description here."
}
```

### Integrating a Different Vision Model

Replace the `call_nvidia_nim()` function:

```python
def call_custom_model(messages: list, image_path: Optional[str] = None) -> str:
    # Your implementation here
    pass
```

### Adding Database Persistence

Replace CSV logging with a database (SQLite, PostgreSQL):

```python
from sqlalchemy import create_engine
engine = create_engine("sqlite:///chart_accessibility.db")
# Use SQLAlchemy ORM for user/log management
```

---

## Testing

### Unit Tests
```bash
pytest tests/test_constitution.py
pytest tests/test_routing.py
pytest tests/test_endpoints.py
```

### Integration Tests
```bash
# Test the full workflow
python tests/e2e_workflow.py
```

### Manual Testing
```bash
# Using the interactive Swagger docs
http://127.0.0.1:8000/docs
```

---

## Deployment

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & Run:**
```bash
docker build -t chart-accessibility:latest .
docker run -e NIM_API_KEY=your-key -p 8000:8000 chart-accessibility:latest
```

### Cloud Deployment
- **AWS**: Deploy to EC2, Lambda (with API Gateway), or ECS
- **GCP**: Cloud Run, Compute Engine
- **Azure**: Container Instances, App Service
- **Heroku**: Push to Heroku Git remote

---

## Troubleshooting

### Issue: "Max retries exceeded" for NVIDIA NIM API

**Solution:**
1. Check your API key is correct
2. Verify rate limit (40 RPM); spread requests across time
3. Increase `max_retries` in `call_nvidia_nim()` (currently 8)
4. Check NVIDIA NIM service status

### Issue: Image upload fails with "Unknown user_id"

**Solution:**
1. Call `/login` first to create/verify user
2. Use the returned `user_id` in `/upload/` request

### Issue: Description contains markdown (asterisks, hashes)

**Solution:**
- The system strips these with `.replace("*", "").replace("#", "")`
- If still present, check NVIDIA NIM prompt — ensure "NO markdown" instruction is clear

### Issue: Logs directory not found

**Solution:**
```bash
mkdir -p logs
```

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-principle`)
3. Add tests for new functionality
4. Submit a pull request with a detailed description

---



