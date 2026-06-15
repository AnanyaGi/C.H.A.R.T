"""
VisuAlly — Image Accessibility API
Multi-Agent Architecture via LangGraph + Constitutional AI (P1–P10)
FastAPI endpoints remain intact; all LLM orchestration runs through the graph.
Updated to use NVIDIA NIM API (google/gemma-3-27b-it).
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import os
import io
import csv
import uuid
import json
import re
import pandas as pd
from datetime import datetime
from textstat import flesch_reading_ease, flesch_kincaid_grade
from typing import List, Dict, Tuple, Optional, Annotated
import glob
import secrets
import requests
import base64
import mimetypes
import time
from werkzeug.utils import secure_filename


# ── LangGraph ──────────────────────────────────────────────────────────────────
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# =============================================================================
# 0.  NVIDIA NIM API CONFIGURATION
# =============================================================================

NIM_API_KEY = "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"
NIM_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL = "google/gemma-3-27b-it"

def call_nvidia_nim(messages: list, image_path: Optional[str] = None) -> str:
    """Helper function to call the NVIDIA NIM API with aggressive retry logic for strict rate limits."""
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Accept": "application/json"
    }

    formatted_messages = []
    for msg in messages:
        role = msg["role"]
        content_text = msg["content"]
        
        if role == "user" and image_path and os.path.exists(image_path) and msg == messages[-1]:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            mime_type, _ = mimetypes.guess_type(image_path)
            mime_type = mime_type or "image/jpeg"

            formatted_content = [
                {"type": "text", "text": content_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]
            formatted_messages.append({"role": role, "content": formatted_content})
        else:
            formatted_messages.append({"role": role, "content": content_text})

    payload = {
        "model": NIM_MODEL,
        "messages": formatted_messages,
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "stream": False
    }

    max_retries = 8 

    for attempt in range(max_retries):
        try:
            response = requests.post(NIM_INVOKE_URL, headers=headers, json=payload, timeout=90)
            
            if response.status_code == 429:
                # 40 RPM means we need to wait for the minute window to clear.
                wait_time = 45 * (attempt + 1) # Wait 20s, 40s, 60s...
                print(f"[NIM API] Rate limit hit. Waiting {wait_time}s to clear quota window...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            
            # Safe JSON parsing to prevent KeyErrors on unexpected API returns
            resp_json = response.json()
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                return resp_json["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unexpected response format from API: {resp_json}")

        except requests.exceptions.Timeout:
            wait_time = 15
            print(f"[NIM API] Request timed out. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        
        except requests.exceptions.RequestException as e:
            wait_time = 15
            print(f" [NIM API] Network/HTTP error: {e}. Retrying in {wait_time}s...")
            if getattr(e, 'response', None) is not None:
                print(f"Details: {e.response.text}")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f" [NIM API] Unexpected parsing error: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(10)

    raise Exception("Max retries exceeded. The NVIDIA NIM API is currently unreachable or rate limits are too high.")

# =============================================================================
# 1.  STATE DEFINITION
# =============================================================================

class GraphState(TypedDict):
    """Shared state that flows through every node in the graph."""
    user_profile:        dict   
    image_path:          str    
    tone:                str    
    summary_type:        str    
    user_history:        list   
    vlm_prompt:          str    
    raw_description:     str    
    current_description: str    
    critique_report:     dict   
    compliance_score:    float  
    iteration_count:     int    
    revisions_history:   list   
    principle_status:    str    


# =============================================================================
# 2.  CONSTITUTIONAL AI — Accessibility Constitution (P1–P10)
# =============================================================================

CONSTITUTION = {
    "P1":  "Chart Identification — Must explicitly state the type of visualization "
           "(e.g., bar chart, line graph) and its exact title.",
    "P2":  "Structural Elements — Must define the core structure. IF APPLICABLE "
           "(like in graphs), explicitly state the X and Y axes and units. For other "
           "types (like flowcharts or pie charts), define the layout, main categories, "
           "or legends.",
    "P3":  "High-Level Summary — Must provide a 1-2 sentence overview of the primary "
           "trend, correlation, or insight shown.",
    "P4":  "Key Data Points — Must identify the most critical exact values (maximums, "
           "minimums, outliers) OR the most critical steps/nodes in a process diagram.",
    "P5":  "No Decorative Clutter — Must NOT describe background colors, grid lines, "
           "or font styles unless they explicitly encode data.",
    "P6":  "No Inference — Must strictly report the data shown. Do NOT guess, assume "
           "context, or explain the real-world causes of the trends.",
    "P7":  "Data Generalization — For dense data (e.g., scatter plots), describe the "
           "general cluster or trajectory instead of listing every single coordinate.",
    "P8":  "Semantic Structure — Must follow a predictable flow for screen readers: "
           "Title/Type → Structure (Axes/Categories/Layout) → Summary → Key Data/Steps.",
    "P9":  "Readability — Use clear, precise language. Avoid conversational filler "
           "like 'Here is a chart showing...'. Start directly with the facts.",
    "P10": "Objective Tone — Use entirely objective, factual language. Avoid emotional, "
           "evaluative, or promotional words.",
}

CONSTITUTION_BLOCK = "\n".join(f"{pid}: {rule}" for pid, rule in CONSTITUTION.items())


# =============================================================================
# 3.  HELPER — JSON critique parser
# =============================================================================

def parse_critique_json(raw: str) -> dict:
    cleaned = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        critique = json.loads(cleaned)
        for pid in CONSTITUTION:
            if pid not in critique:
                critique[pid] = {"status": "satisfied", "reason": ""}
        return critique
    except json.JSONDecodeError:
        pass

    fallback: dict = {}
    for line in cleaned.split("\n"):
        m = re.match(r"(P\d+):?\s*(satisfied|violated)[,\s]*(.*)", line, re.IGNORECASE)
        if m:
            pid, status, reason = m.group(1).upper(), m.group(2).lower(), m.group(3).strip()
            fallback[pid] = {"status": status, "reason": reason}
    for pid in CONSTITUTION:
        if pid not in fallback:
            fallback[pid] = {"status": "satisfied", "reason": ""}
    return fallback


def _calc_compliance(critique: dict) -> float:
    satisfied = sum(1 for d in critique.values() if d.get("status") == "satisfied")
    return round(satisfied / len(CONSTITUTION), 2)


# =============================================================================
# 4.  AGENT NODES
# =============================================================================

def generate_optimal_prompt(state: GraphState) -> GraphState:
    profile  = state["user_profile"]
    history  = state.get("user_history", [])
    language = profile.get("language", "English")

    profile_str = (
        f"Language: {profile.get('language', 'English')} | "
        f"Vision Type: {profile.get('vision_type', 'Unknown')} | "
        f"Familiarity: {profile.get('familiarity', 'Beginner')} | "
        f"Primary Use: {profile.get('primary_use', 'General')}"
    )

    if history:
        lines = []
        for h in history:
            # FIX: Safely cast to string to prevent float/NaN errors from Pandas
            raw_comment = str(h.get('comment', '')).strip()
            if raw_comment.lower() == 'nan':
                raw_comment = ""
                
            comment_part = f" | Comment: \"{raw_comment}\"" if raw_comment else ""
            
            lines.append(
                f"  • [Iter {h.get('iteration','-')}] "
                f"Rating {h.get('rating','-')}/5 | "
                f"Tone: {h.get('tone','-')} | Length: {h.get('summary_type','-')} | "
                f"Score: {h.get('compliance_score','-')}{comment_part}"
            )
        history_str = "User's last rated sessions:\n" + "\n".join(lines)
    else:
        history_str = "No prior rated sessions for this user."

    meta_prompt = (
        "You are an expert data accessibility assistant.\n"
        "Analyze the user profile and past interaction history below. "
        "Output ONE optimised prompt for a Vision-Language Model to describe a data "
        "visualization (chart, graph, plot) in a way that best serves this visually "
        "impaired user relying on a screen reader. "
        "Consider their vision type, familiarity level, and how past prompts were rated.\n"
        "Output ONLY the prompt text — no preamble, no labels, no explanation.\n\n"
        "CRITICAL: Explicitly instruct the Vision-Language Model to use pure plain text "
        "only, with NO markdown, NO asterisks (*), and NO bullet points.\n"
        f"CRITICAL: Explicitly instruct the Vision model to write the output entirely "
        f"in {language}.\n\n"
        f"User Profile:\n{profile_str}\n\n"
        f"{history_str}"
    )

    vlm_prompt = call_nvidia_nim([{"role": "user", "content": meta_prompt}]).strip()
    vlm_prompt += f"\n\nMANDATORY: You MUST write the entire image description in {language} ONLY."

    print(f"[Profiler] VLM prompt generated ({len(vlm_prompt)} chars)")
    return {**state, "vlm_prompt": vlm_prompt}


def generate_initial_draft(state: GraphState) -> GraphState:
    vlm_prompt = state["vlm_prompt"]
    image_path = state["image_path"]

    raw_output = call_nvidia_nim(
        messages=[{"role": "user", "content": vlm_prompt}], 
        image_path=image_path
    )
    draft = raw_output.replace("*", "").replace("#", "").strip()
    
    print(f"[Vision] Initial draft generated ({len(draft)} chars)")

    return {
        **state,
        "raw_description":   draft,
        "current_description": draft,
        "iteration_count":   0,
        "revisions_history": [],
        "critique_report":   {},
        "compliance_score":  0.0,
        "principle_status":  "{}",
    }


def critique_description(state: GraphState) -> GraphState:
    description   = state["current_description"]
    language      = state["user_profile"].get("language", "English")
    iteration     = state["iteration_count"]

    critique_prompt = (
        f"You are an accessibility auditor. Evaluate the image description below "
        f"against the Accessibility Constitution (P1–P10).\n\n"
        f"Language: {language}\n\n"
        f"Accessibility Constitution:\n{CONSTITUTION_BLOCK}\n\n"
        f'Description to evaluate:\n"{description}"\n\n'
        f"For each principle, output a JSON object with keys P1 to P10, each "
        f"containing \"status\" (\"satisfied\" or \"violated\") and a short \"reason\" "
        f"(empty string if satisfied). Return ONLY valid JSON, no other text."
    )

    raw = call_nvidia_nim([{"role": "user", "content": critique_prompt}]).strip()
    critique = parse_critique_json(raw)
    score    = _calc_compliance(critique)

    principle_status = state.get("principle_status", "{}")
    if iteration == 0:
        status_dict      = {pid: d.get("status") for pid, d in critique.items()}
        principle_status = json.dumps(status_dict, ensure_ascii=False)

    print(f"[Auditor] reviser_iter={iteration} score={score} ({int(score * 10)}/10 principles satisfied)")

    return {
        **state,
        "critique_report":  critique,
        "compliance_score": score,
        "principle_status": principle_status,
    }


def revise_description(state: GraphState) -> GraphState:
    critique     = state["critique_report"]
    current      = state["current_description"]
    language     = state["user_profile"].get("language", "English")
    tone         = state.get("tone", "neutral")
    summary_type = state.get("summary_type", "medium")
    image_path   = state["image_path"]
    iteration    = state["iteration_count"]

    violations = {pid: data for pid, data in critique.items() if data.get("status") == "violated"}

    if not violations:
        print(f"[Reviser] No violations; skipping revision.")
        new_description = current
    else:
        violation_lines = "\n".join(
            f"  {pid} ({CONSTITUTION[pid].split('—')[0].strip()}): {data['reason']}"
            for pid, data in violations.items()
        )

        tone_map   = {"simple": "Use very simple, clear language.",
                      "detailed": "Provide rich detail while remaining clear.",
                      "neutral": "Use balanced, factual language."}
        length_map = {"short": "Keep it concise but include all key information.",
                      "medium": "Aim for moderate length with good detail.",
                      "long": "Be comprehensive but maintain clarity."}

        structural_guidance = (
            "\n**Structural Rules for Visualizations:**\n"
            "1. Present information in a logical, top-down flow: Title and Type → "
            "Overall Structure or Layout → High-Level Summary → Specific Key Data "
            "or Process Steps (as applicable).\n"
            "2. Strip away all visual clutter. Do not mention grid lines, background "
            "colors, or formatting.\n"
            "3. State ONLY what the image shows. Do not invent causes or context.\n"
            "4. FORMATTING RULE: Do NOT use markdown formatting. Absolutely NO "
            "asterisks (*), hash marks (#), bullet points, or bold text. "
            "Output plain text paragraphs only.\n"
        )

        revise_prompt = (
            f"You are rewriting an image description to fix specific accessibility "
            f"violations.\n\n"
            f"Language: {language}\n"
            f"Tone: {tone_map.get(tone, 'Use balanced, factual language.')}\n"
            f"Length: {length_map.get(summary_type, 'Aim for moderate length with good detail.')}\n"
            f"{structural_guidance}\n"
            f"Fix ONLY the violated principles listed below. Look at the provided image "
            f"to find any missing factual information (like data points or axis labels).\n\n"
            f"Original description:\n{current}\n\n"
            f"Principles to fix:\n{violation_lines}\n\n"
            f"CRITICAL: Output ONLY the revised description in pure plain text. "
            f"No markdown, no asterisks, no bullets."
        )

        raw_output = call_nvidia_nim(
            messages=[{"role": "user", "content": revise_prompt}],
            image_path=image_path
        )
        new_description = raw_output.strip().replace("*", "").replace("#", "")

    new_iteration = iteration + 1
    history       = list(state.get("revisions_history", []))

    while len(history) < 3:
        history.append("")
    if new_iteration <= 3:
        history[new_iteration - 1] = new_description

    print(f"[Reviser] Revision {new_iteration} complete ({len(new_description)} chars)")

    return {
        **state,
        "current_description": new_description,
        "iteration_count":     new_iteration,
        "revisions_history":   history,
    }


# =============================================================================
# 5.  CONDITIONAL ROUTING  (Auditor → END or Reviser)
# =============================================================================

def route_after_audit(state: GraphState) -> str:
    score     = state.get("compliance_score", 0.0)
    iteration = state.get("iteration_count", 0)

    if score == 1.0:
        print(f"[Router] Score=1.0 — routing to END.")
        return END
    if iteration >= 3:
        print(f"[Router] Max iterations ({iteration}) reached — routing to END.")
        return END

    print(f"[Router] Score={score}, iter={iteration} — routing to Reviser.")
    return "revise"


# =============================================================================
# 6.  BUILD THE LANGGRAPH WORKFLOW
# =============================================================================

def build_graph() -> StateGraph:
    workflow = StateGraph(GraphState)
    workflow.add_node("profiler", generate_optimal_prompt)
    workflow.add_node("vision",   generate_initial_draft)
    workflow.add_node("auditor",  critique_description)
    workflow.add_node("reviser",  revise_description)
    workflow.set_entry_point("profiler")
    workflow.add_edge("profiler", "vision")
    workflow.add_edge("vision",   "auditor")
    workflow.add_conditional_edges("auditor", route_after_audit, {END: END, "revise": "reviser"})
    workflow.add_edge("reviser", "auditor")
    return workflow.compile()


def build_regen_graph() -> StateGraph:
    workflow = StateGraph(GraphState)
    workflow.add_node("auditor", critique_description)
    workflow.add_node("reviser", revise_description)
    workflow.set_entry_point("auditor")
    workflow.add_conditional_edges("auditor", route_after_audit, {END: END, "revise": "reviser"})
    workflow.add_edge("reviser", "auditor")
    return workflow.compile()

GRAPH       = build_graph()
REGEN_GRAPH = build_regen_graph()

# =============================================================================
# 7.  FASTAPI APP
# =============================================================================

app = FastAPI(title="VisuAlly - Image Accessibility API with LangGraph + Constitutional AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USERS_CSV   = "users.csv"
LOGS_DIR    = "logs"

USERS_HEADERS = [
    "user_id", "language", "vision_type",
    "familiarity", "primary_use", "created_at",
]

USER_LOG_HEADERS = [
    "timestamp", "user_id", "image_id", "iteration", "image_name",
    "vlm_prompt", "raw_description", "fkgl_raw", "fre_raw",
    "principle_status",
    "regenerate_desc_1", "regenerate_desc_2", "regenerate_desc_3",
    "final_description", "fkgl_final", "fre_final",
    "compliance_score", "cai_iterations",
    "rating", "comment", "tone", "summary_type",
]

os.makedirs(LOGS_DIR, exist_ok=True)
chat_histories: Dict[Tuple[str, str], list] = {}

# =============================================================================
# 8.  HELPER UTILITIES
# =============================================================================

def secure_filename(filename: str) -> str:
    filename = os.path.basename(filename)
    filename = re.sub(r"[^a-zA-Z0-9._-]", "", filename)
    return filename[:255]

def init_users_csv():
    if not os.path.exists(USERS_CSV):
        with open(USERS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=USERS_HEADERS).writeheader()
        print(f" Created {USERS_CSV}")
    else:
        df      = pd.read_csv(USERS_CSV)
        missing = [c for c in USERS_HEADERS if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = ""
            df[USERS_HEADERS].to_csv(USERS_CSV, index=False)
            print(f" Migrated {USERS_CSV}: added {missing}")

def get_user_log_path(user_id: str) -> str:
    return os.path.join(LOGS_DIR, f"{user_id}.csv")

def init_user_log(user_id: str):
    path = get_user_log_path(user_id)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=USER_LOG_HEADERS).writeheader()
    else:
        try:
            df      = pd.read_csv(path, dtype=str)
            missing = [c for c in USER_LOG_HEADERS if c not in df.columns]
            if missing:
                for c in missing:
                    df[c] = ""
                df[USER_LOG_HEADERS].to_csv(path, index=False)
        except Exception as e:
            print(f" Could not migrate log for {user_id}: {e}")

def append_to_csv(filepath: str, row_dict: dict):
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        headers = csv.DictReader(f).fieldnames
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=headers, extrasaction="ignore").writerow(row_dict)

def next_user_id() -> str:
    if not os.path.exists(USERS_CSV):
        return "1"
    try:
        df          = pd.read_csv(USERS_CSV, dtype=str)
        numeric_ids = pd.to_numeric(df.get("user_id", pd.Series()), errors="coerce").dropna()
        return str(int(numeric_ids.max()) + 1) if not numeric_ids.empty else "1"
    except Exception:
        return "1"

def get_user_profile(user_id: str) -> dict:
    if not os.path.exists(USERS_CSV):
        return {}
    try:
        df  = pd.read_csv(USERS_CSV, dtype=str)
        row = df[df["user_id"] == str(user_id)]
        return row.iloc[0].to_dict() if not row.empty else {}
    except Exception:
        return {}

def user_exists(user_id: str) -> bool:
    return bool(get_user_profile(user_id))

def get_user_history(user_id: str) -> list:
    path = get_user_log_path(user_id)
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if df.empty:
            return []
        rated = df[df["rating"].notna() & (df["rating"].astype(str).str.strip() != "")]
        return rated.sort_values("timestamp", ascending=False).head(5).to_dict(orient="records")
    except Exception:
        return []

def update_log_row(user_id: str, image_id: str, iteration: int, updates: dict) -> bool:
    path = get_user_log_path(user_id)
    init_user_log(user_id)
    if os.path.exists(path):
        df   = pd.read_csv(path, dtype=str)
        mask = (df["image_id"] == str(image_id)) & (df["iteration"] == str(iteration))
        if mask.any():
            for col, val in updates.items():
                df.loc[mask, col] = str(val)
            df.to_csv(path, index=False)
            return True
    return False

init_users_csv()

# =============================================================================
# 9.  ENDPOINTS
# =============================================================================

@app.post("/login")
async def login(
    user_id:      str = Form(None),
    language:     str = Form(None),
    vision_type:  str = Form(None),
    familiarity:  str = Form(None),
    primary_use:  str = Form(None),
):
    try:
        if user_id and user_id.strip():
            uid = user_id.strip()
            if not user_exists(uid):
                return JSONResponse({"error": "User ID not found."}, status_code=404)
            profile = get_user_profile(uid)
            init_user_log(uid)
            return JSONResponse({
                "user_id":     uid,
                "is_new":      False,
                "language":    profile.get("language",   "English"),
                "vision_type": profile.get("vision_type", ""),
                "familiarity": profile.get("familiarity", ""),
                "primary_use": profile.get("primary_use", ""),
            })

        if not all([language, vision_type, familiarity, primary_use]):
            return JSONResponse({"error": "Please fill in all profile fields."}, status_code=400)

        new_id = next_user_id()
        append_to_csv(USERS_CSV, {
            "user_id":    new_id,
            "language":   language,
            "vision_type": vision_type,
            "familiarity": familiarity,
            "primary_use": primary_use,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        init_user_log(new_id)
        return JSONResponse({
            "user_id":     new_id,
            "is_new":      True,
            "language":    language,
            "vision_type": vision_type,
            "familiarity": familiarity,
            "primary_use": primary_use,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        if not user_exists(user_id):
            return JSONResponse({"error": "Unknown user_id."}, status_code=400)

        image_bytes = await file.read()
        image_id    = str(uuid.uuid4())[:8]
        safe_fname  = secure_filename(file.filename)
        image_path  = f"temp_{image_id}_{safe_fname}"
        Image.open(io.BytesIO(image_bytes)).save(image_path)

        profile = get_user_profile(user_id)
        history = get_user_history(user_id)

        initial_state: GraphState = {
            "user_profile":       profile,
            "image_path":         image_path,
            "tone":               "neutral",
            "summary_type":       "medium",
            "user_history":       history,
            "vlm_prompt":         "",
            "raw_description":    "",
            "current_description": "",
            "critique_report":    {},
            "compliance_score":   0.0,
            "iteration_count":    0,
            "revisions_history":  [],
            "principle_status":   "{}",
        }

        final_state: GraphState = GRAPH.invoke(initial_state)

        vlm_prompt       = final_state["vlm_prompt"]
        raw_draft        = final_state["raw_description"]
        description      = final_state["current_description"]
        critique_report  = final_state["critique_report"]
        score            = final_state["compliance_score"]
        cai_iterations   = final_state["iteration_count"] + 1
        revisions        = final_state["revisions_history"]
        principle_status = final_state["principle_status"]

        while len(revisions) < 3:
            revisions.append("")

        fre_raw    = round(flesch_reading_ease(raw_draft), 2)
        fkgl_raw   = round(flesch_kincaid_grade(raw_draft), 2)
        fre_final  = round(flesch_reading_ease(description), 2)
        fkgl_final = round(flesch_kincaid_grade(description), 2)

        init_user_log(user_id)
        append_to_csv(get_user_log_path(user_id), {
            "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id":           user_id,
            "image_id":          image_id,
            "iteration":         1,
            "image_name":        file.filename,
            "vlm_prompt":        vlm_prompt,
            "raw_description":   raw_draft,
            "fkgl_raw":          fkgl_raw,
            "fre_raw":           fre_raw,
            "principle_status":  principle_status,
            "regenerate_desc_1": revisions[0],
            "regenerate_desc_2": revisions[1],
            "regenerate_desc_3": revisions[2],
            "final_description": description,
            "fkgl_final":        fkgl_final,
            "fre_final":         fre_final,
            "compliance_score":  score,
            "cai_iterations":    cai_iterations,
            "rating":            "",
            "comment":           "",
            "tone":              "neutral",
            "summary_type":      "medium",
        })

        return JSONResponse({
            "description":     description,
            "image_id":        image_id,
            "vlm_prompt":      vlm_prompt,
            "compliance_score": score,
            "cai_iterations":  cai_iterations,
            "critique_report": critique_report,
        })
    except Exception as e:
        import traceback
        print("\n" + "="*50)
        print(" FATAL UPLOAD ERROR TRACEBACK:")
        traceback.print_exc()
        print("="*50 + "\n")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/rate")
async def rate_summary(
    user_id:           str = Form(...),
    image_id:          str = Form(...),
    iteration:         int = Form(...),
    generated_summary: str = Form(...),
    rating:            int = Form(...),
    comment:           str = Form(""),
    tone:              str = Form("neutral"),
    summary_type:      str = Form("medium"),
):
    try:
        fre  = round(flesch_reading_ease(generated_summary), 2)
        fkgl = round(flesch_kincaid_grade(generated_summary), 2)

        updates = {"rating": rating, "comment": comment, "tone": tone, "summary_type": summary_type}

        if not update_log_row(user_id, image_id, iteration, updates):
            init_user_log(user_id)
            append_to_csv(get_user_log_path(user_id), {
                "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_id":           user_id,
                "image_id":          image_id,
                "iteration":         iteration,
                "final_description": generated_summary,
                **updates,
            })
        return JSONResponse({"status": "ok", "fre": fre, "fkgl": fkgl})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/regenerate")
async def regenerate(
    user_id:          str = Form(...),
    image_id:         str = Form(...),
    iteration:        int = Form(...),
    tone:             str = Form("neutral"),
    summary_type:     str = Form("medium"),
    previous_summary: str = Form(...),
):
    try:
        profile  = get_user_profile(user_id)
        language = profile.get("language", "English")

        image_path: Optional[str] = None
        for fname in os.listdir("."):
            if fname.startswith(f"temp_{image_id}_"):
                image_path = fname
                break

        regen_prompt = (
            f"Rewrite the following image description.\n"
            f"Language: {language} — You must write the entire description in {language} only.\n"
            f"Tone: {tone}\n"
            f"Length: {summary_type}\n"
            f"Start directly with the description — no introductions or meta-commentary.\n\n"
            f"FORMATTING RULE: Do NOT use markdown. No asterisks (*), hash marks (#), or bullets.\n\n"
            f"Original description:\n{previous_summary}"
        )

        raw_output = call_nvidia_nim(
            messages=[{"role": "user", "content": regen_prompt}],
            image_path=image_path
        )
        draft = raw_output.strip().replace("*", "").replace("#", "")
        
        fre_raw  = round(flesch_reading_ease(draft), 2)
        fkgl_raw = round(flesch_kincaid_grade(draft), 2)

        initial_state: GraphState = {
            "user_profile":        profile,
            "image_path":          image_path or "",
            "tone":                tone,
            "summary_type":        summary_type,
            "user_history":        [],
            "vlm_prompt":          regen_prompt,
            "raw_description":     draft,
            "current_description": draft,
            "critique_report":     {},
            "compliance_score":    0.0,
            "iteration_count":     0,
            "revisions_history":   [],
            "principle_status":    "{}",
        }

        final_state: GraphState = REGEN_GRAPH.invoke(initial_state)

        new_description  = final_state["current_description"]
        critique_report  = final_state["critique_report"]
        score            = final_state["compliance_score"]
        cai_iterations   = final_state["iteration_count"] + 1
        revisions        = final_state["revisions_history"]
        principle_status = final_state["principle_status"]

        while len(revisions) < 3:
            revisions.append("")

        fre_final  = round(flesch_reading_ease(new_description), 2)
        fkgl_final = round(flesch_kincaid_grade(new_description), 2)

        init_user_log(user_id)
        append_to_csv(get_user_log_path(user_id), {
            "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id":           user_id,
            "image_id":          image_id,
            "iteration":         iteration,
            "image_name":        "regenerated_iteration",
            "vlm_prompt":        regen_prompt,
            "raw_description":   draft,
            "fkgl_raw":          fkgl_raw,
            "fre_raw":           fre_raw,
            "principle_status":  principle_status,
            "regenerate_desc_1": revisions[0],
            "regenerate_desc_2": revisions[1],
            "regenerate_desc_3": revisions[2],
            "final_description": new_description,
            "fkgl_final":        fkgl_final,
            "fre_final":         fre_final,
            "compliance_score":  score,
            "cai_iterations":    cai_iterations,
            "rating":            "",
            "comment":           "",
            "tone":              tone,
            "summary_type":      summary_type,
        })

        key = (user_id, image_id)
        if key in chat_histories:
            del chat_histories[key]

        return JSONResponse({
            "description":     new_description,
            "compliance_score": score,
            "cai_iterations":  cai_iterations,
            "critique_report": critique_report,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/ask/")
async def ask_question(
    user_id:  str = Form(...),
    image_id: str = Form(...),
    question: str = Form(...),
    context:  str = Form(...),
    language: str = Form("English"),
):
    try:
        key = (user_id, image_id)
        
        # Initialize an empty list if this is the first interaction for this image
        if key not in chat_histories:
            chat_histories[key] = []
            
            # Combine the setup context and the first question into one message
            full_content = (
                f"System Note: You are a helpful assistant. The user is viewing an image "
                f"with this description: {context}. Answer questions based on "
                f"this description. Use {language} language. Do NOT use markdown.\n\n"
                f"User Question: {question}"
            )
        else:
            # For follow-up questions, just use the question itself
            full_content = question

        history = chat_histories[key]
        history.append({"role": "user", "content": full_content})

        answer = call_nvidia_nim(history).strip()
        history.append({"role": "assistant", "content": answer})

        return JSONResponse({"answer": answer})

    except Exception as e:
        import traceback
        print("\n" + "="*50)
        print(" ASK ENDPOINT ERROR TRACEBACK:")
        traceback.print_exc()
        print("="*50 + "\n")
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================================================================
# 10.  LIFECYCLE & ROOT
# =============================================================================

@app.on_event("shutdown")
def cleanup_temp_files():
    for f in glob.glob("temp_*"):
        try:
            os.remove(f)
            print(f"🧹 Cleaned up temporary file: {f}")
        except Exception as e:
            print(f"  Could not delete {f}: {e}")

@app.get("/")
def home():
    return {"message": "VisuAlly API — LangGraph Multi-Agent + Constitutional AI (NVIDIA NIM integration)"}