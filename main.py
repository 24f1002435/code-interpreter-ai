
import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# Initialize FastAPI
app = FastAPI(title="Code Interpreter with AI Error Analysis")


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class CodeRequest(BaseModel):
    code: str


# AI structured output model
class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# API response model
class CodeResponse(BaseModel):
    error: List[int]
    result: str


# Execute Python code
def execute_python_code(code: str) -> dict:

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        exec(compile(code, "<string>", "exec"), {})

        output = sys.stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:

        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# Analyze Python errors using AI Pipe
def analyze_error_with_ai(code: str, tb: str) -> List[int]:

    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError(
            "AIPIPE_TOKEN is missing. Set it before starting Uvicorn."
        )

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openrouter/v1"
    )

    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Python error analysis assistant. "
                    "Identify the exact source line numbers responsible "
                    "for the Python execution error. "
                    "Return only valid JSON in this format: "
                    '{"error_lines": [3]}. '
                    "Line numbering starts at 1. "
                    "Do not invent line numbers."
                )
            },
            {
                "role": "user",
                "content": (
                    f"PYTHON CODE:\n{code}\n\n"
                    f"ERROR TRACEBACK:\n{tb}"
                )
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("AI returned an empty response.")

    analysis = ErrorAnalysis.model_validate_json(content)

    valid_lines = set(range(1, len(code.splitlines()) + 1))

    return sorted(
        set(analysis.error_lines).intersection(valid_lines)
    )


# Main API endpoint
@app.post(
    "/code-interpreter",
    response_model=CodeResponse
)

@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # Successful execution: no AI call needed
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Failed execution: use AI to find error lines
    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"]
        )

    except Exception as e:
        print("AI ERROR:", type(e).__name__, str(e))
        traceback.print_exc()

        raise HTTPException(
            status_code=502,
            detail="AI error analysis failed"
        )

    return {
        "error": error_lines,
        "result": execution["output"]
    }



# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Code Interpreter API is running",
        "docs": "/docs",
        "endpoint": "/code-interpreter"
    }
