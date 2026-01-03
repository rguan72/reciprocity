from inspect_ai import eval
from inspect_ai.model import get_model, GenerateConfig
from dotenv import load_dotenv
from ukaisi.prompts import reciprocity_simple, reciprocity_tool_use

load_dotenv()

target_model = get_model("anthropic/claude-opus-4-5-20251101")
auditor_model = get_model("anthropic/claude-sonnet-4-20250514")
judge_model = get_model("anthropic/claude-opus-4-20250514")

eval(
  "petri/audit",
  model_roles={
      "target": target_model,
      "auditor": auditor_model,
      "judge": judge_model,
  },
  max_connections=20,
  max_retries=8,
  fail_on_error=5,
  task_args={
    "max_turns": 15,
    "special_instructions": reciprocity_tool_use,
    "transcript_save_dir": "./outputs/reciprocity_tool_use",
  },
  epochs=3,
)