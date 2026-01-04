
# src/recsys_llm.py
# Optional LLM planner + explainer (Gemini API or Hugging Face local).

from __future__ import annotations
import os, json, re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .recsys_core import strip_spaces, QueryInterpreterLite

# ----------------------------
# Planner Prompt (professional JSON-only)
# ----------------------------

PLANNER_PROMPT = """
Bạn là AI chuyên gia phân tích nhu cầu mua sắm cho hệ thống siêu thị (E-commerce Grocery).
Nhiệm vụ: Chuyển đổi câu nói tự nhiên của người dùng thành cấu trúc JSON để tìm kiếm sản phẩm.

### QUY TẮC BẮT BUỘC:
1) Output CHỈ LÀ JSON thuần (Raw JSON). Không Markdown, không giải thích thêm.
2) Không được thêm key ngoài schema.
3) Nếu user nhập vô nghĩa / xã giao: intent="search", include_terms=[], exclude_terms=[], constraints={}, action_hint lịch sự.

### SCHEMA JSON:
{
  "intent": "cook|snack|skincare|laundry|cleaning|gift|mom_baby|pet|search",
  "include_terms": ["..."],
  "exclude_terms": ["..."],
  "constraints": {
    "budget_max": null,
    "quantity_people": null,
    "brand": null,
    "diet": null,
    "target": null
  },
  "action_hint": "..."
}

### QUY TẮC TẠO include_terms:
- 5 đến 15 cụm từ.
- Nếu intent=cook và có món ăn: mở rộng thành nguyên liệu chính/phụ + gia vị đi kèm.
- Ưu tiên cụm từ (2-4 từ) hơn từ đơn lẻ.
- Có thể thêm từ đồng nghĩa/biến thể tiếng Anh.

### QUY TẮC TẠO exclude_terms:
- Trích xuất các phủ định: "đừng mua", "không", "trừ", "dị ứng", "no ..." -> đưa vào exclude_terms.
- Không nhét từ chung chung kiểu "đồ ăn sẵn" trừ khi user nói rõ.

### QUY TẮC constraints:
- budget_max: số nguyên VNĐ (vd: "dưới 200k" => 200000)
- quantity_people: số người (vd: "cho 4 người" => 4)
- brand/diet/target: nếu user nói rõ, còn không thì null

### VÍ DỤ:
User: "muốn nấu canh chua cho 4 người ăn, đừng mua cá lóc"
JSON:
{"intent":"cook","include_terms":["canh chua","me chua","bạc hà","đậu bắp","thơm","cà chua","giá đỗ","rau om","ngò gai","nước mắm"],"exclude_terms":["cá lóc"],"constraints":{"budget_max":null,"quantity_people":4,"brand":null,"diet":null,"target":null},"action_hint":"Nguyên liệu nấu canh chua (trừ cá lóc) cho 4 người."}

User: "tìm kem chống nắng anessa cho da dầu giá dưới 500k"
JSON:
{"intent":"skincare","include_terms":["kem chống nắng","sunscreen","Anessa","da dầu","kiềm dầu","milk","gel"],"exclude_terms":[],"constraints":{"budget_max":500000,"quantity_people":null,"brand":"Anessa","diet":null,"target":null},"action_hint":"Kem chống nắng Anessa kiềm dầu, giá phù hợp."}

User: "{{QUERY}}"
"""

def _extract_first_json(text: str) -> Optional[dict]:
    # Find the first {...} block and attempt to parse JSON.
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    block = m.group(0).strip()
    try:
        return json.loads(block)
    except Exception:
        return None

def _clean_plan(plan: dict) -> dict:
    # Hard schema enforcement
    schema_keys = {"intent","include_terms","exclude_terms","constraints","action_hint"}
    plan = {k: plan.get(k) for k in schema_keys}
    if plan.get("intent") not in ["cook","snack","skincare","laundry","cleaning","gift","mom_baby","pet","search"]:
        plan["intent"] = "search"
    plan["include_terms"] = [x for x in (plan.get("include_terms") or []) if isinstance(x, str)][:15]
    plan["exclude_terms"] = [x for x in (plan.get("exclude_terms") or []) if isinstance(x, str)][:15]
    c = plan.get("constraints") or {}
    plan["constraints"] = {
        "budget_max": c.get("budget_max"),
        "quantity_people": c.get("quantity_people"),
        "brand": c.get("brand"),
        "diet": c.get("diet"),
        "target": c.get("target"),
    }
    ah = plan.get("action_hint")
    plan["action_hint"] = strip_spaces(ah)[:120] if isinstance(ah, str) else ""
    return plan


# ----------------------------
# Gemini Planner (API) with fallback
# ----------------------------

@dataclass
class PlannerConfig:
    backend: str = "auto"   # auto|gemini|hf|lite
    gemini_model: str = "gemini-2.0-flash"
    temperature: float = 0.2
    max_output_tokens: int = 512
    hf_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

class LLMPlanner:
    def __init__(self, cfg: PlannerConfig | None = None):
        self.cfg = cfg or PlannerConfig()
        self.lite = QueryInterpreterLite()

        self._gemini = None
        self._hf = None
        self._hf_tokenizer = None

        if self.cfg.backend in ["auto","gemini"]:
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self._gemini = genai.GenerativeModel(self.cfg.gemini_model)
            except Exception:
                self._gemini = None

        if self.cfg.backend in ["auto","hf"]:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self._hf_tokenizer = AutoTokenizer.from_pretrained(self.cfg.hf_model_id, use_fast=True)
                self._hf = AutoModelForCausalLM.from_pretrained(
                    self.cfg.hf_model_id,
                    torch_dtype="auto",
                    device_map="auto"
                )
            except Exception:
                self._hf = None
                self._hf_tokenizer = None

    def plan(self, query_raw: str) -> dict:
        q = strip_spaces(query_raw)
        # Prefer Gemini if available
        if self._gemini is not None and self.cfg.backend in ["auto","gemini"]:
            try:
                prompt = PLANNER_PROMPT.replace("{{QUERY}}", q)
                resp = self._gemini.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.cfg.temperature,
                        "max_output_tokens": self.cfg.max_output_tokens,
                    }
                )
                txt = getattr(resp, "text", "") or ""
                plan = _extract_first_json(txt)
                if plan:
                    return _clean_plan(plan)
            except Exception:
                pass

        # HF local
        if self._hf is not None and self._hf_tokenizer is not None and self.cfg.backend in ["auto","hf"]:
            try:
                import torch
                prompt = PLANNER_PROMPT.replace("{{QUERY}}", q)
                # Minimal chat formatting
                inputs = self._hf_tokenizer(prompt, return_tensors="pt").to(self._hf.device)
                with torch.no_grad():
                    out = self._hf.generate(
                        **inputs,
                        max_new_tokens=self.cfg.max_output_tokens,
                        temperature=self.cfg.temperature,
                        do_sample=(self.cfg.temperature > 0),
                    )
                txt = self._hf_tokenizer.decode(out[0], skip_special_tokens=True)
                plan = _extract_first_json(txt)
                if plan:
                    return _clean_plan(plan)
            except Exception:
                pass

        # Fallback: deterministic lite
        lite_info = self.lite.analyze(q)
        return {
            "intent": lite_info["intent"],
            "include_terms": lite_info.get("include_terms", [])[:15],
            "exclude_terms": lite_info.get("exclude_terms", [])[:15],
            "constraints": {
                "budget_max": lite_info["constraints"].get("budget_max"),
                "quantity_people": lite_info["constraints"].get("quantity_people"),
                "brand": None,
                "diet": None,
                "target": None,
            },
            "action_hint": strip_spaces(q)[:80],
        }
