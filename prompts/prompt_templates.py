"""
prompts/prompt_templates.py
All prompting strategies for Ambiguity and Incompleteness detection tasks.

Input modes
-----------
  title_only   – only the issue title is passed to the model
  title_desc   – both title and description are passed (original behaviour)

Each strategy × task × input_mode returns a formatted prompt string.
"""

# ─────────────────────────────────────────────
# FEW-SHOT EXAMPLES (shared across strategies)
# ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES_AMBIGUITY = [
    {
        "title": "Fix the issue #1",
        "description": "Something is broken. Please fix it.",
        "label": "Yes",
        "reason": "The issue is vague — 'something is broken' does not specify what component, "
                  "what behavior is expected, or under what conditions the problem occurs.",
    },
    {
        "title": "NullPointerException in UserService at line 142",
        "description": "A NullPointerException is thrown in UserService.java at line 142 "
                       "when userId is null during login. Stack trace attached.",
        "label": "No",
        "reason": "The issue is clear — it names the exact file, line, condition, and context.",
    },
    {
        "title": "Improve performance #3",
        "description": "The app needs to be faster.",
        "label": "Yes",
        "reason": "No baseline, no target metric, and no specific component are mentioned.",
    },
]

FEW_SHOT_EXAMPLES_INCOMPLETENESS = [
    {
        "title": "Export feature needed #1",
        "description": "Add export.",
        "label": "Yes",
        "reason": "Missing: what format to export, which data, who triggers it, and acceptance criteria.",
    },
    {
        "title": "File upload fails for files > 10MB",
        "description": "Uploading files larger than 10MB via POST /api/upload returns HTTP 413. "
                       "Expected: files up to 50MB should be accepted. Reproduction: use any file above 10MB.",
        "label": "No",
        "reason": "Contains endpoint, error code, expected behaviour, and reproduction steps — complete.",
    },
    {
        "title": "Dark mode #3",
        "description": "Add dark mode.",
        "label": "Yes",
        "reason": "No scope, no UI specs, no persistence requirements, and no affected screens are given.",
    },
]

# ─────────────────────────────────────────────
# INPUT MODES
# ─────────────────────────────────────────────

INPUT_MODES = ["title_only", "title_desc"]

# ─────────────────────────────────────────────
# SYSTEM / INSTRUCTION BLOCKS
# ─────────────────────────────────────────────

_AMBIGUITY_INSTRUCTION = (
    "You are an expert software requirements analyst. "
    "Your task is to determine whether a given software issue is AMBIGUOUS.\n"
    "An issue is AMBIGUOUS if it uses vague language, lacks clear context, "
    "has unclear scope, or can be interpreted in multiple ways.\n"
    "Respond with ONLY 'Yes' (ambiguous) or 'No' (not ambiguous)."
)

_AMBIGUITY_INSTRUCTION_TITLE_ONLY = (
    "You are an expert software requirements analyst. "
    "Your task is to determine whether a given software issue title is AMBIGUOUS.\n"
    "An issue title is AMBIGUOUS if it uses vague language, lacks clear context, "
    "has unclear scope, or can be interpreted in multiple ways.\n"
    "You will only be given the issue title — no description is available.\n"
    "Respond with ONLY 'Yes' (ambiguous) or 'No' (not ambiguous)."
)

_INCOMPLETENESS_INSTRUCTION = (
    "You are an expert software requirements analyst. "
    "Your task is to determine whether a given software issue is INCOMPLETE.\n"
    "An issue is INCOMPLETE if it is missing essential information such as "
    "steps to reproduce, expected vs actual behaviour, acceptance criteria, "
    "affected components, or environment details.\n"
    "Respond with ONLY 'Yes' (incomplete) or 'No' (not incomplete)."
)

_INCOMPLETENESS_INSTRUCTION_TITLE_ONLY = (
    "You are an expert software requirements analyst. "
    "Your task is to determine whether a given software issue title suggests the issue is INCOMPLETE.\n"
    "An issue is INCOMPLETE if it is missing essential information such as "
    "steps to reproduce, expected vs actual behaviour, acceptance criteria, "
    "affected components, or environment details.\n"
    "You will only be given the issue title — no description is available.\n"
    "Based on the title alone, assess whether the issue is likely incomplete.\n"
    "Respond with ONLY 'Yes' (incomplete) or 'No' (not incomplete)."
)

_AMBIGUITY_INSTRUCTION_COT = (
    "You are an expert software requirements analyst. "
    "Your task is to determine whether a given software issue is AMBIGUOUS.\n"
    "An issue is AMBIGUOUS if it uses vague language, lacks clear context, "
    "has unclear scope, or can be interpreted in multiple ways.\n"
    "Think step-by-step in at most 5 bullet points before giving your final answer.\n"
    "End your response with 'Final Answer: Yes' or 'Final Answer: No'."
)

_AMBIGUITY_INSTRUCTION_COT_TITLE_ONLY = (
    "You are an expert software requirements analyst. "
    "Your task is to determine whether a given software issue title is AMBIGUOUS.\n"
    "An issue title is AMBIGUOUS if it uses vague language, lacks clear context, "
    "has unclear scope, or can be interpreted in multiple ways.\n"
    "You will only be given the issue title — no description is available.\n"
    "Think step-by-step in at most 5 bullet points before giving your final answer.\n"
    "End your response with 'Final Answer: Yes' or 'Final Answer: No'."
)

_INCOMPLETENESS_INSTRUCTION_COT = (
    "You are an expert software requirements analyst. "
    "Your task is to determine whether a given software issue is INCOMPLETE.\n"
    "An issue is INCOMPLETE if it is missing essential information such as "
    "steps to reproduce, expected vs actual behaviour, acceptance criteria, "
    "affected components, or environment details.\n"
    "Think step-by-step before giving your final answer.\n"
    "End your response with 'Final Answer: Yes' or 'Final Answer: No'."
)

_INCOMPLETENESS_INSTRUCTION_COT_TITLE_ONLY = (
    "You are an expert software requirements analyst. "
    "Your task is to determine whether a given software issue title suggests the issue is INCOMPLETE.\n"
    "An issue is INCOMPLETE if it is missing essential information such as "
    "steps to reproduce, expected vs actual behaviour, acceptance criteria, "
    "affected components, or environment details.\n"
    "You will only be given the issue title — no description is available.\n"
    "Think step-by-step before giving your final answer.\n"
    "End your response with 'Final Answer: Yes' or 'Final Answer: No'."
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _format_issue_title_only(title: str) -> str:
    return f"Issue Title: {title}"


def _format_issue_title_desc(title: str, description: str) -> str:
    return f"Issue Title: {title}\nIssue Description: {description}"


def _format_few_shot_block(examples: list, title_only: bool = False) -> str:
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Title: {ex['title']}")
        if not title_only:
            lines.append(f"  Description: {ex['description']}")
        lines.append(f"  Answer: {ex['label']}")
        lines.append("")
    return "\n".join(lines)


def _format_few_shot_cot_block(examples: list, title_only: bool = False) -> str:
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Title: {ex['title']}")
        if not title_only:
            lines.append(f"  Description: {ex['description']}")
        lines.append(f"  Reasoning: {ex['reason']}")
        lines.append(f"  Final Answer: {ex['label']}")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# AMBIGUITY PROMPTS — TITLE + DESCRIPTION
# ─────────────────────────────────────────────────────────────────

def zero_shot_ambiguity_title_desc(title: str, description: str) -> str:
    return (
        f"{_AMBIGUITY_INSTRUCTION}\n\n"
        f"{_format_issue_title_desc(title, description)}\n\n"
        "Is this issue ambiguous? Answer 'Yes' or 'No':"
    )


def few_shot_ambiguity_title_desc(title: str, description: str) -> str:
    examples_block = _format_few_shot_block(FEW_SHOT_EXAMPLES_AMBIGUITY, title_only=False)
    return (
        f"{_AMBIGUITY_INSTRUCTION}\n\n"
        "Here are some examples:\n\n"
        f"{examples_block}"
        "Now classify the following issue:\n"
        f"{_format_issue_title_desc(title, description)}\n\n"
        "Is this issue ambiguous? Answer 'Yes' or 'No':"
    )


def cot_ambiguity_title_desc(title: str, description: str) -> str:
    return (
        f"{_AMBIGUITY_INSTRUCTION_COT}\n\n"
        f"{_format_issue_title_desc(title, description)}\n\n"
        "Think step-by-step:\n"
        "1. What information is present in this issue?\n"
        "2. Is the language vague or clear?\n"
        "3. Can this issue be interpreted in multiple ways?\n\n"
        "Final Answer (Yes/No):"
    )


def few_shot_cot_ambiguity_title_desc(title: str, description: str) -> str:
    examples_block = _format_few_shot_cot_block(FEW_SHOT_EXAMPLES_AMBIGUITY, title_only=False)
    return (
        f"{_AMBIGUITY_INSTRUCTION_COT}\n\n"
        "Here are examples with step-by-step reasoning:\n\n"
        f"{examples_block}"
        "Now classify the following issue:\n"
        f"{_format_issue_title_desc(title, description)}\n\n"
        "Think step-by-step, then provide your Final Answer (Yes/No):"
    )


# ─────────────────────────────────────────────────────────────────
# AMBIGUITY PROMPTS — TITLE ONLY
# ─────────────────────────────────────────────────────────────────

def zero_shot_ambiguity_title_only(title: str, description: str = "") -> str:
    return (
        f"{_AMBIGUITY_INSTRUCTION_TITLE_ONLY}\n\n"
        f"{_format_issue_title_only(title)}\n\n"
        "Is this issue title ambiguous? Answer 'Yes' or 'No':"
    )


def few_shot_ambiguity_title_only(title: str, description: str = "") -> str:
    examples_block = _format_few_shot_block(FEW_SHOT_EXAMPLES_AMBIGUITY, title_only=True)
    return (
        f"{_AMBIGUITY_INSTRUCTION_TITLE_ONLY}\n\n"
        "Here are some examples (titles only):\n\n"
        f"{examples_block}"
        "Now classify the following issue title:\n"
        f"{_format_issue_title_only(title)}\n\n"
        "Is this issue title ambiguous? Answer 'Yes' or 'No':"
    )


def cot_ambiguity_title_only(title: str, description: str = "") -> str:
    return (
        f"{_AMBIGUITY_INSTRUCTION_COT_TITLE_ONLY}\n\n"
        f"{_format_issue_title_only(title)}\n\n"
        "Think step-by-step:\n"
        "1. What information is present in this title?\n"
        "2. Is the language vague or specific?\n"
        "3. Can this title be interpreted in multiple ways?\n\n"
        "Final Answer (Yes/No):"
    )


def few_shot_cot_ambiguity_title_only(title: str, description: str = "") -> str:
    examples_block = _format_few_shot_cot_block(FEW_SHOT_EXAMPLES_AMBIGUITY, title_only=True)
    return (
        f"{_AMBIGUITY_INSTRUCTION_COT_TITLE_ONLY}\n\n"
        "Here are examples with step-by-step reasoning (titles only):\n\n"
        f"{examples_block}"
        "Now classify the following issue title:\n"
        f"{_format_issue_title_only(title)}\n\n"
        "Think step-by-step, then provide your Final Answer (Yes/No):"
    )


# ─────────────────────────────────────────────────────────────────
# INCOMPLETENESS PROMPTS — TITLE + DESCRIPTION
# ─────────────────────────────────────────────────────────────────

def zero_shot_incompleteness_title_desc(title: str, description: str) -> str:
    return (
        f"{_INCOMPLETENESS_INSTRUCTION}\n\n"
        f"{_format_issue_title_desc(title, description)}\n\n"
        "Is this issue incomplete? Answer 'Yes' or 'No':"
    )


def few_shot_incompleteness_title_desc(title: str, description: str) -> str:
    examples_block = _format_few_shot_block(FEW_SHOT_EXAMPLES_INCOMPLETENESS, title_only=False)
    return (
        f"{_INCOMPLETENESS_INSTRUCTION}\n\n"
        "Here are some examples:\n\n"
        f"{examples_block}"
        "Now classify the following issue:\n"
        f"{_format_issue_title_desc(title, description)}\n\n"
        "Is this issue incomplete? Answer 'Yes' or 'No':"
    )


def cot_incompleteness_title_desc(title: str, description: str) -> str:
    return (
        f"{_INCOMPLETENESS_INSTRUCTION_COT}\n\n"
        f"{_format_issue_title_desc(title, description)}\n\n"
        "Think step-by-step:\n"
        "1. What key information is provided in this issue?\n"
        "2. What essential information is missing (reproduction steps, expected behaviour, environment)?\n"
        "3. Can a developer act on this issue without asking for more information?\n\n"
        "Final Answer (Yes/No):"
    )


def few_shot_cot_incompleteness_title_desc(title: str, description: str) -> str:
    examples_block = _format_few_shot_cot_block(FEW_SHOT_EXAMPLES_INCOMPLETENESS, title_only=False)
    return (
        f"{_INCOMPLETENESS_INSTRUCTION_COT}\n\n"
        "Here are examples with step-by-step reasoning:\n\n"
        f"{examples_block}"
        "Now classify the following issue:\n"
        f"{_format_issue_title_desc(title, description)}\n\n"
        "Think step-by-step, then provide your Final Answer (Yes/No):"
    )


# ─────────────────────────────────────────────────────────────────
# INCOMPLETENESS PROMPTS — TITLE ONLY
# ─────────────────────────────────────────────────────────────────

def zero_shot_incompleteness_title_only(title: str, description: str = "") -> str:
    return (
        f"{_INCOMPLETENESS_INSTRUCTION_TITLE_ONLY}\n\n"
        f"{_format_issue_title_only(title)}\n\n"
        "Is this issue likely incomplete based on the title alone? Answer 'Yes' or 'No':"
    )


def few_shot_incompleteness_title_only(title: str, description: str = "") -> str:
    examples_block = _format_few_shot_block(FEW_SHOT_EXAMPLES_INCOMPLETENESS, title_only=True)
    return (
        f"{_INCOMPLETENESS_INSTRUCTION_TITLE_ONLY}\n\n"
        "Here are some examples (titles only):\n\n"
        f"{examples_block}"
        "Now classify the following issue title:\n"
        f"{_format_issue_title_only(title)}\n\n"
        "Is this issue likely incomplete based on the title alone? Answer 'Yes' or 'No':"
    )


def cot_incompleteness_title_only(title: str, description: str = "") -> str:
    return (
        f"{_INCOMPLETENESS_INSTRUCTION_COT_TITLE_ONLY}\n\n"
        f"{_format_issue_title_only(title)}\n\n"
        "Think step-by-step:\n"
        "1. What information does this title convey?\n"
        "2. Does the title suggest missing reproduction steps, expected behaviour, or context?\n"
        "3. Can a developer understand the full scope from the title alone?\n\n"
        "Final Answer (Yes/No):"
    )


def few_shot_cot_incompleteness_title_only(title: str, description: str = "") -> str:
    examples_block = _format_few_shot_cot_block(FEW_SHOT_EXAMPLES_INCOMPLETENESS, title_only=True)
    return (
        f"{_INCOMPLETENESS_INSTRUCTION_COT_TITLE_ONLY}\n\n"
        "Here are examples with step-by-step reasoning (titles only):\n\n"
        f"{examples_block}"
        "Now classify the following issue title:\n"
        f"{_format_issue_title_only(title)}\n\n"
        "Think step-by-step, then provide your Final Answer (Yes/No):"
    )


# ─────────────────────────────────────────────────────────────────
# REGISTRY
# Structure: PROMPT_REGISTRY[task][strategy][input_mode] -> function
# ─────────────────────────────────────────────────────────────────

PROMPT_REGISTRY = {
    "ambiguity": {
        "zero_shot": {
            "title_only": zero_shot_ambiguity_title_only,
            "title_desc": zero_shot_ambiguity_title_desc,
        },
        "few_shot": {
            "title_only": few_shot_ambiguity_title_only,
            "title_desc": few_shot_ambiguity_title_desc,
        },
        "cot": {
            "title_only": cot_ambiguity_title_only,
            "title_desc": cot_ambiguity_title_desc,
        },
        "few_shot_cot": {
            "title_only": few_shot_cot_ambiguity_title_only,
            "title_desc": few_shot_cot_ambiguity_title_desc,
        },
    },
    "incompleteness": {
        "zero_shot": {
            "title_only": zero_shot_incompleteness_title_only,
            "title_desc": zero_shot_incompleteness_title_desc,
        },
        "few_shot": {
            "title_only": few_shot_incompleteness_title_only,
            "title_desc": few_shot_incompleteness_title_desc,
        },
        "cot": {
            "title_only": cot_incompleteness_title_only,
            "title_desc": cot_incompleteness_title_desc,
        },
        "few_shot_cot": {
            "title_only": few_shot_cot_incompleteness_title_only,
            "title_desc": few_shot_cot_incompleteness_title_desc,
        },
    },
}

PROMPT_STRATEGIES = list(PROMPT_REGISTRY["ambiguity"].keys())   # ["zero_shot","few_shot","cot","few_shot_cot"]
TASKS             = list(PROMPT_REGISTRY.keys())                 # ["ambiguity","incompleteness"]