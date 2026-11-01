"""Solve / verify prompts. Every mode keeps the same user message; only the system prompt differs."""

FORMAT = "Put the final answer in \\boxed{} at the end, for example \\boxed{42}."

NEUTRAL_SOLVE = "You are a helpful assistant. Solve the math problem step by step."

STRATEGIES = [
    ("forward", "Solve the math problem forwards: compute the intermediate quantities one by one, "
                "in the order they can be computed from the given values, until you reach the answer."),
    ("backward", "Solve the math problem backwards: start from the quantity that is asked, write down "
                 "what you need to know to find it, keep going until you reach the given values, and only "
                 "then compute."),
    ("algebra", "Solve the math problem algebraically: first introduce variables and translate the "
                "problem into equations, then solve the equations."),
    ("check", "Solve the math problem, then check the result by an independent method (substitute the "
              "answer back or estimate it) and fix any mistake before giving the final answer."),
]

NEUTRAL_VERIFY = "You are a careful math teacher who checks solutions."

VERIFY_STRATEGIES = [
    "You are a careful math teacher. Check solutions by recomputing every intermediate quantity in order.",
    "You are a careful math teacher. Check solutions by working backwards from what is asked and "
    "confirming each quantity that is needed.",
    "You are a careful math teacher. Check solutions by writing the problem as equations and solving "
    "them yourself.",
    "You are a careful math teacher. Check solutions by substituting the final answer back into all "
    "conditions of the problem.",
]

VERIFY_USER = ("Problem:\n{q}\n\nProposed solution:\n{s}\n\n"
               "Is the final answer of the proposed solution correct? Answer with a single word: Yes or No.")

MAX_SOLUTION_CHARS = 6000


def chat(tok, system, user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def mode_tag(prefix):
    """ModC condition token at the start of the input; prefix = mode index or None."""
    return "" if prefix is None else f"[Mode {prefix + 1}]\n"


def solve_prompt(tok, question, mode, prefix=None):
    """mode: 'neutral' or index into STRATEGIES."""
    system = NEUTRAL_SOLVE if mode == "neutral" else STRATEGIES[mode][1]
    return chat(tok, system, f"{mode_tag(prefix)}{question}\n\n{FORMAT}")


def verify_prompt(tok, question, solution, mode, prefix=None):
    """mode: 'neutral' or index into VERIFY_STRATEGIES."""
    system = NEUTRAL_VERIFY if mode == "neutral" else VERIFY_STRATEGIES[mode]
    s = solution[-MAX_SOLUTION_CHARS:]
    return chat(tok, system, mode_tag(prefix) + VERIFY_USER.format(q=question, s=s))
