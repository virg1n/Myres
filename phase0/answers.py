"""Final-answer extraction and comparison for GSM-style and MATH-style solutions.

Only answers that normalise to a number (integer, decimal, fraction, simple \\frac) are used as
ground truth, so checking is exact and does not need a symbolic grader."""
import re
from fractions import Fraction

_BOXED = re.compile(r"\\boxed\s*\{")


def last_boxed(text):
    """Content of the last \\boxed{...} (handles nested braces)."""
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth, j = 1, i
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[i:j - 1] if depth == 0 else None


def extract(text):
    """Model's final answer string, or None."""
    if not text:
        return None
    b = last_boxed(text)
    if b is not None:
        return b
    m = re.findall(r"[Ff]inal [Aa]nswer\s*[:：]?\s*\$?([^\n$]+)", text)
    if m:
        return m[-1]
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text[-300:])
    return nums[-1] if nums else None


def to_number(s):
    """Fraction value of a numeric answer string, or None if it is not a plain number."""
    if s is None:
        return None
    t = s.strip()
    t = re.sub(r"\\text\{[^}]*\}", "", t)
    t = re.sub(r"\\(?:mathrm|textbf|mathbf)\{([^}]*)\}", r"\1", t)
    t = t.replace("\\$", "").replace("\\!", "").replace("\\,", "").replace("\\ ", "").replace("$", "")
    t = t.replace("\\left", "").replace("\\right", "").replace("\\%", "").replace("%", "")
    t = t.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    t = t.strip().rstrip(".").strip()
    t = re.sub(r"^[a-zA-Z]\s*=\s*", "", t)            # "x = 5"
    t = re.sub(r"\s*(dollars?|cents?|units?|degrees?|meters?|cm|km|hours?|minutes?)$", "", t)
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", t)
    if m:
        num, den = int(m.group(2)), int(m.group(3))
        if den == 0:
            return None
        v = Fraction(num, den)
        return -v if m.group(1) else v
    m = re.fullmatch(r"(-?)\\frac(\d)(\d)", t)
    if m:
        v = Fraction(int(m.group(2)), int(m.group(3)))
        return -v if m.group(1) else v
    t = t.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t) else t
    m = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", t)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2))) if int(m.group(2)) else None
    try:
        if re.fullmatch(r"-?\d+(\.\d+)?", t):
            return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def key(s):
    """Canonical key used for comparing / voting on answers ('' = no answer)."""
    v = to_number(s)
    if v is not None:
        return str(v)
    if s is None:
        return ""
    return re.sub(r"\s+", "", s)


def gsm_truth(answer_field):
    """GSM8K / GSM-Symbolic ground truth: text after '####'."""
    return answer_field.split("####")[-1].strip().replace(",", "")
