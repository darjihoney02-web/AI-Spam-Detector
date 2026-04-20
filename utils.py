"""
SpamShield AI — utils.py
Helper utilities: text highlighting, risk badge, confidence label.
"""

import re
import html

# Words to highlight in results
HIGH_RISK_PATTERNS = [
    "free", "win", "winner", "prize", "cash", "reward", "offer", "click here",
    "urgent", "immediately", "act now", "limited time", "verify", "account",
    "password", "bank", "login", "confirm", "suspended", "congratulations",
    "selected", "claim", "discount", "deal", "sale", "bonus", "gift",
    "expires", "deadline", "last chance", "today only", "final notice",
    "alert", "warning", "attention", "validate", "reset", "blocked"
]

def highlight_text(text: str, top_words: list) -> str:
    """
    Return HTML with risky words wrapped in <mark> spans.
    top_words: list of (word, score) tuples from model output.
    """
    safe = html.escape(text)
    trigger_words = set(w for w, _ in top_words[:8])
    trigger_words.update(p for p in HIGH_RISK_PATTERNS if p in text.lower())

    for word in sorted(trigger_words, key=len, reverse=True):
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        safe = pattern.sub(
            lambda m: f'<mark class="highlight">{m.group(0)}</mark>',
            safe
        )
    return safe

def risk_color(risk_level: str) -> str:
    return {
        "Critical": "#ff2d55",
        "High":     "#ff9500",
        "Medium":   "#ffcc00",
        "Low":      "#34c759",
    }.get(risk_level, "#8e8e93")

def confidence_label(prob: float) -> str:
    if prob >= 90:   return "Very High Confidence"
    if prob >= 70:   return "High Confidence"
    if prob >= 50:   return "Moderate Confidence"
    if prob >= 30:   return "Low Confidence"
    return "Very Low Confidence"

def spam_type_emoji(spam_type: str) -> str:
    return {
        "Phishing":               "🎣",
        "Promotional Spam":       "📢",
        "Urgency Manipulation":   "⏰",
        "General Spam":           "🚫",
        "Legitimate":             "✅",
    }.get(spam_type, "❓")

def format_bulk_summary(results: list[dict]) -> dict:
    """Summarise bulk analysis results."""
    total   = len(results)
    spam    = sum(1 for r in results if r["prediction"] == 1)
    ham     = total - spam
    avg_sp  = round(sum(r["spam_prob"] for r in results) / total, 1) if total else 0
    types   = {}
    for r in results:
        t = r["spam_type"]
        types[t] = types.get(t, 0) + 1
    return {
        "total": total,
        "spam":  spam,
        "ham":   ham,
        "spam_rate": round(spam / total * 100, 1) if total else 0,
        "avg_spam_prob": avg_sp,
        "type_breakdown": types,
    }