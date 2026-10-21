"""
Structural template speculative drafting policy.
v0.14.0 — Improved answer guessing + no-fallback + optimized answer guessing.

Key improvements over v0.12.0:
  1. Token-ID-based "The answer is X." pattern for complex outputs
     - Detects pattern via last generated tokens
     - Drafts guessed answer + period + closing tag in one shot
     - Major improvement for complex conclusion endings
  2. Improved answer guessing heuristic (category-aware, multi-signal)
     - Better detection of math/science/perception questions
     - ~40%+ target accuracy on full dev set
  3. Period-aware answer drafting
     - Predicts whether answer has period based on question format
  4. Gated low-value fallbacks — only use single-token predictions when high confidence
"""

import re
import string
from collections import defaultdict, deque

# =====================================================================
# Known token IDs from observed real outputs
# =====================================================================

# 36-token structural prefix — identical across all observed outputs
COMMON_PREFIX_TOKENS = [
    19389, 2864, 49970, 29, 220, 128013, 694, 28477, 49970, 1363,
    20996, 2599, 60459, 29, 220, 128014, 694, 32500, 60459, 1363,
    27, 793, 36404, 1753, 29, 220, 128015, 694, 793, 36404,
    1753, 1363, 27, 5910, 99769, 29,
]

TEMPLATE_TOKEN_COUNT = len(COMMON_PREFIX_TOKENS)  # 36

# Answer letter token IDs
ANSWER_TOKEN_MAP = {'A': 362, 'B': 426, 'C': 356, 'D': 423}
TOKEN_TO_ANSWER = {v: k for k, v in ANSWER_TOKEN_MAP.items()}
ANSWER_TOKEN_SET = set(ANSWER_TOKEN_MAP.values())

# Closing tag tokens: space + </CONCLUSION> + EOS
CLOSING_TOKENS = [694, 5910, 99769, 29, 128009]
PERIOD_TOKEN = 13
EOS_TOKEN = 128009
SPACE_TOKEN = 694

# "The answer is" tokens (for complex output ending pattern)
THE_TOKEN = 578
ANSWER_WORD_TOKEN = 4320
IS_TOKEN = 374

# "The answer is X." sequence (without X): [The, answer, is]
THE_ANSWER_IS_TOKENS = [THE_TOKEN, ANSWER_WORD_TOKEN, IS_TOKEN]

# Template text (for text-based fallback)
STRUCTURAL_TEMPLATE = (
    "<SUMMARY> <THINKING_OF_SUMMARY> </SUMMARY>\n\n"
    "<CAPTION> <THINKING_OF_CAPTION> </CAPTION>\n\n"
    "<REASONING> <THINKING_OF_REASONING> </REASONING>\n\n"
    "<CONCLUSION> "
)

CLOSING_TAG = "</CONCLUSION>"
_CHARS_PER_TOKEN_EST = 5.0
THE_ANSWER_IS_TEXT = "The answer is "
ANSWER_LETTERS = set("ABCD")

# When remaining template tokens < this, try answer+closing guess
TEMPLATE_END_GUESS_THRESHOLD = 8

# =====================================================================
# Answer guessing heuristics (v0.13 — improved multi-signal)
# =====================================================================

def _guess_answer(question):
    """
    Multi-signal answer guessing heuristic v0.14.
    Uses question text features to predict answer letter.
    Returns (letter, confidence_score).
    Dev accuracy: 43.8% (v0.13: 36.5%, oracle: 45.8%).
    """
    q = question or ""
    q_lower = q.lower()
    
    scores = {'A': 1.0, 'B': 1.0, 'C': 0.5, 'D': 1.0}  # baseline priors (C is rare)
    
    first_word = q_lower.split()[0] if q_lower.split() else ""
    first_word_stripped = first_word.rstrip(':')
    
    # Strong signal: how many/how much → B (counting questions)
    if q_lower.startswith('how many') or q_lower.startswith('how much'):
        scores['B'] += 3.0
    
    # Strong signal: 'The X...' → science/tech → A
    if first_word_stripped == 'the':
        scores['A'] += 2.5
    
    # Strong signal: hint/math format → D
    if first_word_stripped == 'hint':
        scores['D'] += 2.0
        scores['C'] += 0.5
    
    # What questions → B bias (perception majority)
    if first_word_stripped == 'what':
        scores['B'] += 1.0
    
    # Color questions → B (50%+ accuracy on dev)
    color_words = ['color', 'colour', 'red', 'blue', 'green', 'yellow', 'brown', 'orange', 'purple', 'black', 'white', 'gray', 'grey']
    color_count = sum(1 for w in color_words if w in q_lower)
    if color_count >= 2:
        scores['B'] += 1.5
    elif color_count >= 1:
        scores['B'] += 0.5
    
    # Green specifically → A bias
    if 'green' in q_lower:
        scores['A'] += 1.0
    
    # People in question → B (perception of people)
    people_words = [' man ', ' woman ', ' men ', ' people ', ' person ']
    if any(w in q_lower for w in people_words):
        scores['B'] += 1.5
    
    # Left/right position → B
    if ' left ' in q_lower or ' right ' in q_lower:
        scores['B'] += 1.0
    
    # Science keywords → A
    sci_keywords = ['science', 'scientific', 'technology', 'chemical', 'reaction',
                    'physics', 'biology', 'chemistry', 'gas ', 'liquid ', 'metal ']
    if any(kw in q_lower for kw in sci_keywords):
        scores['A'] += 1.5
    
    # Math/data keywords → D
    math_keywords = [' rate ', 'average', 'total ', 'difference', 'highest', 'lowest',
                     ' percentage', 'percent', 'ratio', 'increase', 'decrease']
    math_count = sum(1 for kw in math_keywords if kw in q_lower)
    if math_count >= 2:
        scores['D'] += 2.0
    elif math_count >= 1:
        scores['D'] += 1.0
    
    # "correct" keyword → D
    if 'correct' in q_lower:
        scores['D'] += 0.5
    
    best = max(scores, key=scores.get)
    conf = scores[best] / sum(scores.values()) if sum(scores.values()) > 0 else 0.25
    return best, conf

class _DraftObserver:
    def __init__(self):
        self._last_pos = -1
        self._last_qhash = ""
        self._last_decoded_len = 0
        self._last_proposed = False
        self._last_est_tokens = 0
        self._last_strategy = "none"
        self._last_stage = "unknown"
        self.reset_count = 0
        self.questions_seen = 0
        self.total_drafts = 0
        self.total_accepted = 0
        self.q_drafts = 0
        self.q_accepted = 0
        self._recent = deque(maxlen=50)
        self.strategy_stats = defaultdict(lambda: [0, 0, 0])
        self.stage_stats = defaultdict(lambda: [0, 0, 0])
        self._question_format = "unknown"
        self._guess_letter = "D"
        self._guess_confidence = 0.25
    
    def _qhash(self, q):
        if not q:
            return ""
        return str(hash(q[:200]))
    
    def _detect_reset(self, state):
        pos = len(state.get("generated_token_ids", []) or [])
        decoded = state.get("decoded_text", "") or ""
        qhash = self._qhash(state.get("question", ""))
        
        if self._last_pos < 0:
            return True, "first_call"
        if pos == 0:
            return True, "empty_generated"
        if qhash != self._last_qhash:
            return True, "question_changed"
        if pos < self._last_pos:
            return True, "position_backward"
        if len(decoded) < self._last_decoded_len and self._last_decoded_len > 0:
            return True, "decoded_shorter"
        return False, ""
    
    def _infer_accepted(self, state):
        pos = len(state.get("generated_token_ids", []) or [])
        if self._last_pos < 0:
            return 0, False
        delta = pos - self._last_pos
        if delta < 1:
            return 0, False
        accepted = delta - 1
        if self._last_est_tokens > 0:
            accepted = min(accepted, self._last_est_tokens)
        return max(0, accepted), True
    
    def _classify_question(self, question):
        q = question or ""
        if "Hint:" in q and "option letter" in q.lower():
            return "hint_complex"
        if "Options:" in q or "options:" in q or "Choices:" in q:
            return "simple_mc"
        if "Choose the" in q or "choose the" in q:
            return "simple_mc"
        return "unknown"
    
    def observe(self, state):
        decoded = state.get("decoded_text", "") or ""
        self._last_decoded_len = len(decoded)
        is_reset, reason = self._detect_reset(state)
        if is_reset:
            self._on_reset(reason)
            self._last_pos = len(state.get("generated_token_ids", []) or [])
            self._last_qhash = self._qhash(state.get("question", ""))
            self._last_proposed = False
            self.questions_seen += 1
            self._question_format = self._classify_question(state.get("question", ""))
            letter, conf = _guess_answer(state.get("question", ""))
            self._guess_letter = letter
            self._guess_confidence = conf
            return True, reason, 0
        accepted, valid = self._infer_accepted(state)
        if valid and self._last_proposed:
            self._record(accepted)
        self._last_pos = len(state.get("generated_token_ids", []) or [])
        self._last_proposed = False
        return False, "", accepted
    
    def _on_reset(self, reason):
        self.reset_count += 1
        self._recent.clear()
        self.q_drafts = 0
        self.q_accepted = 0
    
    def _record(self, accepted):
        proposed = max(1, self._last_est_tokens)
        strat = self._last_strategy
        stage = self._last_stage
        s = self.strategy_stats[strat]
        s[0] += accepted
        s[1] += proposed
        s[2] += 1
        t = self.stage_stats[stage]
        t[0] += accepted
        t[1] += proposed
        t[2] += 1
        self._recent.append((strat, accepted, proposed))
        self.total_drafts += 1
        self.total_accepted += accepted
        self.q_drafts += 1
        self.q_accepted += accepted
    
    def record_draft(self, n_tokens, strategy, stage="unknown"):
        self._last_proposed = True
        self._last_strategy = strategy
        self._last_stage = stage
        self._last_est_tokens = max(1, n_tokens)
    
    @property
    def question_format(self):
        return self._question_format
    
    @property
    def guess_letter(self):
        return self._guess_letter
    
    @property
    def guess_confidence(self):
        return self._guess_confidence


_observer = _DraftObserver()


# =====================================================================
# Strategy functions
# =====================================================================

def _draft_prefix_tokens(position, draft_budget, guess_letter, q_format):
    """
    Strategy 1: Token-ID-based structural prefix drafting.
    Uses exact known token IDs from observed data.
    Position tracked via generated_token_ids count.
    """
    if position >= TEMPLATE_TOKEN_COUNT:
        return [], "prefix_done"
    
    remaining = COMMON_PREFIX_TOKENS[position:]
    proposed = remaining[:draft_budget]
    
    if not proposed:
        return [], "prefix_empty"
    
    tokens_left_in_prefix = TEMPLATE_TOKEN_COUNT - position - len(proposed)
    
    if tokens_left_in_prefix <= TEMPLATE_END_GUESS_THRESHOLD and q_format in ("simple_mc", "hint_complex", "unknown"):
        slots = draft_budget - len(proposed)
        if slots >= 7:  # answer + period + space + 4 closing tokens minimum
            answer_tok = ANSWER_TOKEN_MAP.get(guess_letter, ANSWER_TOKEN_MAP['D'])
            # Try with period first (some answers have periods)
            # Actually, no period is more common (91%), try no-period first
            suffix = [answer_tok] + CLOSING_TOKENS
            proposed = proposed + suffix[:slots]
            return proposed, "prefix_answer_guess"
    
    return proposed, f"prefix_{position}"


def _draft_closing_after_answer(position, draft_budget, gen_ids):
    """
    Strategy 2: Closing tag drafting after answer letter is confirmed.
    Handles both period and no-period cases.
    """
    if position <= TEMPLATE_TOKEN_COUNT:
        return [], "too_early"
    
    offset = position - TEMPLATE_TOKEN_COUNT
    
    if offset > 6:
        return [], "too_far_in"
    
    answer_tok = gen_ids[TEMPLATE_TOKEN_COUNT] if TEMPLATE_TOKEN_COUNT < len(gen_ids) else None
    
    if answer_tok not in ANSWER_TOKEN_SET:
        return [], "complex_conclusion"
    
    if offset == 1:
        proposed = CLOSING_TOKENS[:draft_budget]
        if proposed:
            return proposed, "closing_no_period"
        return [], "closing_empty"
    
    # Check for period at position 37
    if offset >= 2 and len(gen_ids) > TEMPLATE_TOKEN_COUNT + 1:
        tok37 = gen_ids[TEMPLATE_TOKEN_COUNT + 1]
        if tok37 == PERIOD_TOKEN:
            if offset == 2:
                proposed = CLOSING_TOKENS[:draft_budget]
                return proposed, "closing_after_period"
    
    if offset <= len(CLOSING_TOKENS) + 1:
        idx = offset - 1
        if idx < len(CLOSING_TOKENS):
            proposed = CLOSING_TOKENS[idx:][:draft_budget]
            return proposed, f"closing_offset_{offset}"
    
    return [], "closing_none"


def _draft_the_answer_is_tokens(position, draft_budget, gen_ids, guess_letter):
    """
    Strategy 3: Token-ID-based "The answer is X." pattern for complex outputs.
    Detects how far into "The answer is" we are and drafts the rest.
    Works for both simple and complex conclusions.
    """
    if position <= TEMPLATE_TOKEN_COUNT:
        return [], "too_early"
    
    offset = position - TEMPLATE_TOKEN_COUNT
    
    # Only applicable when we're well into the conclusion (past answer letter area)
    if offset < 2:
        return [], "too_early_in_conclusion"
    
    # Check if the last few tokens match "The answer is" pattern
    # Look backwards from current position for the pattern
    n = len(gen_ids)
    
    # Case 1: Last token is period AND we can see "answer is <letter>" before it
    # → draft closing tag
    if n >= 4 and gen_ids[-1] == PERIOD_TOKEN:
        # Check if pattern is [..., answer_word, is, <answer_letter>, period]
        if gen_ids[-3] == ANSWER_WORD_TOKEN and gen_ids[-2] == IS_TOKEN:
            if gen_ids[-4] == THE_TOKEN or True:  # The might be further back
                # Just finished "is X." → draft closing
                if draft_budget >= 5:
                    return CLOSING_TOKENS[:draft_budget], "tai_period_closing"
    
    # Case 2: Last tokens are "The answer is " → draft answer letter + period + closing
    if n >= 3 and gen_ids[-3] == THE_TOKEN and gen_ids[-2] == ANSWER_WORD_TOKEN and gen_ids[-1] == IS_TOKEN:
        answer_tok = ANSWER_TOKEN_MAP.get(guess_letter, ANSWER_TOKEN_MAP['D'])
        suffix = [answer_tok, PERIOD_TOKEN] + CLOSING_TOKENS
        proposed = suffix[:draft_budget]
        return proposed, "tai_is_seen"
    
    # Case 3: Last tokens are "The answer" → draft " is <answer>." + closing
    if n >= 2 and gen_ids[-2] == THE_TOKEN and gen_ids[-1] == ANSWER_WORD_TOKEN:
        answer_tok = ANSWER_TOKEN_MAP.get(guess_letter, ANSWER_TOKEN_MAP['D'])
        suffix = [IS_TOKEN, answer_tok, PERIOD_TOKEN] + CLOSING_TOKENS
        proposed = suffix[:draft_budget]
        return proposed, "tai_answer_seen"
    
    # Case 4: Last token is "The" → draft "answer is <answer>." + closing
    if n >= 1 and gen_ids[-1] == THE_TOKEN:
        # Only if we're in complex territory (past position 40 or so)
        if offset >= 5:
            answer_tok = ANSWER_TOKEN_MAP.get(guess_letter, ANSWER_TOKEN_MAP['D'])
            suffix = [ANSWER_WORD_TOKEN, IS_TOKEN, answer_tok, PERIOD_TOKEN] + CLOSING_TOKENS
            proposed = suffix[:draft_budget]
            return proposed, "tai_the_seen"
    
    # Case 5: Period seen recently → next sentence starts with "The answer is"
    # Look for period in last 5 tokens
    lookback = min(8, n)
    for i in range(1, lookback):
        if gen_ids[-(i+1)] == PERIOD_TOKEN if n > i else False:
            break
    
    return [], "tai_not_detected"


def _template_prefix_match(decoded_text):
    template = STRUCTURAL_TEMPLATE
    min_len = min(len(decoded_text), len(template))
    match = 0
    for i in range(min_len):
        if decoded_text[i] == template[i]:
            match = i + 1
        else:
            break
    return match


def _draft_structural_template_text(decoded_text, word_budget, guess_letter, q_format):
    """
    Strategy 4 (fallback): Text-based structural template drafting.
    Used only when position-based tracking seems off.
    """
    match_len = _template_prefix_match(decoded_text)
    
    if match_len < 3 and decoded_text.strip():
        return "", "text_template_no_match"
    
    if match_len >= len(STRUCTURAL_TEMPLATE):
        return "", "text_template_conclusion_boundary"
    
    remaining = STRUCTURAL_TEMPLATE[match_len:]
    if not remaining:
        return "", "text_template_end"
    
    char_budget = int(word_budget * _CHARS_PER_TOKEN_EST) + 10
    
    remaining_chars = len(remaining)
    near_end = remaining_chars < TEMPLATE_END_GUESS_THRESHOLD * _CHARS_PER_TOKEN_EST
    
    if near_end and q_format in ("simple_mc", "hint_complex"):
        suffix = guess_letter + " " + CLOSING_TAG
        draft_text = remaining + suffix
        stage = "text_template_end_guess"
    else:
        draft_text = remaining[:char_budget]
        stage = f"text_template_{match_len // 30}"
    
    if len(draft_text) > char_budget + 80:
        draft_text = draft_text[:char_budget + 80]
    
    return draft_text, stage


def _draft_conclusion_closing_text(decoded_text, word_budget):
    """
    Strategy 5 (fallback): Text-based conclusion closing tag.
    """
    template = STRUCTURAL_TEMPLATE
    if len(decoded_text) <= len(template):
        return "", "closing_too_early"
    
    after = decoded_text[len(template):]
    
    if not after:
        return "", "conclusion_empty"
    
    first_char = after[0].upper()
    if first_char in ANSWER_LETTERS:
        rest = after[1:]
        if rest.startswith('.'):
            return " " + CLOSING_TAG, "closing_with_period"
        elif rest.startswith(' '):
            return CLOSING_TAG, "closing_space_next"
        else:
            return " " + CLOSING_TAG, "closing_after_letter"
    
    return "", "closing_no_match"


def _trigram_fallback(decoded_text, word_budget):
    if word_budget < 1:
        return ""
    stripped = decoded_text.rstrip()
    if not stripped:
        return ""
    words = stripped.split()
    if len(words) < 2:
        return ""
    last_two = ' '.join(words[-2:]).lower()
    last_two_clean = last_two.rstrip(string.punctuation)
    
    HIGH_CONF_TRIGRAMS = {
        "the answer": "is",
        "answer is": "the",
        "correct answer": "is",
        "difference between": "the",
    }
    
    if last_two_clean in HIGH_CONF_TRIGRAMS:
        return HIGH_CONF_TRIGRAMS[last_two_clean]
    return ""


def _bigram_fallback(decoded_text, word_budget):
    if word_budget < 1:
        return ""
    stripped = decoded_text.rstrip()
    if not stripped:
        return ""
    last_space = stripped.rfind(' ')
    if last_space < 0:
        last_word = stripped
    else:
        last_word = stripped[last_space + 1:]
    last_clean = last_word.lower().rstrip(string.punctuation)
    if not last_clean:
        return ""
    
    HIGH_CONF_BIGRAMS = {
        "answer": "is",
        "the": "answer",
        "is": "the",
        "correct": "answer",
        "therefore": "the",
        "thus": "the",
        "so": "the",
    }
    
    if last_clean in HIGH_CONF_BIGRAMS:
        return HIGH_CONF_BIGRAMS[last_clean]
    return ""


def _period_continuation(decoded_text):
    stripped = decoded_text.rstrip()
    if not stripped.endswith('.'):
        return ""
    if len(stripped) < 15:
        return ""
    tail = stripped[-300:] if len(stripped) > 300 else stripped
    has_answer_pattern = 'answer is' in tail.lower() or 'correct answer' in tail.lower()
    if has_answer_pattern:
        return "The"
    return ""


def _safe_int(value, default=32):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _truncate_text_to_tokens(text, max_tokens):
    max_chars = int(max_tokens * _CHARS_PER_TOKEN_EST) + 16
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# =====================================================================
# Main draft function
# =====================================================================

def draft(state):
    global _observer
    
    if not isinstance(state, dict):
        return []
    
    question = state.get("question", "")
    if not isinstance(question, str):
        question = ""
    
    decoded_text = state.get("decoded_text", "")
    if not isinstance(decoded_text, str):
        decoded_text = ""
    
    gen_ids = state.get("generated_token_ids", [])
    if not isinstance(gen_ids, (list, tuple)):
        gen_ids = []
    gen_ids = list(gen_ids)
    position = len(gen_ids)
    
    remaining_tokens = _safe_int(state.get("remaining_tokens"), 32)
    maximum_draft_tokens = _safe_int(state.get("maximum_draft_tokens"), 32)
    
    norm_state = {
        "question": question,
        "decoded_text": decoded_text,
        "generated_token_ids": gen_ids,
        "remaining_tokens": remaining_tokens,
        "maximum_draft_tokens": maximum_draft_tokens,
    }
    
    was_reset, reason, accepted = _observer.observe(norm_state)
    
    draft_budget = min(remaining_tokens, maximum_draft_tokens, 32)
    if draft_budget <= 0:
        return []
    
    word_budget = max(1, draft_budget)
    q_format = _observer.question_format
    guess_letter = _observer.guess_letter
    
    # === Primary strategy: token-ID-based structural prefix ===
    prefix_tokens, stage = _draft_prefix_tokens(position, draft_budget, guess_letter, q_format)
    if prefix_tokens:
        _observer.record_draft(len(prefix_tokens), strategy="prefix_tokens", stage=stage)
        return prefix_tokens
    
    # === Strategy 2: Token-ID-based closing tag (simple MC) ===
    closing_tokens, close_stage = _draft_closing_after_answer(position, draft_budget, gen_ids)
    if closing_tokens:
        _observer.record_draft(len(closing_tokens), strategy="closing_tokens", stage=close_stage)
        return closing_tokens
    
    # === Strategy 3: Token-ID-based "The answer is" pattern (complex) ===
    tai_tokens, tai_stage = _draft_the_answer_is_tokens(position, draft_budget, gen_ids, guess_letter)
    if tai_tokens:
        _observer.record_draft(len(tai_tokens), strategy="the_answer_is", stage=tai_stage)
        return tai_tokens
    
    # === Fallback: Text-based structural template ===
    if position < 5 and decoded_text and not decoded_text.startswith("<SUMMARY>"):
        template_draft, t_stage = _draft_structural_template_text(decoded_text, word_budget, guess_letter, q_format)
        if template_draft:
            template_draft = _truncate_text_to_tokens(template_draft, draft_budget)
            if template_draft:
                _observer.record_draft(len(template_draft), strategy="text_template", stage=t_stage)
                return {"text": template_draft}
    
    # === Strategy 4: Text-based conclusion closing ===
    closing_draft_text, ct_stage = _draft_conclusion_closing_text(decoded_text, word_budget)
    if closing_draft_text:
        closing_draft_text = _truncate_text_to_tokens(closing_draft_text, draft_budget)
        if closing_draft_text:
            _observer.record_draft(len(closing_draft_text), strategy="text_closing", stage=ct_stage)
            return {"text": closing_draft_text}
    
    # === Default: empty draft (ordinary AR step) ===
    # Single-token fallbacks removed — per-call verification overhead
    # makes 1-token drafts worse than baseline AR.
    return []
