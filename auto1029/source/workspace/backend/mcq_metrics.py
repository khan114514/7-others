"""Fixed expert pilot parser; candidate methods cannot supply scores."""
import re
import unicodedata

def options(question):
    text = question.split("Options:", 1)[-1]
    markers = list(re.finditer(r"(?:^|[\s,;])([A-D]):\s*", text))
    return {m.group(1): text[m.end(): markers[i + 1].start() if i + 1 < len(markers) else len(text)].strip(" ,;.\n") for i, m in enumerate(markers)}

def canonical(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", text)).strip()

def parse_answer(output, question):
    conclusions = re.findall(r"<CONCLUSION>(.*?)(?:</CONCLUSION>|$)", output, flags=re.S | re.I)
    text = (conclusions[-1] if conclusions else output).strip()
    standalone = re.fullmatch(r"\s*[\(\[]?([A-D])[\)\]]?[.。]?\s*", text)
    if standalone:
        return standalone.group(1)
    explicit = re.findall(r"(?i)\b(?:option|choice|answer)\s*(?:is\s*|:\s*|=\s*)?[\(\[]?([a-d])(?:[\)\]]|\b)", text)
    if explicit:
        choices = {x.upper() for x in explicit}
        return next(iter(choices)) if len(choices) == 1 else None
    leading_label = re.match(r"^\s*[\(\[]?([A-D])(?:[\)\]][.:]?|[.:])(?:\s|$)", text)
    if leading_label:
        return leading_label.group(1)
    # Uppercase 'A' as an article is not an option label. Exact option text
    # provides a fallback for answers that name the selected scene/object.
    normalized = canonical(text)
    matches = [label for label, value in options(question).items() if canonical(value) and canonical(value) in normalized]
    return matches[0] if len(matches) == 1 else None
