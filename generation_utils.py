def should_skip_page(text):
    if not text.strip():
        return True
    words = text.strip().split()
    if len(words) < 5:
        return True
    lines = text.strip().split('\n')
    if len(set(lines)) <= 2:
        return True
    return False

def build_prompt(text, lang):
    prompt = (
        f"Please translate the following resume or document into {lang}. "
        "Preserve its structure, section headers, formatting hints (like bullet points or line breaks), and professional tone. "
        "Do not repeat or summarize content. Only translate what's meaningful.\n\n---\n"
        f"{text}\n---"
    )
    return prompt

def clean_translation(text):
    lines = text.split('\n')
    seen, cleaned = set(), []
    for line in lines:
        l = line.strip()
        if l and l not in seen:
            cleaned.append(l)
            seen.add(l)
    return '\n'.join(cleaned)
