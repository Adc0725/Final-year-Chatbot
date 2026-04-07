def check_crisis(text):
    crisis_keywords = [
        "suicide",
        "kill myself",
        "end my life",
        "self harm"
    ]

    text = text.lower()

    for word in crisis_keywords:
        if word in text:
            return True

    return False