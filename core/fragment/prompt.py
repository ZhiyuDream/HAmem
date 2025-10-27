def build_split_fragment_prompt(buffer, new_turn):
    """
    Build fragment splitting prompt.
    buffer: List of existing dialogue turns, each as dict with speaker and text.
    new_turn: New dialogue turn dict.
    Returns: prompt string.
    在线系统prompt
    """
    context = "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in buffer])
    new_line = f"{new_turn['speaker']}: {new_turn['text']}"
    prompt = (
        "Below is a continuous conversation. Please judge whether the last sentence belongs to a different topic or fragment compared to the previous content. Should it be split?\n"
        f"{context}\n{new_line}\n"
        "Please answer only: Yes or No"
    )
    return prompt

def build_batch_split_fragment_prompt(turns):
    """
    Build batch fragment splitting prompt for multiple turns.
    turns: List of dialogue turns, each as dict with speaker and text.
    Returns: prompt string.
    近线系统prompt
    """
    context = "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in turns])
    prompt = (
        "Below is a continuous conversation with multiple turns. Please analyze where to split this conversation into meaningful fragments.\n"
        f"Conversation:\n{context}\n\n"
        "Please identify the best split point. Return only a JSON object with the format:\n"
        '{"split_point": N, "reason": "brief explanation"}\n'
        "Where N is the number of turns that should be in the first fragment (0-based index, so N=8 means turns 0-7).\n"
        "If no split is needed, return split_point as -1.\n"
        "Example: {\"split_point\": 8, \"reason\": \"topic changed from work to personal life\"}"
    )
    return prompt
