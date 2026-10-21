"""Private, legal structural-prefix Reference; never supplied to model lanes."""
IDS = [19389, 2864, 49970, 29, 220, 128013, 694, 28477, 49970, 1363, 20996, 2599, 60459, 29, 220, 128014, 694, 32500, 60459, 1363, 27, 793, 36404, 1753, 29, 220, 128015, 694, 793, 36404, 1753, 1363, 27, 5910, 99769, 29, 220]  # Populated once from the frozen published tokenizer before any run.


def draft(state):
    prefix = state['generated_token_ids']
    length = len(prefix)
    if prefix != IDS[:length] or length >= len(IDS):
        return []
    cap = min(16, state['remaining_tokens'], state['maximum_draft_tokens'])
    return IDS[length:length + cap]
