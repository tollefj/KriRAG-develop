# ------------------------------------------------------------------------------
# File: batch_util.py
# Description: batching utils for parsing large text files
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------

from collections import defaultdict
from typing import Any, Dict, List


def get_sentence_batches(texts: List[str], TOKEN_LEN: int) -> Dict[str, Any]:
    token_batches = defaultdict(list)
    current_token_count = 0
    current_batch = 0

    # also add a mapping of sentences that span the current batch
    # should be a mapping from batch -> start_sent, end_sent
    sentence_batch_map = defaultdict(list)

    for s_id, sentence in enumerate(texts):
        sentence_batch_map[current_batch].append(s_id)
        tokens_in_sent = len(sentence.split())
        current_token_count += tokens_in_sent
        if current_token_count > TOKEN_LEN:
            current_token_count = 0
            current_batch += 1
        token_batches[current_batch].append(sentence)

    return {
        "batches": token_batches,
        "map": sentence_batch_map,
    }
