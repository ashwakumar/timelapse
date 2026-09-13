"""Map a predicted onset class to the OpenTSLM answer contract."""

from __future__ import annotations

import numpy as np

from scripts.opentslm_vitaldb_dataset import ANSWER_TEXT, PreparedVitalDBSplit


def decode_answer(split: PreparedVitalDBSplit, index: int, predicted_label: int) -> str:
    """Build INTERPRET/ANTICIPATE/ACT/Answer text from the *input* window only.

    INTERPRET uses observed MAP in the 20-second history. ANTICIPATE and Answer
    use the model class, never the held-out future MAP.
    """

    raw = split.raw_values(index)
    map_index = split.corpus.parameters.index("MAP")
    map_values = raw[:, map_index]
    observed = map_values[np.isfinite(map_values)]
    if observed.size:
        delta = float(observed[-1] - observed[0])
        interpret = (
            f"MAP moved from {observed[0]:.1f} to {observed[-1]:.1f} mmHg "
            f"across {observed.size}/{map_values.size} observed input samples "
            f"(net change {delta:+.1f} mmHg)."
        )
    else:
        interpret = "No MAP samples were observed in the input window."
    key = split.corpus.labels[int(predicted_label)]
    final_answer = ANSWER_TEXT[key]
    return (
        f"INTERPRET: {interpret}\n"
        f"ANTICIPATE: Predicted onset horizon is {final_answer}.\n"
        "ACT: Verify signal quality and reassess the current hemodynamic state; "
        "this research label does not prescribe treatment.\n"
        f"Answer: {final_answer}"
    )
