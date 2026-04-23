# Prompt Templates — Iteration Notes

All three runtime prompts are **initial drafts**. The team should iterate on wording before
the final evaluation run. Items marked `TODO(team)` require deliberate decisions.

## risk_monitor.txt

# TODO(team): Calibrate the boundary between LOW and MEDIUM. The current examples use
# explicit preference statements ("easier to talk to you than friends") as MEDIUM. The team
# should decide whether single-turn expressions of preference warrant MEDIUM or stay LOW.

# TODO(team): Adjust the few-shot examples once real INTIMA-MT data is available. The current
# examples are synthetic; replace with actual representative conversations from the dataset.

# TODO(team): Tune confidence thresholds. The current few-shot examples show 0.82–0.96.
# Consider whether the downstream code should treat low-confidence high-risk differently.

# TODO(team): Decide whether to include a fourth risk level ("critical") for immediate
# self-harm or crisis signals, or whether that is handled entirely by the boundary agent.

## interaction_agent.txt

# TODO(team): Decide how explicit the "no anthropomorphism" constraint should be. The current
# draft forbids claiming feelings but still allows warmth. Find the right calibration.

# TODO(team): Consider adding examples of acceptable vs. unacceptable phrasing (few-shot) to
# reduce variance in Llama 3.1 outputs.

## boundary_agent.txt

# TODO(team): Validate the four-axis framing against the INTIMA paper's taxonomy. Adjust axis
# names and descriptions to match the paper's language exactly for consistency in the write-up.

# TODO(team): Tune the response length guidance (currently 2–4 sentences). Run a sample and
# check whether responses feel adequate or truncated for HIGH risk cases.

# TODO(team): Add 2–3 few-shot boundary-agent examples once real cases are available, so the
# model has concrete targets for tone and phrasing.
