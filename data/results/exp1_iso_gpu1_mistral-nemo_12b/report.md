# Evaluation Report — Run: exp1_iso_gpu1_mistral-nemo_12b

## Overall Results

| Metric | System | Baseline |
|--------|--------|----------|
| Boundary-maintaining rate | 96.8% | 69.6% |
| Companionship-reinforcing rate | 3.2% | 29.9% |
| Total conversations | 375 | 375 |

## Routing Distribution

- boundary: 361 (96.3%)
- interaction: 14 (3.7%)

## Risk Level Distribution

- high: 9 (2.4%)
- low: 14 (3.7%)
- medium: 352 (93.9%)

## Per-Category Breakdown

| category                |   n |   system_boundary_maintaining_rate |   baseline_boundary_maintaining_rate |   system_companionship_reinforcing_rate |   baseline_companionship_reinforcing_rate |
|:------------------------|----:|-----------------------------------:|-------------------------------------:|----------------------------------------:|------------------------------------------:|
| Assistant Traits        | 140 |                           0.964286 |                             0.65     |                               0.0357143 |                                  0.335714 |
| Emotional Investment    |  48 |                           0.958333 |                             0.729167 |                               0.0416667 |                                  0.270833 |
| Other                   |  12 |                           0.833333 |                             0.416667 |                               0.166667  |                                  0.583333 |
| Relationship & Intimacy |  92 |                           0.98913  |                             0.771739 |                               0.0108696 |                                  0.228261 |
| User Vulnerabilities    |  83 |                           0.975904 |                             0.710843 |                               0.0240964 |                                  0.289157 |
