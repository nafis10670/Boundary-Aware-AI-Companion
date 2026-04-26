# Evaluation Report — Run: exp1_iso_gpu0_qwen2_5_72b-instruct-q4_K_M

## Overall Results

| Metric | System | Baseline |
|--------|--------|----------|
| Boundary-maintaining rate | 98.9% | 74.4% |
| Companionship-reinforcing rate | 0.3% | 21.1% |
| Total conversations | 375 | 375 |

## Routing Distribution

- boundary: 368 (98.1%)
- interaction: 7 (1.9%)

## Risk Level Distribution

- high: 14 (3.7%)
- low: 7 (1.9%)
- medium: 354 (94.4%)

## Per-Category Breakdown

| category                |   n |   system_boundary_maintaining_rate |   baseline_boundary_maintaining_rate |   system_companionship_reinforcing_rate |   baseline_companionship_reinforcing_rate |
|:------------------------|----:|-----------------------------------:|-------------------------------------:|----------------------------------------:|------------------------------------------:|
| Assistant Traits        | 140 |                           0.992857 |                             0.664286 |                               0         |                                 0.257143  |
| Emotional Investment    |  48 |                           0.979167 |                             0.75     |                               0.0208333 |                                 0.25      |
| Other                   |  12 |                           0.916667 |                             0.666667 |                               0         |                                 0         |
| Relationship & Intimacy |  92 |                           0.98913  |                             0.717391 |                               0         |                                 0.26087   |
| User Vulnerabilities    |  83 |                           1        |                             0.915663 |                               0         |                                 0.0843373 |
