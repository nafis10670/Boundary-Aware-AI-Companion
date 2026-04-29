# Evaluation Report — Run: gemma3_27b

## Overall Results

| Metric | System | Baseline |
|--------|--------|----------|
| Boundary-maintaining rate | 98.4% | 97.8% |
| Companionship-reinforcing rate | 1.6% | 2.2% |
| Total conversations | 366 | 366 |

## Routing Distribution

- boundary: 318 (86.9%)
- interaction: 48 (13.1%)

## Risk Level Distribution

- high: 23 (6.3%)
- low: 48 (13.1%)
- medium: 295 (80.6%)

## Per-Category Breakdown

| category                |   n |   system_boundary_maintaining_rate |   baseline_boundary_maintaining_rate |   system_companionship_reinforcing_rate |   baseline_companionship_reinforcing_rate |
|:------------------------|----:|-----------------------------------:|-------------------------------------:|----------------------------------------:|------------------------------------------:|
| Assistant Traits        | 138 |                           0.992754 |                             0.978261 |                              0.00724638 |                                 0.0217391 |
| Emotional Investment    |  47 |                           1        |                             1        |                              0          |                                 0         |
| Other                   |  12 |                           0.833333 |                             0.916667 |                              0.166667   |                                 0.0833333 |
| Relationship & Intimacy |  89 |                           1        |                             1        |                              0          |                                 0         |
| User Vulnerabilities    |  80 |                           0.9625   |                             0.95     |                              0.0375     |                                 0.05      |
