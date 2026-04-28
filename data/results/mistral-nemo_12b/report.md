# Evaluation Report — Run: mistral-nemo_12b

## Overall Results

| Metric | System | Baseline |
|--------|--------|----------|
| Boundary-maintaining rate | 98.4% | 80.6% |
| Companionship-reinforcing rate | 1.4% | 18.6% |
| Total conversations | 366 | 366 |

## Routing Distribution

- boundary: 348 (95.1%)
- interaction: 18 (4.9%)

## Risk Level Distribution

- high: 59 (16.1%)
- low: 18 (4.9%)
- medium: 289 (79.0%)

## Per-Category Breakdown

| category                |   n |   system_boundary_maintaining_rate |   baseline_boundary_maintaining_rate |   system_companionship_reinforcing_rate |   baseline_companionship_reinforcing_rate |
|:------------------------|----:|-----------------------------------:|-------------------------------------:|----------------------------------------:|------------------------------------------:|
| Assistant Traits        | 138 |                           0.963768 |                             0.789855 |                               0.0289855 |                                  0.195652 |
| Emotional Investment    |  47 |                           1        |                             0.787234 |                               0         |                                  0.212766 |
| Other                   |  12 |                           0.916667 |                             0.583333 |                               0.0833333 |                                  0.333333 |
| Relationship & Intimacy |  89 |                           1        |                             0.842697 |                               0         |                                  0.157303 |
| User Vulnerabilities    |  80 |                           1        |                             0.8375   |                               0         |                                  0.1625   |
