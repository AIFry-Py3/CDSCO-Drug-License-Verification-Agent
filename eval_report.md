# Evaluation
48 cases, thresholds approve>=90 ambiguous>=70, verifier=off
|            |   accuracy |   false_approval_rate |   approved_precision |   false_reject_rate |   safe_accuracy |   approved_recall |   escalation_rate |   hit_at_5 |   avg_tool_calls |   avg_latency_s |
|:-----------|-----------:|----------------------:|---------------------:|--------------------:|----------------:|------------------:|------------------:|-----------:|-----------------:|----------------:|
| fuzzy_only |      0.583 |                     0 |                    1 |               0     |           1     |             0.875 |             0.417 |        nan |             1    |           0.033 |
| rag_only   |      0.5   |                     0 |                  nan |               0.958 |           0.521 |             0     |             0.021 |          0 |             1    |           0.719 |
| agent      |      0.625 |                     0 |                    1 |               0     |           1     |             0.958 |             0.375 |        nan |             2.44 |           0.776 |

Accuracy by kind: {'fuzzy_only': {'clear_neg': 0.58, 'exact_pos': 0.92, 'near_miss': 0.0, 'noisy_pos': 0.83}, 'rag_only': {'clear_neg': 1.0, 'exact_pos': 0.0, 'near_miss': 1.0, 'noisy_pos': 0.0}, 'agent': {'clear_neg': 0.58, 'exact_pos': 1.0, 'near_miss': 0.0, 'noisy_pos': 0.92}}