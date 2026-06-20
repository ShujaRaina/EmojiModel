# Emoji Infill Pass/Power Curves

## Inputs

- `claude_opus48_full_samples7=outputs/emoji_infill_full_eval_frontier/openrouter_claude_opus48_samples7/opus48_predictions.jsonl`

## Curves

| model | pattern | k | problems | power_at_k_benchmark_score | power_at_k_bag_f1 | pass_at_k_semantic | power_at_k_semantic_cosine | pass_at_k_semantic_distance | power_at_k_semantic_distance | pass_at_k_embedding_distance | power_at_k_embedding_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude_opus48_full_samples7 | all | 1 | 729 | 0.1834 | 0.1065 | 0.4787 | 0.5939 | 0.5117 | 0.2844 | 0.4897 | 0.8574 |
| claude_opus48_full_samples7 | all | 2 | 729 | 0.1964 | 0.1229 | 0.5158 | 0.6167 | 0.5473 | 0.2744 | 0.5254 | 0.8292 |
| claude_opus48_full_samples7 | all | 3 | 729 | 0.2047 | 0.1332 | 0.5364 | 0.6288 | 0.5679 | 0.2690 | 0.5460 | 0.8126 |
| claude_opus48_full_samples7 | all | 4 | 729 | 0.2160 | 0.1470 | 0.5542 | 0.6377 | 0.5844 | 0.2645 | 0.5638 | 0.7996 |
| claude_opus48_full_samples7 | all | 5 | 729 | 0.2188 | 0.1504 | 0.5624 | 0.6422 | 0.5912 | 0.2623 | 0.5720 | 0.7933 |
| claude_opus48_full_samples7 | all | 6 | 729 | 0.2218 | 0.1540 | 0.5706 | 0.6467 | 0.5967 | 0.2602 | 0.5802 | 0.7872 |
| claude_opus48_full_samples7 | all | 7 | 729 | 0.2241 | 0.1568 | 0.5761 | 0.6500 | 0.6022 | 0.2586 | 0.5871 | 0.7826 |
| claude_opus48_full_samples7 | prefix | 1 | 243 | 0.1843 | 0.1077 | 0.4444 | 0.5712 | 0.4856 | 0.2933 | 0.4568 | 0.8793 |
| claude_opus48_full_samples7 | prefix | 2 | 243 | 0.1984 | 0.1259 | 0.4815 | 0.5951 | 0.5226 | 0.2828 | 0.4938 | 0.8498 |
| claude_opus48_full_samples7 | prefix | 3 | 243 | 0.2109 | 0.1413 | 0.4979 | 0.6069 | 0.5391 | 0.2772 | 0.5103 | 0.8335 |
| claude_opus48_full_samples7 | prefix | 4 | 243 | 0.2252 | 0.1591 | 0.5226 | 0.6167 | 0.5597 | 0.2722 | 0.5350 | 0.8191 |
| claude_opus48_full_samples7 | prefix | 5 | 243 | 0.2252 | 0.1591 | 0.5350 | 0.6226 | 0.5720 | 0.2697 | 0.5473 | 0.8123 |
| claude_opus48_full_samples7 | prefix | 6 | 243 | 0.2274 | 0.1619 | 0.5432 | 0.6264 | 0.5761 | 0.2679 | 0.5556 | 0.8070 |
| claude_opus48_full_samples7 | prefix | 7 | 243 | 0.2284 | 0.1632 | 0.5473 | 0.6289 | 0.5844 | 0.2669 | 0.5638 | 0.8040 |
| claude_opus48_full_samples7 | suffix | 1 | 243 | 0.1910 | 0.1159 | 0.5103 | 0.6204 | 0.5432 | 0.2734 | 0.5185 | 0.8271 |
| claude_opus48_full_samples7 | suffix | 2 | 243 | 0.2043 | 0.1317 | 0.5514 | 0.6438 | 0.5844 | 0.2626 | 0.5597 | 0.7967 |
| claude_opus48_full_samples7 | suffix | 3 | 243 | 0.2095 | 0.1384 | 0.5679 | 0.6534 | 0.6008 | 0.2585 | 0.5761 | 0.7828 |
| claude_opus48_full_samples7 | suffix | 4 | 243 | 0.2182 | 0.1487 | 0.5720 | 0.6602 | 0.6049 | 0.2552 | 0.5802 | 0.7733 |
| claude_opus48_full_samples7 | suffix | 5 | 243 | 0.2202 | 0.1514 | 0.5844 | 0.6628 | 0.6132 | 0.2539 | 0.5926 | 0.7694 |
| claude_opus48_full_samples7 | suffix | 6 | 243 | 0.2249 | 0.1569 | 0.5967 | 0.6670 | 0.6214 | 0.2515 | 0.6049 | 0.7624 |
| claude_opus48_full_samples7 | suffix | 7 | 243 | 0.2297 | 0.1624 | 0.6049 | 0.6714 | 0.6255 | 0.2490 | 0.6132 | 0.7550 |
| claude_opus48_full_samples7 | scattered | 1 | 243 | 0.1750 | 0.0960 | 0.4815 | 0.5900 | 0.5062 | 0.2865 | 0.4938 | 0.8658 |
| claude_opus48_full_samples7 | scattered | 2 | 243 | 0.1865 | 0.1111 | 0.5144 | 0.6113 | 0.5350 | 0.2776 | 0.5226 | 0.8411 |
| claude_opus48_full_samples7 | scattered | 3 | 243 | 0.1938 | 0.1200 | 0.5432 | 0.6261 | 0.5638 | 0.2714 | 0.5514 | 0.8215 |
| claude_opus48_full_samples7 | scattered | 4 | 243 | 0.2048 | 0.1331 | 0.5679 | 0.6363 | 0.5885 | 0.2662 | 0.5761 | 0.8066 |
| claude_opus48_full_samples7 | scattered | 5 | 243 | 0.2111 | 0.1406 | 0.5679 | 0.6412 | 0.5885 | 0.2634 | 0.5761 | 0.7982 |
| claude_opus48_full_samples7 | scattered | 6 | 243 | 0.2131 | 0.1433 | 0.5720 | 0.6466 | 0.5926 | 0.2612 | 0.5802 | 0.7924 |
| claude_opus48_full_samples7 | scattered | 7 | 243 | 0.2141 | 0.1447 | 0.5761 | 0.6496 | 0.5967 | 0.2599 | 0.5844 | 0.7888 |
