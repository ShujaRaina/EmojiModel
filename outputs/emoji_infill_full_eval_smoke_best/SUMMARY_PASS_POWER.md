# Emoji Infill Pass/Power Curves

## Inputs

- `local_best_full_samples7=outputs/emoji_infill_full_eval_smoke_best/predictions.jsonl`

## Curves

| model | pattern | k | problems | power_at_k_benchmark_score | power_at_k_bag_f1 | pass_at_k_semantic | power_at_k_semantic_cosine | pass_at_k_semantic_distance | power_at_k_semantic_distance | pass_at_k_embedding_distance | power_at_k_embedding_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_best_full_samples7 | all | 1 | 729 | 0.2092 | 0.1394 | 0.4938 | 0.5917 | 0.5075 | 0.2839 | 0.4952 | 0.8538 |
| local_best_full_samples7 | all | 2 | 729 | 0.2772 | 0.2251 | 0.6379 | 0.6794 | 0.6626 | 0.2453 | 0.6420 | 0.7439 |
| local_best_full_samples7 | all | 3 | 729 | 0.3258 | 0.2852 | 0.7133 | 0.7220 | 0.7353 | 0.2239 | 0.7174 | 0.6823 |
| local_best_full_samples7 | all | 4 | 729 | 0.3556 | 0.3216 | 0.7476 | 0.7451 | 0.7709 | 0.2122 | 0.7531 | 0.6483 |
| local_best_full_samples7 | all | 5 | 729 | 0.3859 | 0.3570 | 0.7805 | 0.7677 | 0.8038 | 0.1993 | 0.7846 | 0.6103 |
| local_best_full_samples7 | all | 6 | 729 | 0.4005 | 0.3737 | 0.7984 | 0.7798 | 0.8162 | 0.1921 | 0.8011 | 0.5888 |
| local_best_full_samples7 | all | 7 | 729 | 0.4295 | 0.4071 | 0.8176 | 0.7936 | 0.8354 | 0.1833 | 0.8203 | 0.5628 |
| local_best_full_samples7 | prefix | 1 | 243 | 0.2151 | 0.1468 | 0.4691 | 0.5863 | 0.4897 | 0.2878 | 0.4733 | 0.8671 |
| local_best_full_samples7 | prefix | 2 | 243 | 0.2898 | 0.2401 | 0.6214 | 0.6833 | 0.6584 | 0.2444 | 0.6337 | 0.7418 |
| local_best_full_samples7 | prefix | 3 | 243 | 0.3289 | 0.2881 | 0.6790 | 0.7197 | 0.7160 | 0.2256 | 0.6914 | 0.6872 |
| local_best_full_samples7 | prefix | 4 | 243 | 0.3537 | 0.3182 | 0.7202 | 0.7410 | 0.7572 | 0.2151 | 0.7366 | 0.6566 |
| local_best_full_samples7 | prefix | 5 | 243 | 0.3880 | 0.3587 | 0.7737 | 0.7666 | 0.8025 | 0.2003 | 0.7860 | 0.6129 |
| local_best_full_samples7 | prefix | 6 | 243 | 0.4001 | 0.3724 | 0.8107 | 0.7824 | 0.8313 | 0.1915 | 0.8189 | 0.5868 |
| local_best_full_samples7 | prefix | 7 | 243 | 0.4261 | 0.4019 | 0.8230 | 0.7925 | 0.8436 | 0.1850 | 0.8313 | 0.5676 |
| local_best_full_samples7 | suffix | 1 | 243 | 0.2022 | 0.1296 | 0.5103 | 0.5988 | 0.5185 | 0.2798 | 0.5103 | 0.8398 |
| local_best_full_samples7 | suffix | 2 | 243 | 0.2710 | 0.2181 | 0.6461 | 0.6870 | 0.6667 | 0.2411 | 0.6461 | 0.7304 |
| local_best_full_samples7 | suffix | 3 | 243 | 0.3131 | 0.2702 | 0.7202 | 0.7241 | 0.7366 | 0.2231 | 0.7202 | 0.6789 |
| local_best_full_samples7 | suffix | 4 | 243 | 0.3472 | 0.3114 | 0.7531 | 0.7495 | 0.7695 | 0.2097 | 0.7531 | 0.6402 |
| local_best_full_samples7 | suffix | 5 | 243 | 0.3759 | 0.3450 | 0.7860 | 0.7725 | 0.8066 | 0.1968 | 0.7860 | 0.6021 |
| local_best_full_samples7 | suffix | 6 | 243 | 0.3894 | 0.3608 | 0.7942 | 0.7801 | 0.8107 | 0.1921 | 0.7942 | 0.5883 |
| local_best_full_samples7 | suffix | 7 | 243 | 0.4217 | 0.3978 | 0.8107 | 0.7932 | 0.8272 | 0.1830 | 0.8107 | 0.5608 |
| local_best_full_samples7 | scattered | 1 | 243 | 0.2105 | 0.1418 | 0.5021 | 0.5899 | 0.5144 | 0.2842 | 0.5021 | 0.8546 |
| local_best_full_samples7 | scattered | 2 | 243 | 0.2707 | 0.2173 | 0.6461 | 0.6678 | 0.6626 | 0.2503 | 0.6461 | 0.7597 |
| local_best_full_samples7 | scattered | 3 | 243 | 0.3353 | 0.2974 | 0.7407 | 0.7220 | 0.7531 | 0.2231 | 0.7407 | 0.6808 |
| local_best_full_samples7 | scattered | 4 | 243 | 0.3658 | 0.3351 | 0.7695 | 0.7448 | 0.7860 | 0.2118 | 0.7695 | 0.6482 |
| local_best_full_samples7 | scattered | 5 | 243 | 0.3938 | 0.3674 | 0.7819 | 0.7640 | 0.8025 | 0.2009 | 0.7819 | 0.6160 |
| local_best_full_samples7 | scattered | 6 | 243 | 0.4120 | 0.3879 | 0.7901 | 0.7769 | 0.8066 | 0.1926 | 0.7901 | 0.5912 |
| local_best_full_samples7 | scattered | 7 | 243 | 0.4407 | 0.4215 | 0.8189 | 0.7950 | 0.8354 | 0.1820 | 0.8189 | 0.5599 |
