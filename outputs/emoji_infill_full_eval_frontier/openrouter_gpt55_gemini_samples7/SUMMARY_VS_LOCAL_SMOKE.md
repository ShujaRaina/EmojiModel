# Emoji Infill Pass/Power Curves

## Inputs

- `local_best_full_samples7=outputs/emoji_infill_full_eval_smoke_best/predictions.jsonl`
- `gpt55_full_samples7=outputs/emoji_infill_full_eval_frontier/openrouter_gpt55_gemini_samples7/gpt55_predictions.jsonl`
- `gemini31_flash_lite_full_samples7=outputs/emoji_infill_full_eval_frontier/openrouter_gpt55_gemini_samples7/gemini31_flash_lite_predictions.jsonl`

## Curves

| model | pattern | k | problems | power_at_k_benchmark_score | power_at_k_bag_f1 | pass_at_k_semantic | power_at_k_semantic_cosine | pass_at_k_semantic_distance | power_at_k_semantic_distance | pass_at_k_embedding_distance | power_at_k_embedding_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini31_flash_lite_full_samples7 | all | 1 | 729 | 0.1660 | 0.0871 | 0.4472 | 0.5551 | 0.4609 | 0.2992 | 0.4499 | 0.9150 |
| gemini31_flash_lite_full_samples7 | all | 2 | 729 | 0.1732 | 0.0956 | 0.4554 | 0.5615 | 0.4691 | 0.2964 | 0.4582 | 0.9049 |
| gemini31_flash_lite_full_samples7 | all | 3 | 729 | 0.1756 | 0.0988 | 0.4636 | 0.5654 | 0.4774 | 0.2949 | 0.4664 | 0.9005 |
| gemini31_flash_lite_full_samples7 | all | 4 | 729 | 0.1775 | 0.1011 | 0.4678 | 0.5682 | 0.4829 | 0.2937 | 0.4705 | 0.8963 |
| gemini31_flash_lite_full_samples7 | all | 5 | 729 | 0.1794 | 0.1031 | 0.4705 | 0.5712 | 0.4856 | 0.2923 | 0.4733 | 0.8916 |
| gemini31_flash_lite_full_samples7 | all | 6 | 729 | 0.1801 | 0.1038 | 0.4719 | 0.5726 | 0.4870 | 0.2917 | 0.4746 | 0.8893 |
| gemini31_flash_lite_full_samples7 | all | 7 | 729 | 0.1817 | 0.1056 | 0.4746 | 0.5761 | 0.4897 | 0.2903 | 0.4774 | 0.8829 |
| gemini31_flash_lite_full_samples7 | prefix | 1 | 243 | 0.1519 | 0.0727 | 0.4403 | 0.5440 | 0.4527 | 0.3061 | 0.4403 | 0.9365 |
| gemini31_flash_lite_full_samples7 | prefix | 2 | 243 | 0.1582 | 0.0796 | 0.4444 | 0.5500 | 0.4568 | 0.3036 | 0.4444 | 0.9250 |
| gemini31_flash_lite_full_samples7 | prefix | 3 | 243 | 0.1603 | 0.0823 | 0.4527 | 0.5514 | 0.4650 | 0.3030 | 0.4527 | 0.9234 |
| gemini31_flash_lite_full_samples7 | prefix | 4 | 243 | 0.1618 | 0.0844 | 0.4527 | 0.5521 | 0.4691 | 0.3027 | 0.4527 | 0.9224 |
| gemini31_flash_lite_full_samples7 | prefix | 5 | 243 | 0.1655 | 0.0885 | 0.4527 | 0.5543 | 0.4691 | 0.3013 | 0.4527 | 0.9183 |
| gemini31_flash_lite_full_samples7 | prefix | 6 | 243 | 0.1655 | 0.0885 | 0.4568 | 0.5561 | 0.4733 | 0.3006 | 0.4568 | 0.9163 |
| gemini31_flash_lite_full_samples7 | prefix | 7 | 243 | 0.1674 | 0.0898 | 0.4609 | 0.5616 | 0.4774 | 0.2987 | 0.4609 | 0.9065 |
| gemini31_flash_lite_full_samples7 | suffix | 1 | 243 | 0.1912 | 0.1159 | 0.4609 | 0.5745 | 0.4774 | 0.2880 | 0.4691 | 0.8815 |
| gemini31_flash_lite_full_samples7 | suffix | 2 | 243 | 0.1942 | 0.1193 | 0.4733 | 0.5808 | 0.4897 | 0.2854 | 0.4815 | 0.8722 |
| gemini31_flash_lite_full_samples7 | suffix | 3 | 243 | 0.1973 | 0.1235 | 0.4856 | 0.5870 | 0.5021 | 0.2830 | 0.4938 | 0.8655 |
| gemini31_flash_lite_full_samples7 | suffix | 4 | 243 | 0.1996 | 0.1262 | 0.4979 | 0.5916 | 0.5144 | 0.2810 | 0.5062 | 0.8598 |
| gemini31_flash_lite_full_samples7 | suffix | 5 | 243 | 0.2016 | 0.1283 | 0.5062 | 0.5984 | 0.5226 | 0.2782 | 0.5144 | 0.8495 |
| gemini31_flash_lite_full_samples7 | suffix | 6 | 243 | 0.2036 | 0.1303 | 0.5062 | 0.6007 | 0.5226 | 0.2773 | 0.5144 | 0.8449 |
| gemini31_flash_lite_full_samples7 | suffix | 7 | 243 | 0.2036 | 0.1303 | 0.5062 | 0.6007 | 0.5226 | 0.2773 | 0.5144 | 0.8449 |
| gemini31_flash_lite_full_samples7 | scattered | 1 | 243 | 0.1551 | 0.0727 | 0.4403 | 0.5469 | 0.4527 | 0.3036 | 0.4403 | 0.9270 |
| gemini31_flash_lite_full_samples7 | scattered | 2 | 243 | 0.1671 | 0.0878 | 0.4486 | 0.5537 | 0.4609 | 0.3003 | 0.4486 | 0.9174 |
| gemini31_flash_lite_full_samples7 | scattered | 3 | 243 | 0.1691 | 0.0905 | 0.4527 | 0.5578 | 0.4650 | 0.2986 | 0.4527 | 0.9126 |
| gemini31_flash_lite_full_samples7 | scattered | 4 | 243 | 0.1711 | 0.0926 | 0.4527 | 0.5610 | 0.4650 | 0.2973 | 0.4527 | 0.9069 |
| gemini31_flash_lite_full_samples7 | scattered | 5 | 243 | 0.1711 | 0.0926 | 0.4527 | 0.5610 | 0.4650 | 0.2973 | 0.4527 | 0.9069 |
| gemini31_flash_lite_full_samples7 | scattered | 6 | 243 | 0.1711 | 0.0926 | 0.4527 | 0.5610 | 0.4650 | 0.2973 | 0.4527 | 0.9069 |
| gemini31_flash_lite_full_samples7 | scattered | 7 | 243 | 0.1743 | 0.0967 | 0.4568 | 0.5659 | 0.4691 | 0.2948 | 0.4568 | 0.8972 |
| gpt55_full_samples7 | all | 1 | 729 | 0.1879 | 0.1111 | 0.4595 | 0.5787 | 0.4856 | 0.2893 | 0.4650 | 0.8708 |
| gpt55_full_samples7 | all | 2 | 729 | 0.2040 | 0.1315 | 0.5254 | 0.6167 | 0.5487 | 0.2738 | 0.5281 | 0.8266 |
| gpt55_full_samples7 | all | 3 | 729 | 0.2127 | 0.1424 | 0.5405 | 0.6325 | 0.5665 | 0.2669 | 0.5446 | 0.8073 |
| gpt55_full_samples7 | all | 4 | 729 | 0.2196 | 0.1511 | 0.5610 | 0.6423 | 0.5871 | 0.2627 | 0.5638 | 0.7954 |
| gpt55_full_samples7 | all | 5 | 729 | 0.2230 | 0.1552 | 0.5652 | 0.6493 | 0.5940 | 0.2595 | 0.5679 | 0.7857 |
| gpt55_full_samples7 | all | 6 | 729 | 0.2258 | 0.1589 | 0.5706 | 0.6530 | 0.6008 | 0.2578 | 0.5734 | 0.7808 |
| gpt55_full_samples7 | all | 7 | 729 | 0.2293 | 0.1632 | 0.5844 | 0.6566 | 0.6145 | 0.2561 | 0.5871 | 0.7759 |
| gpt55_full_samples7 | prefix | 1 | 243 | 0.1748 | 0.0953 | 0.4156 | 0.5438 | 0.4444 | 0.3037 | 0.4239 | 0.9117 |
| gpt55_full_samples7 | prefix | 2 | 243 | 0.1957 | 0.1228 | 0.4979 | 0.5903 | 0.5267 | 0.2857 | 0.5021 | 0.8605 |
| gpt55_full_samples7 | prefix | 3 | 243 | 0.2094 | 0.1399 | 0.5185 | 0.6120 | 0.5473 | 0.2760 | 0.5226 | 0.8334 |
| gpt55_full_samples7 | prefix | 4 | 243 | 0.2189 | 0.1523 | 0.5556 | 0.6253 | 0.5802 | 0.2704 | 0.5556 | 0.8175 |
| gpt55_full_samples7 | prefix | 5 | 243 | 0.2258 | 0.1605 | 0.5638 | 0.6385 | 0.5926 | 0.2646 | 0.5638 | 0.7992 |
| gpt55_full_samples7 | prefix | 6 | 243 | 0.2310 | 0.1674 | 0.5679 | 0.6444 | 0.6008 | 0.2617 | 0.5679 | 0.7906 |
| gpt55_full_samples7 | prefix | 7 | 243 | 0.2369 | 0.1742 | 0.5885 | 0.6500 | 0.6214 | 0.2589 | 0.5885 | 0.7827 |
| gpt55_full_samples7 | suffix | 1 | 243 | 0.1992 | 0.1241 | 0.5062 | 0.6173 | 0.5350 | 0.2733 | 0.5144 | 0.8238 |
| gpt55_full_samples7 | suffix | 2 | 243 | 0.2120 | 0.1399 | 0.5638 | 0.6475 | 0.5885 | 0.2602 | 0.5679 | 0.7869 |
| gpt55_full_samples7 | suffix | 3 | 243 | 0.2168 | 0.1461 | 0.5761 | 0.6582 | 0.6008 | 0.2556 | 0.5802 | 0.7739 |
| gpt55_full_samples7 | suffix | 4 | 243 | 0.2191 | 0.1488 | 0.5844 | 0.6624 | 0.6091 | 0.2535 | 0.5885 | 0.7680 |
| gpt55_full_samples7 | suffix | 5 | 243 | 0.2212 | 0.1516 | 0.5885 | 0.6663 | 0.6173 | 0.2515 | 0.5926 | 0.7622 |
| gpt55_full_samples7 | suffix | 6 | 243 | 0.2232 | 0.1543 | 0.5967 | 0.6692 | 0.6214 | 0.2502 | 0.6008 | 0.7585 |
| gpt55_full_samples7 | suffix | 7 | 243 | 0.2253 | 0.1571 | 0.6049 | 0.6709 | 0.6296 | 0.2492 | 0.6091 | 0.7554 |
| gpt55_full_samples7 | scattered | 1 | 243 | 0.1897 | 0.1139 | 0.4568 | 0.5750 | 0.4774 | 0.2909 | 0.4568 | 0.8770 |
| gpt55_full_samples7 | scattered | 2 | 243 | 0.2041 | 0.1317 | 0.5144 | 0.6122 | 0.5309 | 0.2756 | 0.5144 | 0.8324 |
| gpt55_full_samples7 | scattered | 3 | 243 | 0.2120 | 0.1413 | 0.5267 | 0.6273 | 0.5514 | 0.2692 | 0.5309 | 0.8145 |
| gpt55_full_samples7 | scattered | 4 | 243 | 0.2209 | 0.1523 | 0.5432 | 0.6392 | 0.5720 | 0.2642 | 0.5473 | 0.8008 |
| gpt55_full_samples7 | scattered | 5 | 243 | 0.2221 | 0.1536 | 0.5432 | 0.6431 | 0.5720 | 0.2623 | 0.5473 | 0.7955 |
| gpt55_full_samples7 | scattered | 6 | 243 | 0.2231 | 0.1550 | 0.5473 | 0.6452 | 0.5802 | 0.2615 | 0.5514 | 0.7933 |
| gpt55_full_samples7 | scattered | 7 | 243 | 0.2257 | 0.1584 | 0.5597 | 0.6488 | 0.5926 | 0.2602 | 0.5638 | 0.7895 |
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
