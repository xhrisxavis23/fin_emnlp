# Validation Stage Analysis (Stage2) — Summary

- target_n_quantiles: 20

## ours
- n_total: 54
- pass_rate: 0.556 (30/54)
- |rho|(DIR_mean): mean=0.5869243816612237, median=0.7157894736842105
- |rho|(POS_mean): mean=0.4497784866205919, median=0.21052631578947367
- |rho|(MAG_mean): mean=0.4642367703771213, median=0.42857142857142855
- fail_modes: {'non-monotonic/unstable': 24}
- exemplar_pass: `results/formulas/formula010_formula010_07037812/07037812a7504b7b9d26cb1dd581ef55`
- exemplar_reference: `results/formulas/formula010_formula010_07037812/07037812a7504b7b9d26cb1dd581ef55`
- exemplar_fail: `results/formulas/formula005_v1_formula005_v1_cef41b31/cef41b312cd848a09d20192687c01ff2`

## alphaagent
- n_total: 17
- pass_rate: 0.588 (10/17)
- |rho|(DIR_mean): mean=0.7342688271480842, median=0.9878787878787879
- |rho|(POS_mean): mean=0.6309436693337622, median=0.7212121212121212
- |rho|(MAG_mean): mean=0.22533338694329408, median=0.21503759398496242
- fail_modes: {'non-monotonic/unstable': 7}
- exemplar_pass: `results/alphaagent/Deviation_Below_WMA_10D`
- exemplar_reference: `results/alphaagent/Price_Volatility_Momentum_Factor_5D`
- exemplar_fail: `results/alphaagent/Downward_Deviation_Recovery_Factor_10D`

## alpha101
- n_total: 51
- pass_rate: 0.000 (0/51)
- |rho|(DIR_mean): mean=0.9487306755792746, median=0.9458646616541353
- |rho|(POS_mean): mean=0.35011438191827243, median=0.4616541353383459
- |rho|(MAG_mean): mean=0.8858213151082929, median=0.8902255639097745
- fail_modes: {'non-monotonic/unstable': 51}
- exemplar_pass: ``
- exemplar_reference: `results/alpha101/alpha002/bdae212315924e869f30f6664b6314a5`
- exemplar_fail: `results/alpha101/alpha002/96345ef591e8436790c80a59996ed5c1`
