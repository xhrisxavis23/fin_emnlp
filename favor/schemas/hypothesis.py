HYPOTHESIS_TOOL = {
    "type": "function",
    "function": {
        "name": "hypothesis_tool",
        "description": "Store behavioral hypotheses (structural/behavioral explanation).",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "description": "List of generated hypotheses (v1 expects exactly one).",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "hypothesis_id": {
                                "type": "string",
                                "description": "Unique identifier for the behavioral hypothesis (e.g., 'BH_MR_Exhaustion_1D_v1').",
                            },
                            "hypothesis_name": {
                                "type": "string",
                                "description": "Human-readable short name describing the behavior (not a strategy name).",
                            },
                            "behavioral_description": {
                                "type": "string",
                                "description": "Behavioral/structural reason why the price behavior appears (why, not just up/down).",
                            },
                            "horizon_days": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 30,
                                "default": 5,
                                "description": (
                                    "Expected holding period in trading days. "
                                    "This determines the forward return horizon (return_{n}d) for factor evaluation. "
                                ),
                            },
                        },
                        "required": [
                            "hypothesis_id",
                            "hypothesis_name",
                            "behavioral_description",
                            "horizon_days",
                        ],
                    },
                }
            },
            "required": ["hypotheses"],
        },
    },
}

# HYPOTHESIS_TOOL = {
#     "type": "function",
#     "function": {
#         "name": "hypothesis_tool",
#         "description": "Store trading hypotheses (observable market state + expected short-horizon price tendency + brief causal rationale).",
#         "parameters": {
#             "type": "object",
#             "additionalProperties": False,
#             "properties": {
#                 "hypotheses": {
#                     "type": "array",
#                     "minItems": 1,
#                     "maxItems": 1,
#                     "description": "List of generated trading hypotheses (v1 expects exactly one).",
#                     "items": {
#                         "type": "object",
#                         "additionalProperties": False,
#                         "properties": {
#                             "hypothesis_id": {
#                                 "type": "string",
#                                 "description": "Unique identifier for the trading hypothesis (e.g., 'TH_MR_LiqShock_3D_v1').",
#                             },
#                             "hypothesis_name": {
#                                 "type": "string",
#                                 "description": "Human-readable short name describing the hypothesis (not a strategy name).",
#                             },
#                             "trading_description": {
#                                 "type": "string",
#                                 "description": (
#                                     "A concise trading hypothesis (1–3 sentences) that includes: "
#                                     "(1) an OHLCV-observable market state, "
#                                     "(2) an expected short-horizon price tendency (mean reversion or continuation), "
#                                     "and (3) brief causal reasoning explaining WHY the tendency may occur. "
#                                     "Do NOT include indicators, thresholds, rules, or formulas."
#                                 ),
#                             },
#                             "horizon_days": {
#                                 "type": "integer",
#                                 "minimum": 1,
#                                 "maximum": 30,
#                                 "default": 5,
#                                 "description": (
#                                     "Forward return horizon in trading days used for evaluation/labeling "
#                                     "(not an explicit trading rule). Must be causally consistent with the hypothesis."
#                                 ),
#                             },
#                         },
#                         "required": [
#                             "hypothesis_id",
#                             "hypothesis_name",
#                             "trading_description",
#                             "horizon_days",
#                         ],
#                     },
#                 }
#             },
#             "required": ["hypotheses"],
#         },
#     },
# }
