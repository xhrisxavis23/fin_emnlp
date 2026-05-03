BEHAVIORAL_FORMULA_TOOL = {
    "type": "function",
    "function": {
        "name": "behavioral_formula_tool",
        "description": (
            "Store observation-driven alpha bundle derived from a behavioral hypothesis: "
            "observation descriptions -> evidence formulas."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hypothesis_id": {
                    "type": "string",
                    "description": "Behavioral hypothesis id that this bundle is derived from.",
                },
                "observation_descriptions": {
                    "type": "array",
                    "minItems": 1,
                    "description": "Decomposition of the behavioral hypothesis into observable phenomena descriptions.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "observation_id": {
                                "type": "string",
                                "description": "Links this item to a specific observation from the Observation Plan",
                            },
                            "description": {
                                "type": "string",
                                "description": "Observable phenomenon description: 'Mechanism X is visible...'",
                            },
                        },
                        "required": ["observation_id", "description"],
                    },
                },
                "formulas": {
                    "type": "array",
                    "minItems": 1,
                    "description": (
                        "Continuous evidence alpha formulas in the project DSL. "
                        "These are OBSERVATION signals (not directly 'predict return' claims)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Unique name for this formula (e.g., 'formula001', 'formula001_v1'). Required.",
                            },
                            "kind": {
                                "type": "string",
                                "enum": ["evidence"],
                                "description": "Evidence formula.",
                            },
                            "observation_id": {
                                "type": ["string", "null"],
                                "description": "Links this evidence formula to its source observation (required).",
                            },
                            "definition": {
                                "type": "string",
                                "description": (
                                    "Formula expression in DSL using ONLY allowed columns and allowed operators/functions; "
                                    "must be continuous numeric (not boolean)."
                                ),
                            },
                            "polarity": {
                                "type": "string",
                                "enum": ["higher_is_more_true", "lower_is_more_true"],
                                "description": "Interpretation direction for the behavioral claim (fixed by meaning, not IC).",
                            },
                            "description": {
                                "type": "string",
                                "description": "Explain what observable evidence this formula measures and how to interpret it.",
                            },
                        },
                        "required": ["name", "kind", "definition", "polarity", "description"],
                        "additionalProperties": False,
                    },
                },
                "notes": {
                    "type": ["string", "null"],
                    "description": "Optional notes for traceability (no performance metrics).",
                },
            },
            "required": [
                "hypothesis_id",
                "observation_descriptions",
                "formulas",
            ],
        },
    },
}