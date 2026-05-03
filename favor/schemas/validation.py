DISTRIBUTION_JUDGMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "distribution_judgment_tool",
        "description": "Return PASS/FAIL judgment for whether a formula empirically supports an observation, based on descriptive OHLCV statistics.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["PASS", "FAIL"],
                    "description": "Final validation verdict"
                },
                "checks": {
                    "type": "object",
                    "description": "Mandatory interpretation checks (A–D)",
                    "properties": {
                        "location_involved": {
                            "type": "boolean",
                            "description": "At least one of DIR/POS/MAG shows consistent location shift (mean+median); VOL alone should not satisfy"
                        },
                        "tail_amplified": {
                            "type": "boolean",
                            "description": "At least one of DIR/POS/MAG shows meaningful tail evidence via q10/q90/kurtosis (can be expansion or compression depending on observation)"
                        },
                        "multi_stat_consistent": {
                            "type": "boolean",
                            "description": "Evidence uses a valid multi-statistic combination"
                        },
                        "no_contradiction": {
                            "type": "boolean",
                            "description": "No explicit contradiction with observation description (apply only when direction is clearly stated)"
                        }
                    },
                    "required": [
                        "location_involved",
                        "tail_amplified",
                        "multi_stat_consistent",
                        "no_contradiction"
                    ]
                },
                "primary_evidence": {
                    "type": "array",
                    "description": "Primary numeric evidence used in the judgment",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature": {
                                "type": "string",
                                "enum": ["DIR", "MAG", "POS", "VOL"],
                                "description": "Observed OHLCV-derived quantity"
                            },
                            "stat": {
                                "type": "string",
                                "enum": ["mean", "median", "q10", "q90", "kurtosis", "skewness"],
                                "description": "Descriptive statistic"
                            },
                            "pattern": {
                                "type": "string",
                                "enum": ["increasing", "decreasing"],
                                "description": "Direction of change across bins"
                            },
                            "bins": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Bin labels (must match evidence_json.bins)"
                            },
                            "numbers": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Statistic values for each bin"
                            }
                        },
                        "required": ["feature", "stat", "pattern", "bins", "numbers"]
                    }
                },
                "reasoning": {
                    "type": "string",
                    "description": "2–4 sentence explanation interpreting location, tail, and consistency"
                }
            },
            "required": ["verdict", "checks", "primary_evidence", "reasoning"]
        }
    }
}
