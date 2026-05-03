from typing import List, Dict, Any, Optional

OBSERVATION_TOOL = {
    "type": "function",
    "function": {
        "name": "observation_plan_tool",
        "description": "Submit a structured observation plan describing directly observable market states that characterize when a hypothesis is applicable.",
        "parameters": {
            "type": "object",
            "properties": {
                "hypothesis_id": {
                    "type": "string",
                    "description": "ID of the hypothesis being operationalized."
                },
                "observations": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 6,
                    "description": "List of directly observable market states that may co-occur within the same short window when the hypothesis is present (setup + transition evidence)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "observation_id": {
                                "type": "string",
                                "description": "Unique identifier for this observation (e.g., 'obs_price_decline')"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of a directly observable market state (setup or transition evidence) definable from OHLCV, without referencing causes, valuations, participant intentions, or outcomes."
                            }
                        },
                        "required": ["observation_id", "description"]
                    }
                }
            },
            "required": ["hypothesis_id", "observations"]
        }
    }
}
