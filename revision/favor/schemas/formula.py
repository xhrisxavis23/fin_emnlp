FORMULA_TOOL = {
    "type": "function",
    "function": {
        "name": "formula_tool",
        "description": "Store formula definitions generated from a given hypothesis.",
        "parameters": {
            "type": "object",
            "properties": {
                "hypothesis_id": {
                    "type": "string",
                    "description": "ID of the hypothesis from which these formulas are derived."
                },
                "formulas": {
                    "type": "array",
                    "description": "List of formula definitions.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the formula."
                            },
                            "definition": {
                                "type": "string",
                                "description": "Formula expression expressed strictly using metadata columns and allowed operators."
                            },
                            "description": {
                                "type": "string",
                                "description": "Brief explanation of how this formula reflects the hypothesis."
                            }
                        },
                        "required": ["name", "definition", "description"]
                    }
                }
            },
            "required": ["hypothesis_id", "formulas"]
        }
    }
}