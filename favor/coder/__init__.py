"""
coder/

This package contains optional code-generation backends.

NOTE:
Legacy `agent/code_agent.py` is the default backend used by the finance loops.
The modules under `coder/` can provide alternative implementations with
stronger retry/memory behavior, while keeping the same output contract
(`code_tool` schema: {codes:[{implementation, entry_point, ...}]}).
"""

