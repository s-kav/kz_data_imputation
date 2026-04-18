---
name: Debug Issue
description: Systematically debug issues using graph-powered code navigation
---
## Debug Issue
1. Use `semantic_search_nodes` to find code related to the issue.
2. Use `query_graph` with callers_of and callees_of to trace call chains.
3. Run `detect_changes` to check if recent changes caused the issue.
4. Use `get_impact_radius` on suspected files.
