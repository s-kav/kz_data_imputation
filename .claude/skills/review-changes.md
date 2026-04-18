---
name: Review Changes
description: Perform a structured code review using change detection and impact
---
## Review Changes
1. Run `detect_changes` to get risk-scored change analysis.
2. Run `get_affected_flows` to find impacted execution paths.
3. For each high-risk function, run `query_graph` with pattern="tests_for".
4. Run `get_impact_radius` to understand the blast radius.
