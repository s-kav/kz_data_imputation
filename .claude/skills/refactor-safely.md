---
name: Refactor Safely
description: Plan and execute safe refactoring using dependency analysis
---
## Refactor Safely
1. Use `refactor_tool` with mode="suggest" for suggestions.
2. Use `refactor_tool` with mode="dead_code" to find unreferenced code.
3. For renames, use `refactor_tool` with mode="rename" to preview changes.
4. Use `apply_refactor_tool` with the refactor_id to apply renames.
