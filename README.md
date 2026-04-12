## CometCode

CometCode is an interactive CLI coding assistant built with LangGraph.

### Current Runtime Shape
- LangGraph loop: `call_llm -> execute_tools -> call_llm` until completion.
- Repository tools:
  - read-only: `list_files`, `search_text`, `find_files`, `print_tree`, `read_file`, `read_range`
  - mutating with approval: `write_file`, `replace_text`
- Shared tool-definition layer for all models.
- Model adapter strategy:
  - Native function-calling when the selected model is known to support it.
  - Automatic JSON tool-call fallback for weaker/unsupported models.
- Streaming UX in CLI:
  - incremental assistant token output
  - tool start/end events with summarized output
  - approval prompt before mutating tools execute

### Modes (Current Stage)
- `explain` and `plan` use read-only tools only.
- `debug`, `refactor`, and `implement` can propose mutating tools, which require user approval.
- Arbitrary command execution is still not enabled yet in this stage.

### Run
```bash
comet
```

Ensure `OPENROUTER_API_KEY` is set in your environment or `.env`.
