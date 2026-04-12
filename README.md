## CometCode

CometCode is an interactive CLI coding assistant built with LangGraph.

### Current Runtime Shape
- LangGraph loop: `call_llm -> execute_tools -> call_llm` until completion.
- Read-only repository tools: `list_files`, `search_text`, `read_file`, `read_range`.
- Shared tool-definition layer for all models.
- Model adapter strategy:
  - Native function-calling when the selected model is known to support it.
  - Automatic JSON tool-call fallback for weaker/unsupported models.
- Streaming UX in CLI:
  - incremental assistant token output
  - tool start/end events with summarized output

### Modes (Current Stage)
- `explain` and `plan` can use read-only tools.
- Mutation tools and command execution are intentionally not enabled yet in this stage.

### Run
```bash
comet
```

Ensure `OPENROUTER_API_KEY` is set in your environment or `.env`.
