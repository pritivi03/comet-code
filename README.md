# CometCode

An agentic coding assistant for the terminal. Give it a task — it reads your codebase, reasons about what to change, proposes edits, and applies them only after you approve.

Built from scratch on **LangGraph** and **OpenRouter**, with a streaming **Rich** terminal UI.

---

## Demo

<img alt="Screen Recording 2026-04-19 at 8 03 50 PM" src="https://github.com/user-attachments/assets/5e051a0c-4a59-4204-9fda-06cd7d91126c" />

---

## Architecture

CometCode is structured as a **LangGraph state machine** where each node receives the full `AgentState` and returns state updates. The graph has two nodes — `call_llm` and `execute_tools` — connected by conditional edges that route based on the model's response type.

```
START → call_llm ──→ execute_tools ──┐
            ↑                        │
            └────────────────────────┘
            (loops until final answer or budget exhausted)
```

**Key design decisions:**

- **AgentState as the single source of truth** — conversation history, token budgets, tool-call counters, evidence notes, and attempt metadata all live in one typed dict that flows through the graph. Nodes are pure: they receive state, return a patch.

- **Dual invocation paths** — models that support native function calling stream through `_invoke_native`; models that don't fall back to `_invoke_json_fallback` with a structured Pydantic output schema. The rest of the system is unaware of the difference.

- **Explicit budget enforcement** — the graph tracks `tool_calls_used`, `consecutive_no_signal`, and `repeat_call_streak`. Hitting any limit triggers a soft-stop: the model is forced into a final answer using whatever evidence it collected, rather than erroring.

- **Human-in-the-loop approval** — mutating tools (`write_file`, `replace_text`) are gated behind a `request_approval` callback. After approval and execution, the loop returns to the LLM so it sees the updated file state before proposing the next edit — preventing duplicate proposals.

- **Mode-scoped prompting** — `explain`, `debug`, `refactor`, `implement`, and `plan` modes each carry different system instructions and tool permissions. Read-only modes (`explain`, `plan`) never receive mutating tools in their schema.

---

## Features

- **Five task modes** — `explain`, `debug`, `refactor`, `implement`, `plan`
- **Multi-model support** — route to any OpenRouter model; native tool calling used when available, JSON schema fallback otherwise
- **Streaming terminal UI** — live spinner with real-time token count and elapsed timer; tool history with colored unified diffs for proposed changes
- **Persistent conversation history** — context carries across turns within a session; mode switches are surfaced to the model explicitly
- **Slash command interface** — `/mode`, `/model`, `/tools`, `/clear`, `/help` with tab-completion
- **Self-limiting agent** — budget caps on tool calls, no-signal streaks, and repeated identical calls; graceful degradation to best-effort answers

---

## Stack

| Layer | Technology |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM routing | [OpenRouter](https://openrouter.ai) via LangChain `ChatOpenAI` |
| Terminal UI | [Rich](https://github.com/Textualize/rich) + [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) |
| Schema validation | [Pydantic v2](https://docs.pydantic.dev) |
| Python | 3.12+ |

---

## Getting Started

```bash
pip install comet-code
# The first run will prompt for your OpenRouter API key.
# You can also set it later via `/key set <key>` or the `OPENROUTER_API_KEY` environment variable.
comet
```

### Slash commands

| Command | Description |
|---|---|
| `/mode <name>` | Switch task mode (`explain`, `debug`, `refactor`, `implement`, `plan`) |
| `/model <name>` | Switch model (by alias, slug, or label) |
| `/tools` | Show last run tool history |
| `/clear` | Reset conversation history |
| `/help` | List all commands |

---

## Project Layout

```
src/
  cli/          # Terminal UI, input handling, rendering, slash commands
  core/         # LangGraph graph, nodes, orchestrator, state schema
  llm/          # OpenRouter client, model catalog, prompt builder
  schemas/      # Pydantic models (events, tasks, state, tools)
  tools/        # Tool implementations (read_file, search_text, replace_text, …)
```

---

## License

MIT
