# LangGraph Support Agent

A **LangGraph** agent with real tool use — customer support that queries a live catalog and **computes pricing with tools** (no hallucinated math). Built with the modern `StateGraph` API (agent node → tools node → conditional loop).

**Verified output:**
```
👤 ¿Cuánto cuesta el data studio?
🤖 Data Studio: $499 (setup + 49/mes)

👤 ¿Y si lo pago anual, cuánto es en total?
🤖 El pago total anual es de $1,087   ← computed by the tool (499 + 12×49), not guessed
```

## What it demonstrates
- **LangGraph StateGraph**: `add_node` / `add_conditional_edges` / tool-call loop until resolved
- **Tool binding** with `bind_tools` (function calling) + `ToolMessage` handling
- **Grounded answers**: prices come from a `@tool`, totals are calculated by a tool — the LLM only orchestrates

## Run
```bash
pip install langgraph langchain-openai langchain-core
# LLM vía gateway OpenAI-compatible (o apunta a tu proveedor)
python langgraph_support_agent.py
```

## Architecture
```
StateGraph:
  START → agente → (¿tool_calls?) → tools → agente → ... → END
```

## Role
Built as a hands-on lab from the ed-donner/agents course material (module: LangChain/LangGraph), adapted to a real business use case. The same pattern (graph + tools + grounded responses) powers the production agents in my other repos.
