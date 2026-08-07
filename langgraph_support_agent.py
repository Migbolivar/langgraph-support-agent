"""
LangGraph Customer Support Agent — lab práctico (para portafolio Koso AI)
Un agente con flujo de estado (StateGraph): nodo LLM + nodo herramientas,
con loop condicional hasta resolver. LLM: Gemini 2.5 Flash (gratis).

Correr:  GEMINI_API_KEY=... python langgraph_support_agent.py
"""
import os
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ── Tools (herramientas reales del agente) ──
CATALOGO = {
    "chatbot rag": ("Chatbot RAG", 49, "mes"),
    "data studio": ("Data Studio", 499, "setup + 49/mes"),
    "crm": ("CRM + pipeline", 199, "setup + 29/mes"),
    "ai agents": ("AI Agents", 299, "setup + 99/mes"),
}

@tool
def buscar_catalogo(producto: str) -> str:
    """Busca un servicio en el catálogo BMOPS y devuelve precio y condiciones."""
    for k, (nombre, precio, cond) in CATALOGO.items():
        if k in producto.lower():
            return f"{nombre}: ${precio} ({cond})"
    return f"No encontrado. Disponibles: {', '.join(CATALOGO)}"

@tool
def calcular_pago(anual: bool, setup: float, mensual: float) -> str:
    """Calcula el pago total: setup + 12 meses (anual) o setup + 1 mes."""
    total = setup + (12 * mensual if anual else mensual)
    return f"Total: ${total:.2f} ({'anual' if anual else 'mensual'})"

tools = [buscar_catalogo, calcular_pago]
tools_by_name = {t.name: t for t in tools}

# ── LLM (Gemini gratis, compatible OpenAI) ──
llm = ChatOpenAI(
    model="auto",  # gateway FreeLLMAPI elige modelo free disponible
    api_key=os.environ.get("FREELLMAPI_KEY", "x"),
    base_url="http://127.0.0.1:3010/v1",
).bind_tools(tools)

# ── Estado + nodos ──
class State(TypedDict):
    messages: Annotated[list, add_messages]

def agente_nodo(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def tools_nodo(state: State):
    last = state["messages"][-1]
    msgs = []
    for tc in last.tool_calls:
        out = tools_by_name[tc["name"]].invoke(tc["args"])
        msgs.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": msgs}

def debe_continuar(state: State):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

# ── Grafo ──
g = StateGraph(State)
g.add_node("agente", agente_nodo)
g.add_node("tools", tools_nodo)
g.add_edge(START, "agente")
g.add_conditional_edges("agente", debe_continuar, {"tools": "tools", END: END})
g.add_edge("tools", "agente")
app = g.compile()

SYSTEM = ("Eres el asistente de ventas de BMOPS Consulting (IA para PYMEs). "
          "Responde breve, en español. Usa las herramientas para precios reales.")

def chat(pregunta: str):
    out = app.invoke({"messages": [SystemMessage(SYSTEM), HumanMessage(pregunta)]})
    return out["messages"][-1].content

if __name__ == "__main__":
    for q in [
        "¿Cuánto cuesta el data studio?",
        "¿Y si lo pago anual, cuánto es en total?",
    ]:
        print(f"👤 {q}\n🤖 {chat(q)}\n")
