"""
CLI demand-forecasting agent.

Claude decides which tools to call (get_forecast / get_inventory_policy /
check_stockout_risk), runs them against the real trained pipeline, and answers
in plain language.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install anthropic
    python3 agent/agent.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agent.tools import TOOLS, run_tool, _world

# Editable. If your API account doesn't have this model, try "claude-sonnet-4-5".
MODEL = "claude-sonnet-5"

SYSTEM = (
    "You are an inventory analyst assistant for a retail demand-forecasting system. "
    "You have tools to look up demand forecasts and inventory policies for any store "
    "(1-10) and item (1-50). When the user asks about stock levels, reordering, or "
    "stockout risk, call the appropriate tool(s) and explain the numbers in plain "
    "language. Pass store/item as integers. Be concise. If a tool returns an error, "
    "relay it plainly rather than guessing."
)


def main():
    try:
        import anthropic
    except ImportError:
        print("Install the SDK first:  pip install anthropic")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set your API key first:  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic()

    _world()  # warm up the model once so the first question is fast

    print("\nDemand-forecast agent ready. Ask about forecasts, inventory, or stockout risk.")
    print("Examples:")
    print("  - What's the forecast for store 3 item 25?")
    print("  - How much safety stock for store 1 item 1 at 99% service?")
    print("  - Store 2 item 10 has 200 units left; stockout risk this week?")
    print("Type 'quit' to exit.\n")

    messages = []
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"quit", "exit", "q"}:
            break
        if not user:
            continue

        messages.append({"role": "user", "content": user})

        # Tool loop: keep going while Claude wants to call tools.
        while True:
            resp = client.messages.create(
                model=MODEL, max_tokens=1024, system=SYSTEM,
                tools=TOOLS, messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                print(f"\nagent > {text}\n")
                break

            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, block.input)
                    print(f"   [tool] {block.name}({block.input}) -> {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    main()
