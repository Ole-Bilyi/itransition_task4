import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from openai import OpenAI

from src.data import load_and_clean, compute_metrics, build_sql_tables, visible_tables
from src.ai import ask, make_context


DATA_ROOT = Path(__file__).parent / "data"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

st.set_page_config(page_title="Bookstore BI", layout="wide")


def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def make_ctx(clean, api_key, views):
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return make_context(clean, build_sql_tables(clean), views, client)


@st.cache_data(show_spinner="Loading and cleaning...")
def load_cached(path: str) -> dict:
    return load_and_clean(Path(path))


def trace_sql(args, result):
    st.code(args.get("sql", ""), language="sql")
    if "error" in result:
        st.error(result["error"])
        return
    cap = f"{result.get('row_count', '?')} rows"
    if result.get("truncated"):
        cap += " (truncated)"
    st.caption(cap)


def trace_view(args, result):
    if args.get("sql"):
        st.code(args["sql"], language="sql")
    else:
        st.caption(
            f"{args.get('predicate')!r} on "
            f"{args.get('table')}.{args.get('column')}"
            + (" (excluded)" if args.get("exclude") else "")
        )
    if "error" in result:
        st.error(result["error"])
    elif "visible_rows" in result:
        st.caption(f"kept {result['visible_rows']} of {result['total_rows']} rows")


def render_trace(trace):
    if not trace:
        return
    with st.expander("Tool calls"):
        for i, step in enumerate(trace, 1):
            tool = step["tool"]
            args = step["args"]
            result = step.get("result", {})
            st.markdown(f"**Call {i}: {tool}**")
            if tool == "run_sql":
                trace_sql(args, result)
            elif tool in ("create_view", "create_semantic_view"):
                trace_view(args, result)
            else:
                st.json(args)


def render_chat(history):
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                render_trace(msg["trace"])


def render_kpis(m):
    st.metric("Revenue", f"${m['total_revenue']:,.2f}")
    st.metric("Unique users", f"{m['n_users']:,}")
    st.metric("Author sets", f"{m['n_author_sets']:,}")
    st.metric("Books", f"{m['n_books']:,}")
    st.metric("Top author", f"{m['top_author']}")


def render_active_views(views):
    if not views:
        return
    with st.container(border=True):
        st.markdown("**Active views**")
        for table, v in views.items():
            st.caption(f"{table}: {v['description']} ({len(v['ids'])} rows)")


def render_top_days(m):
    st.subheader("Top 5 days by revenue")
    t5 = m["top5"].copy()
    t5["Revenue"] = t5["revenue"].map(lambda x: f"${x:,.2f}")
    st.dataframe(
        t5[["date_str", "Revenue"]].rename(columns={"date_str": "Date"}),
        use_container_width=True, hide_index=True,
    )


def render_best_buyer(m):
    st.subheader("Best buyer")
    names = m.get("top_buyer_names") or []
    if names:
        st.markdown(f"**{', '.join(names)}**")
        st.caption(
            f"entity {m.get('top_buyer_entity')} — spent "
            f"${m.get('top_buyer_total', 0.0):,.2f} on visible orders"
        )
    else:
        st.caption("No orders in the current view.")

    st.code(str(m["top_buyer_ids"]), language="python")


def render_daily_chart(daily):
    st.subheader("Daily revenue")
    daily = daily.sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(pd.to_datetime(daily["date"]), daily["revenue"],
            marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue (USD)")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)


def render_author_popularity(popularity):
    st.subheader("Author popularity (units sold)")
    st.dataframe(
        popularity.rename(columns={"label": "Author / set", "units": "Units sold"}),
        use_container_width=True, hide_index=True,
    )


def render_dashboard(clean, views, store_name):
    m = compute_metrics(clean, views)

    st.title(f"{store_name} bookstore")

    render_kpis(m)
    render_active_views(views)

    st.divider()
    col_left, col_right = st.columns([3, 2])
    with col_left:
        render_top_days(m)
    with col_right:
        render_best_buyer(m)

    st.divider()
    render_daily_chart(m["daily"])

    st.divider()
    render_author_popularity(m["author_popularity"])


def handle_question(clean, api_key, question):
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    ctx = make_ctx(clean, api_key, st.session_state.views)
    with st.chat_message("assistant"):
        with st.spinner("Computing..."):
            answer, trace = ask(question, ctx, mode="ask")
        st.markdown(answer)
        render_trace(trace)

    st.session_state.chat.append(
        {"role": "assistant", "content": answer, "trace": trace}
    )


def render_ask_tab(clean, api_key):
    st.title("Ask questions about the data")
    st.caption(
        "The model writes SQL; every number comes from DuckDB. "
    )
    render_chat(st.session_state.chat)

    question = st.chat_input("e.g. How many real unique users are there?")
    if question and api_key:
        handle_question(clean, api_key, question)


def render_view_preview(name, df_v, view):
    label = f"{name} - {len(df_v)} visible rows"
    if view:
        label += f" - {view['description']}"
    with st.expander(label, expanded=bool(view)):
        if view and view["kind"] == "sql":
            st.code(view["sql"], language="sql")
        elif view and view["kind"] == "semantic":
            st.caption(
                f"{view['predicate']!r} on {view['column']}"
                + (" (excluded)" if view["exclude"] else "")
            )
        st.dataframe(df_v.head(10), use_container_width=True, hide_index=True)


def render_preview(clean, views):
    st.subheader("Current views and data preview")
    users_v, books_v, orders_v = visible_tables(clean, views)
    for name, df_v in (("users", users_v), ("books", books_v), ("orders", orders_v)):
        render_view_preview(name, df_v, views.get(name))


def handle_command(clean, api_key, command):
    st.session_state.control_chat.append({"role": "user", "content": command})
    with st.chat_message("user"):
        st.markdown(command)

    scratch = {t: dict(v) for t, v in st.session_state.views.items()}
    ctx = make_ctx(clean, api_key, scratch)
    with st.chat_message("assistant"):
        with st.spinner("Applying..."):
            answer, trace = ask(command, ctx, mode="control")
            st.session_state.views = ctx["views"]
        st.markdown(answer)
        render_trace(trace)

    st.session_state.control_chat.append(
        {"role": "assistant", "content": answer, "trace": trace}
    )
    st.rerun()


def render_control_tab(clean, api_key):
    st.title("Control the dashboard")
    st.caption(
        "For commands like: filter out books with people names in the title, "
        "select users with emails starting with s or p"
    )

    render_preview(clean, st.session_state.views)
    st.divider()
    st.subheader("Command chat")
    render_chat(st.session_state.control_chat)

    command = st.chat_input("filter out books whose title contains a person's name")
    if command and api_key:
        handle_command(clean, api_key, command)


if "views" not in st.session_state:
    st.session_state.views = {}
if "chat" not in st.session_state:
    st.session_state.chat = []
if "control_chat" not in st.session_state:
    st.session_state.control_chat = []

stores = sorted(p for p in DATA_ROOT.glob("STORE*") if p.is_dir())
if not stores:
    st.error(f"No STORE folders under {DATA_ROOT}")
    st.stop()

store = st.sidebar.selectbox("Store", stores, format_func=lambda p: p.name)
if st.sidebar.button("Reset all views"):
    st.session_state.views = {}
    st.rerun()
if st.session_state.views:
    st.sidebar.info(f"{len(st.session_state.views)} active view(s)")

api_key = get_api_key()
if not api_key:
    st.warning("No GROQ_API_KEY set (env var or .streamlit/secrets.toml).")

clean = load_cached(str(store))

tab_dash, tab_ask, tab_control = st.tabs(["Dashboard", "Ask questions", "Control dashboard"])

with tab_dash:
    render_dashboard(clean, st.session_state.views, store.name)

with tab_ask:
    render_ask_tab(clean, api_key)

with tab_control:
    render_control_tab(clean, api_key)