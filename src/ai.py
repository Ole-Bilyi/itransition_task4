import json
import re
import time
import urllib.request
from pathlib import Path

import duckdb
from openai import APIError
from .tools import RUN_SQL_TOOL, CREATE_VIEW_TOOL, CREATE_SEMANTIC_VIEW_TOOL, REMOVE_VIEW_TOOL, SYSTEM_BASE, TOOLS_LINE

ID_COL = {"users": "user_id", "books": "book_id", "orders": "order_id"}
MAX_ROWS = 500                   # rows returned to the model per run_sql
MAX_COMPLETION_TOKENS = 2000     # max tokens per model turn
MAX_TOOL_RESULT_CHARS = 1500     # truncation of a tool result in messages
MAX_STEPS = 6                    # agent loop limit
BATCH = 155                      # classifier batch size
CLASSIFIER_MODEL = "openai/gpt-oss-20b"

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|CREATE|ALTER|TRUNCATE|ATTACH|DETACH|COPY|EXPORT|IMPORT|PRAGMA|CALL|INSTALL|LOAD|RESET)\b",
    re.IGNORECASE,
)

_NAMES_CACHE = Path(__file__).parent / "_names_cache_v1.txt"
_NAMES_URLS = (
    "https://raw.githubusercontent.com/smashew/NameDatabases/master/NamesDatabases/first%20names/us.txt",
    "https://raw.githubusercontent.com/smashew/NameDatabases/master/NamesDatabases/surnames/us.txt",
)
_NAME_STOPWORDS = frozenset({
    "the", "of", "and", "a", "an", "in", "on", "at", "to", "for", "with",
    "is", "it", "as", "by", "or", "be", "no", "not", "so", "up", "we", "my",
    "way", "wall", "war", "world", "life", "death", "love", "light", "dark",
    "fire", "water", "earth", "air", "sun", "moon", "star", "sea", "sky",
    "day", "night", "time", "year", "house", "home", "land", "city", "town",
    "king", "queen", "lord", "lady", "god", "man", "woman", "men", "women",
    "boy", "girl", "son", "daughter", "father", "mother", "brother", "sister",
    "all", "one", "two", "three", "first", "last", "new", "old", "young",
    "good", "bad", "great", "small", "big", "long", "short", "high", "low",
    "hand", "heart", "head", "eyes", "eye", "face", "name", "word", "song",
    "story", "book", "road", "door", "gate", "bridge", "river", "mountain",
    "tree", "flower", "bird", "fish", "dog", "cat", "horse",
    "rain", "snow", "wind", "storm", "cloud", "stone", "rock", "iron", "gold",
    "silver", "black", "white", "red", "blue", "green", "yellow",
    "south", "north", "east", "west", "here", "there", "where", "when",
    "what", "who", "why", "how", "which", "this", "that", "these", "those",
    "will", "bill", "mark", "art", "ray", "don", "dan", "chase", "cash",
    "hunter", "jack", "grace", "hope", "faith", "rose", "may", "june",
    "summer", "winter", "spring", "autumn", "fall", "dawn", "eve", "noon",
    "victory", "joy", "peace", "glory", "honor", "pride",
})
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")

_PERSON_PREDICATE_RE = re.compile(
    r"\b(?:person(?:'s)?|personal|human|people|real|actual|proper|"
    r"first|last|given|christian|sur|family)\s+names?",
    re.IGNORECASE,
)

_names_loaded: set[str] | None = None


def _load_names() -> set[str]:
    global _names_loaded
    if _names_loaded is not None:
        return _names_loaded

    if _NAMES_CACHE.exists():
        try:
            with open(_NAMES_CACHE, encoding="utf-8") as f:
                _names_loaded = {line.strip().lower() for line in f if line.strip()}
            return _names_loaded
        except OSError:
            pass

    names: set[str] = set()
    for url in _NAMES_URLS:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                for line in resp:
                    token = line.decode("utf-8", errors="ignore").strip().lower()
                    if token:
                        names.add(token)
        except Exception:
            continue

    if names:
        try:
            _NAMES_CACHE.write_text("\n".join(sorted(names)), encoding="utf-8")
        except OSError:
            pass

    _names_loaded = names
    return names


def _text_contains_person_name(text: str, names: set[str]) -> bool:
    for match in _TOKEN_RE.finditer(text):
        token = match.group().lower()
        if token in names and token not in _NAME_STOPWORDS:
            return True
    return False


def _person_name_mask(col, names: set[str]) -> list[bool]:
    if col.apply(lambda x: isinstance(x, list)).any():
        return [
            any(_text_contains_person_name(str(v), names) for v in lst)
            if isinstance(lst, list) else False
            for lst in col
        ]
    return [_text_contains_person_name(str(v), names) for v in col]


def _validate_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("multiple statements not allowed")
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("only SELECT/WITH queries are allowed")
    if _FORBIDDEN.search(sql):
        raise ValueError("DDL/DML keywords are not allowed")
    return sql


def _connect(tables: dict) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET enable_external_access = false;")
    except Exception:
        pass
    for name, df in tables.items():
        con.register(name, df)
    return con


def _run_sql(con, sql: str) -> dict:
    sql = _validate_sql(sql)
    df = con.execute(f"SELECT * FROM ({sql}) AS q LIMIT {MAX_ROWS + 1}").fetchdf()
    truncated = len(df) > MAX_ROWS
    if truncated:
        df = df.head(MAX_ROWS)
    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "row_count": len(df),
        "truncated": truncated,
    }


def _execute_ids(con, sql: str, id_col: str) -> set[int]:
    sql = _validate_sql(sql)
    df = con.execute(f"SELECT {id_col} FROM ({sql}) AS q").fetchdf()
    return set(int(x) for x in df[id_col].dropna())


def _build_prompt(values: list[str], predicate: str) -> str:
    numbered = "\n".join(f"{i}. {v!r}" for i, v in enumerate(values, 1))
    return f"Predicate: {predicate}\n\nValues:\n{numbered}"


_SYS = "Return ONLY a JSON array of 1-based indices that satisfy the predicate."

def _ask_classifier(client, values: list[str], predicate: str) -> str:
    resp = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        messages=[
            {"role": "system", "content": _SYS},
            {"role": "user", "content": _build_prompt(values, predicate)},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _parse_indices(raw: str) -> set[int]:
    m = re.search(r"\[[^\]]*\]", raw)
    if not m:
        raise ValueError(f"classifier returned no JSON array: {raw[:200]!r}")
    try:
        return {int(x) for x in json.loads(m.group()) if str(x).isdigit()}
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"classifier returned bad JSON: {raw[:200]!r}") from exc


def _classify(client, values: list[str], predicate: str) -> list[bool]:
    keep: list[bool] = []
    for start in range(0, len(values), BATCH):
        batch = values[start : start + BATCH]
        matched = _parse_indices(_ask_classifier(client, batch, predicate))
        keep.extend((i + 1) in matched for i in range(len(batch)))
        if start + BATCH < len(values):
            time.sleep(1)
    return keep


def _list_mask(col, predicate: str, client) -> list[bool]:
    unique_vals = sorted({str(v) for lst in col if isinstance(lst, list) for v in lst})
    flags = _classify(client, unique_vals, predicate)
    matched = {v for v, k in zip(unique_vals, flags) if k}
    return [
        any(str(v) in matched for v in lst) if isinstance(lst, list) else False
        for lst in col
    ]


def _scalar_mask(col, predicate: str, client) -> list[bool]:
    values = col.astype(str).tolist()
    unique = sorted(set(values))
    flags = _classify(client, unique, predicate)
    mapping = dict(zip(unique, flags))
    return [mapping.get(v, False) for v in values]


def _classify_column(col, predicate: str, client) -> list[bool]:
    if col.apply(lambda x: isinstance(x, list)).any():
        return _list_mask(col, predicate, client)
    return _scalar_mask(col, predicate, client)


def _semantic_mask(col, predicate: str, client) -> list[bool]:
    if _PERSON_PREDICATE_RE.search(predicate):
        names = _load_names()
        if names:
            return _person_name_mask(col, names)
    return _classify_column(col, predicate, client)


def _create_semantic_view(ctx: dict, table: str, column: str,
                          predicate: str, exclude: bool, description: str) -> dict:
    df = ctx["base_tables"][table]
    if column not in df.columns:
        return {"error": f"column '{column}' not in {table}"}

    keep = _semantic_mask(df[column], predicate, ctx["client"])
    if exclude:
        keep = [not k for k in keep]

    visible_ids = {int(x) for x in df.loc[keep, ID_COL[table]]}
    ctx["views"][table] = {
        "kind": "semantic",
        "ids": visible_ids,
        "sql": None,
        "column": column,
        "predicate": predicate,
        "exclude": exclude,
        "description": description,
    }
    return {
        "ok": True,
        "table": table,
        "visible_rows": len(visible_ids),
        "total_rows": len(df),
        "description": description,
    }


def _create_sql_view(ctx: dict, table: str, sql: str, description: str) -> dict:
    try:
        ids = _execute_ids(ctx["duck"], sql, ID_COL[table])
    except Exception as exc:
        return {"error": f"SQL failed: {exc}"}

    ctx["views"][table] = {
        "kind": "sql",
        "ids": ids,
        "sql": sql,
        "description": description,
    }
    return {
        "ok": True,
        "table": table,
        "visible_rows": len(ids),
        "total_rows": len(ctx["base_tables"][table]),
        "description": description,
    }


def _h_run_sql(ctx, args):
    return _run_sql(ctx["duck"], args["sql"])


def _h_create_view(ctx, args):
    return _create_sql_view(ctx, args["table"], args["sql"], args["description"])


def _h_create_semantic_view(ctx, args):
    return _create_semantic_view(
        ctx, args["table"], args["column"], args["predicate"],
        bool(args.get("exclude", False)), args["description"],
    )


def _h_remove_view(ctx, args):
    if t := args.get("table"):
        ctx["views"].pop(t, None)
    else:
        ctx["views"].clear()
    return {"ok": True}


HANDLERS = {
    "run_sql": _h_run_sql,
    "create_view": _h_create_view,
    "create_semantic_view": _h_create_semantic_view,
    "remove_view": _h_remove_view,
}


def _dispatch(name: str, args: dict, ctx: dict) -> dict:
    handler = HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool {name}"}
    try:
        return handler(ctx, args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _views_banner(views: dict) -> str:
    if not views:
        return "\n\nNo dashboard views are currently active."
    rows = [f"  {t}: {v['description']} ({len(v['ids'])} visible rows)"
            for t, v in views.items()]
    return "\n".join(["\n\nCURRENT DASHBOARD VIEWS:", *rows])


def _initial_messages(question: str, views: dict, mode: str) -> list:
    system = SYSTEM_BASE + "\n\n" + TOOLS_LINE[mode] + _views_banner(views)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def _chat(client, messages: list, tools: list, model: str):
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=tools,
        tool_choice="auto", temperature=0, max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    return resp.choices[0].message


def _run_tool_call(call, ctx: dict) -> tuple[dict, dict]:
    try:
        args = json.loads(call.function.arguments)
        return args, _dispatch(call.function.name, args, ctx)
    except Exception as exc:
        return {}, {"error": str(exc)}


def _append_tool_call(messages: list, trace: list, call, ctx: dict) -> None:
    args, result = _run_tool_call(call, ctx)
    trace.append({"tool": call.function.name, "args": args, "result": result})
    messages.append({
        "role": "tool", "tool_call_id": call.id,
        "content": json.dumps(result, default=str)[:MAX_TOOL_RESULT_CHARS],
    })


def _step(messages, tools, model, ctx, trace):
    try:
        msg = _chat(ctx["client"], messages, tools, model)
    except APIError as exc:
        return (
            f"The LLM call failed ({type(exc).__name__}). Try rephrasing — "
            "e.g. 'filter the dashboard so only the top buyer remains'."
        )
    messages.append(msg)
    if not msg.tool_calls:
        return msg.content or ""
    for call in msg.tool_calls:
        _append_tool_call(messages, trace, call, ctx)
    return None


TOOLS_BY_MODE = {
    "ask": [RUN_SQL_TOOL],
    "control": [CREATE_VIEW_TOOL, CREATE_SEMANTIC_VIEW_TOOL, REMOVE_VIEW_TOOL],
}


def ask(question: str, ctx: dict, mode: str, model: str = CLASSIFIER_MODEL) -> tuple[str, list]:
    tools = TOOLS_BY_MODE[mode]
    messages = _initial_messages(question, ctx["views"], mode)
    trace = []
    for _ in range(MAX_STEPS):
        answer = _step(messages, tools, model, ctx, trace)
        if answer is not None:
            return answer, trace

    return "(too many tool calls)", trace


def make_context(clean: dict, sql_tables: dict, views: dict, client) -> dict:
    return {
        "client": client,
        "base_tables": {"users": clean["users"], "books": clean["books"], "orders": clean["orders"]},
        "sql_tables": sql_tables,
        "views": views,
        "duck": _connect(sql_tables),
    }