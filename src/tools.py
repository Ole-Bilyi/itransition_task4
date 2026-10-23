RUN_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": (
            "Run a read-only DuckDB SELECT against the bookstore tables. "
            "orders.paid_price already equals quantity * unit_price — sum it "
            "directly; never multiply by quantity again. "
            "For 'how many unique users' use COUNT(*) FROM entities. "
            "For 'top buyer' group by entity_id, not user_id or name. "
            "NEVER COUNT(DISTINCT user_id) FROM users."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "One SELECT or WITH query."},
            },
            "required": ["sql"],
        },
    },
}

CREATE_VIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "create_view",
        "description": (
            "Create a filtered 'view' on the dashboard using SQL. "
            "Provide a full SELECT that produces a subset of the table. "
            "The dashboard will show only the rows the SELECT returns. "
            "Use this for LIKE filters, ranges, id conditions — anything SQL can express. "
            "For predicates SQL cannot express (e.g. 'has a person name'), "
            "use create_semantic_view."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "enum": ["users", "books", "orders"]},
                "sql": {"type": "string",
                        "description": "SELECT * FROM <table> WHERE <condition>"},
                "description": {"type": "string",
                                "description": "Short label for the dashboard, e.g. 'emails starting with s or p'"},
            },
            "required": ["table", "sql", "description"],
        },
    },
}

CREATE_SEMANTIC_VIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "create_semantic_view",
        "description": (
            "Create a view filtered by a natural-language predicate that SQL cannot express. "
            "A classifier labels each value in the column. "
            "Rows matching the predicate are kept (or removed if exclude=true). "
            "Use for: 'contains a person name', 'sounds like a western genre', etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "enum": ["users", "books", "orders"]},
                "column": {"type": "string"},
                "predicate": {"type": "string"},
                "exclude": {"type": "boolean",
                            "description": "true → remove rows that match (use for 'filter out')"},
                "description": {"type": "string"},
            },
            "required": ["table", "column", "predicate", "description"],
        },
    },
}

REMOVE_VIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "remove_view",
        "description": "Clear a view (or all views if table is omitted).",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "enum": ["users", "books", "orders"]},
            },
        },
    },
}

SYSTEM_BASE = """You control a bookstore dashboard. Tables:
  users(user_id, name, address, phone, email, entity_id)
  books(book_id, title, authors, genre, publisher, year)
    -- authors is a comma-separated string.
  orders(order_id, user_id, book_id, quantity, unit_price, paid_price, date)
    -- paid_price = quantity * unit_price (line total). Sum it directly.
  entities(entity_id, user_ids, n_records)
  order_authors(order_id, user_id, book_id, quantity, paid_price, date, author)
    -- orders exploded by book author; use for per-author revenue.

For user counts: SELECT COUNT(*) FROM entities.
Never COUNT(DISTINCT user_id) FROM users.

Terminology:
  - "user" means one row in users (unique user_id = one account).
  - "buyer" / "person" / "customer" means one person, which may own
    several user_ids. Use entity_id to group aliases:
        SELECT entity_id, SUM(paid_price) ... GROUP BY entity_id
    Do NOT GROUP BY name — names are not deduplicated and may differ
    across a person's accounts.
"""

TOOLS_LINE = {
    "ask": (
        "You can call exactly one tool: run_sql(sql). "
        "Use it for every question that needs numbers or rows."
    ),
    "control": (
        "You can call exactly these tools: "
        "create_view(table, sql, description) to filter a table by SQL, "
        "create_semantic_view(...) for predicates SQL cannot express, "
        "remove_view(table) to clear a view. "
        "You cannot run SQL directly — to change what the dashboard shows, "
        "call create_view with a SELECT that returns the rows to keep. "
        "For example, 'keep only the top buyer' → "
        "create_view(table='users', sql=\"SELECT * FROM users WHERE entity_id IN "
        "(SELECT entity_id FROM (SELECT u.entity_id, SUM(o.paid_price) AS total "
        "FROM users u JOIN orders o ON u.user_id=o.user_id GROUP BY u.entity_id) "
        "ORDER BY total DESC LIMIT 1)\", description='top buyer')."
        "Never state numbers, IDs, dates, or rows you did not obtain from a"
        "run_sql result. If you cannot call run_sql in this mode, say so"
        "explicitly instead of guessing. Never invent a table or fill in"
        "placeholder values."
    ),
}