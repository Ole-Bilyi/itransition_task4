import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ID_COL = {"users": "user_id", "books": "book_id", "orders": "order_id"}

_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S %p", "%H:%M:%S %p %Y-%m-%d",
    "%H:%M:%S %d-%b-%Y",    "%H:%M %d-%b-%Y",
    "%Y-%m-%d %H:%M",       "%Y-%m-%d %H:%M:%S",
)


def _strip_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def _coerce_int_id(df: pd.DataFrame, src: str, dst: str) -> pd.DataFrame:
    out = df.rename(columns={src: dst}).copy()
    out[dst] = pd.to_numeric(out[dst], errors="coerce").astype("Int64")
    return out.dropna(subset=[dst])


def _to_int_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _read_yaml(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame(yaml.safe_load(f))


def _clean_users(raw: pd.DataFrame) -> pd.DataFrame:
    df = _coerce_int_id(raw, "id", "user_id")
    df = _strip_cols(df, ("name", "email", "phone", "address"))
    df["email"] = df["email"].str.lower()
    df = df.replace({"NULL": pd.NA, "NAN": pd.NA, "": pd.NA})
    return df.drop_duplicates()


def _split_authors(val) -> list[str]:
    if isinstance(val, list):
        return [str(a).strip() for a in val if str(a).strip()]
    if pd.isna(val) or not str(val).strip():
        return []
    return [p.strip() for p in re.split(r"\s*(?:,|&|\band\b)\s*", str(val)) if p.strip()]


def _clean_books(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.lstrip(":") for c in df.columns]
    df = df.rename(columns={"author": "authors"})
    df = _coerce_int_id(df, "id", "book_id")
    df["authors"] = df["authors"].apply(_split_authors)
    df = _to_int_col(df, "year")
    return df.drop_duplicates(subset=["book_id"])


def _parse_price(s):
    if pd.isna(s):
        return np.nan
    text = str(s).strip().upper()
    if text in ("", "NULL", "NAN"):
        return np.nan
    is_eur = "€" in text or "EUR" in text
    m = re.search(r"[\d,]+\.?\d*", text)
    if not m:
        return np.nan
    return float(m.group().replace(",", "")) * (1.2 if is_eur else 1.0)


def _normalize_timestamp(s) -> str:
    text = re.sub(r"A\.M\.", "AM", str(s).strip(), flags=re.I)
    text = re.sub(r"P\.M\.", "PM", text, flags=re.I)
    return text.replace(";", " ").replace(",", " ")


def _parse_timestamp(s):
    if pd.isna(s):
        return pd.NaT
    text = _normalize_timestamp(s)
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return pd.to_datetime(text, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(text, errors="coerce")


def _add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["date"]  = ts.dt.date
    df["year"]  = ts.dt.year
    df["month"] = ts.dt.month
    df["day"]   = ts.dt.day
    return df


def _clean_orders(raw: pd.DataFrame) -> pd.DataFrame:
    df = _coerce_int_id(raw, "id", "order_id")

    df["unit_price"] = df["unit_price"].apply(_parse_price)
    df["quantity"]   = pd.to_numeric(df["quantity"], errors="coerce")
    df["timestamp"]  = df["timestamp"].apply(_parse_timestamp)
    df = _to_int_col(df, "user_id")
    df = _to_int_col(df, "book_id")

    df = df.replace({"NULL": pd.NA, "": pd.NA})
    df = df.dropna(subset=["quantity", "unit_price", "timestamp", "book_id", "user_id"])
    df = df.drop_duplicates(subset=["order_id"])
    df = _add_date_parts(df)
    df["paid_price"] = (df["quantity"] * df["unit_price"]).round(2)
    return df


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        pi, pj = self.find(i), self.find(j)
        if pi != pj:
            self.parent[pj] = pi


def _entity_fields(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "_name":  df["name"].fillna("").astype(str).str.lower().str.strip(),
        "_email": df["email"].fillna("").astype(str).str.lower().str.strip(),
        "_phone": df["phone"].fillna("").astype(str).str.replace(r"\D", "", regex=True),
        "_addr":  df["address"].fillna("").astype(str).str.lower().str.strip(),
    })


def _block_indices(records: list[dict], fields: list[str]) -> dict:
    blocks = defaultdict(set)
    for i, rec in enumerate(records):
        for f in fields:
            if rec[f]:
                blocks[(f, rec[f])].add(i)
    return blocks


def _n_matching(a: dict, b: dict, fields: list[str]) -> int:
    return sum(1 for f in fields if a[f] and a[f] == b[f])


def _resolve_entities(users: pd.DataFrame) -> tuple[dict, int]:
    df = users.copy()
    norm = _entity_fields(df)
    fields = list(norm.columns)
    records = norm.to_dict("records")

    uf = _UnionFind(len(records))
    for idxs in _block_indices(records, fields).values():
        if len(idxs) < 2:
            continue
        for a, b in combinations(sorted(idxs), 2):
            if _n_matching(records[a], records[b], fields) >= 3:
                uf.union(a, b)

    cluster_map = {int(uid): uf.find(i) for i, uid in enumerate(df["user_id"])}
    return cluster_map, len(set(cluster_map.values()))


def load_and_clean(store_dir: Path) -> dict:
    users  = _clean_users(pd.read_csv(store_dir / "users.csv", dtype="string"))
    books  = _clean_books(_read_yaml(store_dir / "books.yaml"))
    orders = _clean_orders(pd.read_parquet(store_dir / "orders.parquet"))

    cluster_map, n_unique = _resolve_entities(users)
    users["entity_id"] = users["user_id"].map(cluster_map)

    return {
        "store_name": store_dir.name,
        "users": users,
        "books": books,
        "orders": orders,
        "n_unique_users_total": n_unique,
    }


def _apply_view(df: pd.DataFrame, view: dict | None, id_col: str) -> pd.DataFrame:
    if not view:
        return df
    return df[df[id_col].isin(view["ids"])]


def visible_tables(clean: dict, views: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users  = _apply_view(clean["users"],  views.get("users"),  "user_id")
    books  = _apply_view(clean["books"],  views.get("books"),  "book_id")
    orders = _apply_view(clean["orders"], views.get("orders"), "order_id")

    orders = orders[
        orders["user_id"].isin(users["user_id"]) &
        orders["book_id"].isin(books["book_id"])
    ]
    return users, books, orders


def _daily_revenue(orders: pd.DataFrame) -> pd.DataFrame:
    daily = orders.groupby("date", as_index=False).agg(revenue=("paid_price", "sum"))
    daily["date_str"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    return daily


def _top5_days(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.nlargest(5, "revenue")[["date_str", "revenue"]].reset_index(drop=True)


def _authorset(a) -> frozenset:
    if not isinstance(a, list):
        return frozenset()
    return frozenset(x.lower().strip() for x in a if str(x).strip())


def _author_sets(books: pd.DataFrame) -> pd.Series:
    return books["authors"].apply(_authorset)


def _smart_title(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in s.split())


def _author_popularity(orders: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["label", "units"])
    if orders.empty or books.empty:
        return empty

    pairs = (
        books[["book_id", "authors"]]
        .explode("authors")
        .dropna(subset=["authors"])
        .rename(columns={"authors": "author"})
    )
    if pairs.empty:
        individual = empty.copy()
    else:
        individual = (
            orders.merge(pairs, on="book_id", how="inner")
            .groupby("author", as_index=False)
            .agg(units=("quantity", "sum"))
            .rename(columns={"author": "label"})
        )

    book_to_set = dict(zip(books["book_id"], books["authors"].apply(_authorset)))
    enriched = orders.assign(author_set=orders["book_id"].map(book_to_set))

    if "author_set" not in enriched.columns or enriched.empty:
        return individual if not individual.empty else empty

    mask = enriched["author_set"].apply(
        lambda s: isinstance(s, frozenset) and len(s) > 1
    )
    enriched = enriched[mask].copy()

    if enriched.empty:
        sets = empty.copy()
    else:
        sets = (
            enriched.groupby("author_set", as_index=False)
            .agg(units=("quantity", "sum"))
        )
        sets["label"] = sets["author_set"].apply(
            lambda s: ", ".join(sorted(_smart_title(a) for a in s))
        )
        sets = sets[["label", "units"]]

    return (
        pd.concat([individual, sets], ignore_index=True)
        .nlargest(20, "units")
        .reset_index(drop=True)
    )


def _best_buyer(orders: pd.DataFrame, users: pd.DataFrame) -> dict:
    merged = orders.merge(users[["user_id", "entity_id"]], on="user_id", how="inner")
    spend = merged.groupby("entity_id", as_index=False).agg(total=("paid_price", "sum"))
    if spend.empty:
        return {"ids": [], "names": [], "entity_id": None, "total": 0.0}

    row = spend.nlargest(1, "total").iloc[0]
    best_entity = row["entity_id"]
    sub = users.loc[users["entity_id"] == best_entity]
    return {
        "ids":       sorted(int(x) for x in sub["user_id"]),
        "names":     sorted({str(n) for n in sub["name"].dropna()}),
        "entity_id": int(best_entity),
        "total":     float(row["total"]),
    }


def compute_metrics(clean: dict, views: dict) -> dict:
    users, books, orders = visible_tables(clean, views)

    daily      = _daily_revenue(orders)
    book_sets  = _author_sets(books)
    popularity = _author_popularity(orders, books)
    buyer      = _best_buyer(orders, users)
    
    return {
        "daily":             daily,
        "top5":              _top5_days(daily),
        "n_users":           int(users["entity_id"].nunique()),
        "n_author_sets":     len({s for s in book_sets if s}),
        "n_books":           len(books),
        "top_author":        popularity.iloc[0]["label"] if len(popularity) else "—",
        "author_popularity": popularity,
        "top_buyer_ids":     buyer["ids"],
        "top_buyer_names":   buyer["names"],
        "top_buyer_entity":  buyer["entity_id"],
        "top_buyer_total":   buyer["total"],
        "total_revenue":     float(orders["paid_price"].sum()) if len(orders) else 0.0,
    }


def _build_entities(users: pd.DataFrame) -> pd.DataFrame:
    return users.groupby("entity_id", as_index=False).agg(
        user_ids=("user_id", lambda s: ", ".join(str(int(x)) for x in s)),
        n_records=("user_id", "size"),
    )


def _flatten_books(books: pd.DataFrame) -> pd.DataFrame:
    flat = books[["book_id", "title", "genre", "publisher", "year"]].copy()
    flat["authors"] = books["authors"].apply(
        lambda a: ", ".join(a) if isinstance(a, list) else ""
    )
    return flat


def _explode_order_authors(orders: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    return orders.merge(
        books[["book_id", "authors"]]
        .explode("authors")
        .rename(columns={"authors": "author"})
        .dropna(subset=["author"]),
        on="book_id", how="inner",
    )


def build_sql_tables(clean: dict) -> dict:
    users  = clean["users"]
    books  = clean["books"]
    orders = clean["orders"]

    return {
        "users":         users[["user_id", "name", "address", "phone", "email", "entity_id"]],
        "entities":      _build_entities(users),
        "orders":        orders,
        "books":         _flatten_books(books),
        "order_authors": _explode_order_authors(orders, books),
    }