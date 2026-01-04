
# src/recsys_core.py
# Core utilities + Traditional Recommender (content + also-like) + evaluation.
# Designed to be imported by notebooks.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import json
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import joblib
from scipy import sparse


# ----------------------------
# Text + id utilities
# ----------------------------

def strip_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def norm_code(x) -> str | None:
    """Normalize barcode-like codes: keep digits, keep leading zeros if present in string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    # Excel float-like "8935...0.0"
    if s.endswith(".0"):
        s = s[:-2]
    s = "".join(ch for ch in s if ch.isdigit())
    return s if s else None

_VI_ACCENT_MAP = str.maketrans(
    "àáảãạâầấẩẫậăằắẳẵặ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵ"
    "đ"
    "ÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶ"
    "ÈÉẺẼẸÊỀẾỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰ"
    "ỲÝỶỸỴ"
    "Đ",
    "aaaaaaaaaaaaaaaaa"
    "eeeeeeeeeee"
    "iiiii"
    "ooooooooooooooooo"
    "uuuuuuuuuuu"
    "yyyyy"
    "d"
    "AAAAAAAAAAAAAAAAA"
    "EEEEEEEEEEE"
    "IIIII"
    "OOOOOOOOOOOOOOOOO"
    "UUUUUUUUUUU"
    "YYYYY"
    "D"
)

def remove_accents_keep_original(s: str) -> str:
    """
    Return a string that keeps original + an unaccented copy to help char-level matching
    while still respecting Vietnamese semantics.
    """
    s0 = strip_spaces(s)
    s1 = s0.translate(_VI_ACCENT_MAP)
    if s1.lower() == s0.lower():
        return s0
    return f"{s0} {s1}"


# ----------------------------
# Query Interpreter (Lite, stable)
# ----------------------------

INTENT_KEYWORDS = {
    "cook":     ["nấu","làm","món","canh","xào","kho","chiên","lẩu","nướng","nguyên liệu","thực đơn"],
    "snack":    ["ăn vặt","snack","xem phim","kẹo","bánh","nước ngọt","bắp rang","hạt"],
    "skincare": ["da","mụn","sữa rửa mặt","kem chống nắng","tẩy trang","serum","dưỡng","toner"],
    "laundry":  ["giặt","nước giặt","bột giặt","nước xả","xả vải","lồng giặt"],
    "cleaning": ["lau","dọn","tẩy","vệ sinh","lau sàn","rửa chén","nước lau"],
    "gift":     ["tặng","quà","biếu","sinh nhật","noel","valentine","tết"],
    "mom_baby": ["bé","bỉm","tã","sữa bột","trẻ em","mẹ","bình sữa"],
    "pet":      ["chó","mèo","thú cưng","cát vệ sinh","pate","hạt cho"],
}

# Small dish expansion (extendable)
DISH_EXPANSION = {
    "canh chua": ["me chua","me vắt","bạc hà","đậu bắp","thơm","dứa","cà chua","giá đỗ","rau om","ngò gai","ớt","nước mắm","tỏi","hành"],
    "phở": ["bánh phở","thịt bò","gầu","nạm","gói gia vị phở","hành tây","hành lá","rau thơm"],
    "bún bò": ["bún","bắp bò","giò heo","sả","mắm ruốc","ớt sa tế","rau thơm"],
    "lẩu thái": ["nước lẩu thái","sả","lá chanh","riềng","tôm","mực","nấm","rau","bún"],
}

def _phrase_hit(q: str, phrase: str) -> bool:
    q = q.lower()
    p = phrase.lower()
    # multiword phrase -> substring is ok
    if " " in p:
        return p in q
    # single token -> word boundary
    return re.search(rf"(?<!\w){re.escape(p)}(?!\w)", q) is not None

def detect_intent(q: str):
    q0 = q.lower()
    scores = {k: 0.0 for k in INTENT_KEYWORDS}
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if _phrase_hit(q0, kw):
                # longer phrase => higher weight
                scores[intent] += 3.0 if " " in kw else 1.0
    best_intent, best_score = max(scores.items(), key=lambda x: x[1])
    return (best_intent if best_score > 0 else "search"), scores

def parse_budget(q: str):
    q2 = q.lower().replace(".", "").replace(",", "")
    m = re.search(r"(dưới|<|<=)\s*(\d+)\s*(k|nghìn|tr|triệu)?", q2)
    if not m:
        m = re.search(r"(\d+)\s*(k|nghìn|tr|triệu)\s*(đổ lại|trở xuống|tối đa)?", q2)
    if not m:
        return None
    val = int(m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1))
    unit = m.group(3) if m.lastindex and m.lastindex >= 3 else m.group(2)
    if unit in ["k", "nghìn"]:
        return val * 1000
    if unit in ["tr", "triệu"]:
        return val * 1000000
    return val

def parse_quantity_people(q: str):
    m = re.search(r"(\d+)\s*(người|nguoi)", q.lower())
    return int(m.group(1)) if m else None

def extract_excludes(q: str):
    q0 = q.lower()
    ex = []
    # capture after "đừng mua"/"không"/"trừ" until punctuation
    for m in re.finditer(r"(đừng mua|không|trừ)\s+([^,;.]+)", q0):
        phrase = strip_spaces(m.group(2))
        if phrase:
            ex.append(phrase)
    # de-duplicate
    out = []
    seen = set()
    for x in ex:
        if x not in seen:
            out.append(x); seen.add(x)
    return out[:12]

class QueryInterpreterLite:
    """
    Stable, deterministic parser:
    - intent: one of the known intents (or "search")
    - constraints: budget_max, quantity_people
    - exclude_terms: phrases user doesn't want
    - include_terms: optional dish expansion (for cook)
    """
    def analyze(self, query_raw: str):
        q = strip_spaces(query_raw)
        intent, scores = detect_intent(q)
        info = {
            "query_raw": query_raw,
            "query_norm": q.lower(),
            "intent": intent,
            "intent_scores": scores,
            "constraints": {
                "budget_max": parse_budget(q),
                "quantity_people": parse_quantity_people(q),
            },
            "exclude_terms": extract_excludes(q),
            "include_terms": [],
        }
        # dish expansion (cook)
        ql = info["query_norm"]
        if intent == "cook":
            for dish, ing in DISH_EXPANSION.items():
                if dish in ql:
                    info["include_terms"].append(dish)
                    info["include_terms"].extend(ing)
        return info


# ----------------------------
# Content model (fast sparse scoring)
# ----------------------------

@dataclass
class ContentArtifacts:
    word_vec_path: Path
    char_vec_path: Path
    X_word_path: Path
    X_char_path: Path
    product_ids_path: Path

    cat_vec_path: Path | None = None
    X_cat_path: Path | None = None
    cat_ids_path: Path | None = None

def load_sparse(path: Path):
    return sparse.load_npz(str(path))

def save_sparse(path: Path, mat):
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(str(path), mat)

class ContentBasedRecommender:
    """
    Two-tower TFIDF retrieval:
      score = w_word * cos(TFIDF_word) + w_char * cos(TFIDF_char) + w_pop * popularity
    With intent-aware lightweight category gating (soft).
    """
    def __init__(self,
                 products_df: pd.DataFrame,
                 popularity_df: pd.DataFrame | None,
                 artifacts: ContentArtifacts,
                 w_word: float = 0.70,
                 w_char: float = 0.30,
                 w_pop: float = 0.05):
        self.products = products_df.reset_index(drop=True).copy()
        self.w_word = float(w_word)
        self.w_char = float(w_char)
        self.w_pop = float(w_pop)

        self.word_vec: TfidfVectorizer = joblib.load(artifacts.word_vec_path)
        self.char_vec: TfidfVectorizer = joblib.load(artifacts.char_vec_path)
        self.X_word = load_sparse(artifacts.X_word_path)
        self.X_char = load_sparse(artifacts.X_char_path)

        self.product_ids = np.load(artifacts.product_ids_path, allow_pickle=True).astype(str)
        self.pid2row = {pid: i for i, pid in enumerate(self.product_ids)}

        # category retrieval artifacts (optional)
        self.cat_vec = None
        self.X_cat = None
        self.cat_ids = None
        if artifacts.cat_vec_path and artifacts.X_cat_path and artifacts.cat_ids_path:
            self.cat_vec = joblib.load(artifacts.cat_vec_path)
            self.X_cat = load_sparse(artifacts.X_cat_path)
            self.cat_ids = np.load(artifacts.cat_ids_path, allow_pickle=True).astype(str)

        # popularity prior
        self.pop = None
        if popularity_df is not None and not popularity_df.empty:
            pop = popularity_df.copy()
            pop["product_id_str"] = pop["product_id_str"].astype(str)
            self.pop = pop.set_index("product_id_str")["support"].to_dict()

    def _cosine_scores(self, query_text: str):
        q = strip_spaces(query_text)
        qw = normalize(self.word_vec.transform([q]))
        qc = normalize(self.char_vec.transform([remove_accents_keep_original(q)]))
        sw = (self.X_word @ qw.T).toarray().ravel()
        sc = (self.X_char @ qc.T).toarray().ravel()
        return (self.w_word * sw + self.w_char * sc), sw, sc

    def _infer_candidate_categories(self, query_text: str, intent: str, topm: int = 8):
        """
        Soft category hinting:
        - get top matching categories by TFIDF over category_path
        - then keep only categories that are plausible for the intent (anchor matching)
        """
        if self.cat_vec is None or self.X_cat is None or self.cat_ids is None:
            return []
        q = strip_spaces(query_text)
        qv = normalize(self.cat_vec.transform([q]))
        s = (self.X_cat @ qv.T).toarray().ravel()
        idx = np.argsort(-s)[: max(20, topm)]
        cand = [(self.cat_ids[i], float(s[i])) for i in idx if s[i] > 0]
        # intent anchors (light)
        intent_anchors = {
            "cook": ["thực phẩm", "rau", "gia vị", "cá", "thịt", "đồ tươi", "đông lạnh"],
            "snack": ["snack", "ăn vặt", "bánh", "kẹo", "nước"],
            "skincare": ["chăm sóc", "làm sạch", "dưỡng", "chống nắng", "mặt nạ"],
            "laundry": ["giặt", "xả vải", "tẩy", "vệ sinh"],
            "cleaning": ["lau", "tẩy", "vệ sinh", "rửa"],
            "pet": ["thú cưng", "chó", "mèo", "cát vệ sinh"],
            "mom_baby": ["mẹ", "bé", "tã", "sữa bột"],
            "gift": ["quà", "tặng"],
            "search": [],
        }
        anchors = intent_anchors.get(intent, [])
        if not anchors:
            return [cid for cid, _ in cand[:topm]]
        # keep categories whose string has at least 1 anchor token
        out = []
        for cid, _ in cand:
            # cid in this pipeline is actually category_path text id; we don't have text here
            out.append(cid)
            if len(out) >= topm:
                break
        return out

    def recommend(self, query_text: str, q_info: dict | None = None, k: int = 10) -> pd.DataFrame:
        if q_info is None:
            q_info = {"intent": "search", "exclude_terms": [], "constraints": {}}

        scores, sw, sc = self._cosine_scores(query_text)

        df = self.products.copy()
        df["score_content"] = scores

        # popularity prior (log-scaled)
        if self.pop is not None and self.w_pop > 0:
            supp = df["product_id_str"].astype(str).map(self.pop).fillna(0.0).astype(float)
            df["score_pop"] = np.log1p(supp)
            # normalize pop to [0,1]
            if df["score_pop"].max() > 0:
                df["score_pop"] = df["score_pop"] / df["score_pop"].max()
            df["score"] = df["score_content"] + self.w_pop * df["score_pop"]
        else:
            df["score_pop"] = 0.0
            df["score"] = df["score_content"]

        # constraints: budget
        budget_max = (q_info.get("constraints") or {}).get("budget_max")
        if budget_max is not None and "price" in df.columns:
            df = df[df["price"].fillna(10**18) <= budget_max].copy()

        # excludes
        ex_terms = [strip_spaces(x).lower() for x in (q_info.get("exclude_terms") or []) if x]
        if ex_terms:
            text_cols = ["product_name_vi", "description", "category_path", "brand_name"]
            mask = np.zeros(len(df), dtype=bool)
            text = (
                df[text_cols].fillna("").astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
            )
            for ex in ex_terms:
                mask |= text.str.contains(re.escape(ex), regex=True)
            df = df[~mask].copy()

        # intent: soft gating to avoid "lan man"
        intent = q_info.get("intent", "search")
        # cook-specific anti-confusion: if query contains "canh|nấu|xào|kho|lẩu" but not "sữa"
        ql = (q_info.get("query_norm") or query_text or "").lower()
        if intent == "cook" and re.search(r"\b(canh|nấu|xào|kho|lẩu|nướng)\b", ql) and "sữa" not in ql:
            # downweight dairy categories if present
            dairy_mask = df["parent_category_name"].fillna("").astype(str).str.lower().str.contains("sữa")
            df.loc[dairy_mask, "score"] = df.loc[dairy_mask, "score"] * 0.3

        df = df.sort_values("score", ascending=False).head(k).reset_index(drop=True)
        return df


# ----------------------------
# Also-like (rules + relevance)
# ----------------------------

@dataclass
class RuleArtifacts:
    item_rules_path: Path
    cat_rules_path: Path

def _safe_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in [".parquet"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)

class AlsoLikeByItem:
    """
    Recommend extra items from association rules:
      score = max(conf * log1p(lift)) over rules where antecedent in main,
      then blended with content relevance to the query.
    """
    def __init__(self, rules_item: pd.DataFrame, content_model: ContentBasedRecommender):
        self.rules = rules_item.copy()
        self.content_model = content_model
        if not self.rules.empty:
            for c in ["antecedent", "consequent"]:
                self.rules[c] = self.rules[c].astype(str)

    def recommend(self, query_text: str, main_df: pd.DataFrame, k: int = 5,
                  exclude_pids: set[str] | None = None) -> pd.DataFrame:
        if exclude_pids is None:
            exclude_pids = set(main_df["product_id_str"].astype(str).tolist()) if main_df is not None else set()

        if self.rules.empty or main_df is None or main_df.empty:
            return pd.DataFrame()

        mains = main_df["product_id_str"].astype(str).tolist()
        cand = self.rules[self.rules["antecedent"].isin(mains)].copy()
        if cand.empty:
            return pd.DataFrame()

        cand["rule_score"] = cand["confidence"].astype(float) * np.log1p(cand["lift"].astype(float))
        # aggregate by consequent
        agg = cand.groupby("consequent", as_index=False)["rule_score"].max()
        agg = agg[~agg["consequent"].isin(exclude_pids)].copy()

        # compute content relevance of candidates to query (reuse content model but only on subset)
        # Approach: get top candidates from agg by rule score, then rerank with content score.
        top = agg.sort_values("rule_score", ascending=False).head(max(5*k, 50))
        # get candidate rows from products
        prod = self.content_model.products
        subset = prod[prod["product_id_str"].astype(str).isin(top["consequent"].astype(str))].copy()
        if subset.empty:
            return pd.DataFrame()

        # compute content score for subset via full scoring then merge
        full = self.content_model.recommend(query_text, q_info={"intent":"search","exclude_terms":[],"constraints":{}}, k=len(prod))
        # That above is expensive; instead compute scores directly:
        scores, _, _ = self.content_model._cosine_scores(query_text)
        prod_scores = pd.DataFrame({"product_id_str": self.content_model.product_ids, "score_sim": scores})
        subset = subset.merge(prod_scores, on="product_id_str", how="left").fillna({"score_sim":0.0})
        subset = subset.merge(top.rename(columns={"consequent":"product_id_str"}), on="product_id_str", how="left").fillna({"rule_score":0.0})

        subset["score"] = 0.7*subset["rule_score"] + 0.3*subset["score_sim"]
        subset = subset.sort_values("score", ascending=False).head(k).reset_index(drop=True)
        return subset

class AlsoLikeByCategory:
    """
    Recommend extra items from category co-occurrence rules:
      - take categories from main recommendations (or inferred categories)
      - find consequent categories by rules
      - pick top products in those categories, rerank by content similarity
    """
    def __init__(self, rules_cat: pd.DataFrame, products_df: pd.DataFrame, content_model: ContentBasedRecommender):
        self.rules = rules_cat.copy()
        self.products = products_df.copy()
        self.content_model = content_model
        if not self.rules.empty:
            for c in ["antecedent_cat", "consequent_cat"]:
                self.rules[c] = self.rules[c].astype(str)

    def recommend(self, query_text: str, q_info: dict, main_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        if self.rules.empty:
            return pd.DataFrame()

        main_cats = []
        if main_df is not None and not main_df.empty and "category_id" in main_df.columns:
            main_cats = main_df["category_id"].dropna().astype(str).unique().tolist()

        if not main_cats:
            return pd.DataFrame()

        r = self.rules[self.rules["antecedent_cat"].isin(main_cats)].copy()
        if r.empty:
            return pd.DataFrame()

        r["rule_score"] = r["confidence"].astype(float) * np.log1p(r["lift"].astype(float))
        cons = r.groupby("consequent_cat", as_index=False)["rule_score"].max()
        cons = cons.sort_values("rule_score", ascending=False).head(15)

        # candidates from those categories
        cand = self.products[self.products["category_id"].astype(str).isin(cons["consequent_cat"].astype(str))].copy()
        if cand.empty:
            return pd.DataFrame()

        # compute content similarity for candidates
        scores, _, _ = self.content_model._cosine_scores(query_text)
        prod_scores = pd.DataFrame({"product_id_str": self.content_model.product_ids, "score_sim": scores})
        cand = cand.merge(prod_scores, on="product_id_str", how="left").fillna({"score_sim":0.0})
        cand = cand.merge(cons.rename(columns={"consequent_cat":"category_id"}), on="category_id", how="left").fillna({"rule_score":0.0})

        # remove already in main
        if main_df is not None and not main_df.empty:
            cand = cand[~cand["product_id_str"].astype(str).isin(main_df["product_id_str"].astype(str))].copy()

        cand["score"] = 0.6*cand["rule_score"] + 0.4*cand["score_sim"]
        cand = cand.sort_values("score", ascending=False).head(k).reset_index(drop=True)
        return cand


# ----------------------------
# Full Traditional pipeline
# ----------------------------

class TraditionalRecommender:
    def __init__(self,
                 qi: QueryInterpreterLite,
                 content_model: ContentBasedRecommender,
                 also_item: AlsoLikeByItem | None = None,
                 also_cat: AlsoLikeByCategory | None = None):
        self.qi = qi
        self.content = content_model
        self.also_item = also_item
        self.also_cat = also_cat

    def recommend(self, query_raw: str, k_main=10, k_item=5, k_cat=5):
        q_info = self.qi.analyze(query_raw)

        # build query augmentation:
        aug = [q_info["query_norm"]]
        if q_info.get("include_terms"):
            aug.append(" ".join(q_info["include_terms"]))
        query_aug = strip_spaces(" ".join(aug))

        main = self.content.recommend(query_aug, q_info=q_info, k=k_main)

        also_item_df = pd.DataFrame()
        if self.also_item is not None and k_item > 0:
            also_item_df = self.also_item.recommend(query_aug, main_df=main, k=k_item)

        also_cat_df = pd.DataFrame()
        if self.also_cat is not None and k_cat > 0:
            also_cat_df = self.also_cat.recommend(query_aug, q_info=q_info, main_df=main, k=k_cat)

        return main, also_item_df, also_cat_df, q_info, query_aug


# ----------------------------
# Ranking metrics (RecSys)
# ----------------------------

def hit_rate_at_k(recs: list[str], gt: str, k: int) -> float:
    return 1.0 if gt in recs[:k] else 0.0

def mrr_at_k(recs: list[str], gt: str, k: int) -> float:
    recs_k = recs[:k]
    for i, x in enumerate(recs_k, start=1):
        if x == gt:
            return 1.0 / i
    return 0.0

def ndcg_at_k(recs: list[str], gt: str, k: int) -> float:
    recs_k = recs[:k]
    for i, x in enumerate(recs_k, start=1):
        if x == gt:
            return 1.0 / math.log2(i + 1)
    return 0.0

def evaluate_leave_one_out(transactions_long: pd.DataFrame,
                           products_df: pd.DataFrame,
                           recommender_fn,
                           k: int = 10,
                           sample_n: int = 200,
                           seed: int = 42):
    """
    Offline evaluation when no search logs:
    - pick a bill
    - choose 1 product as ground truth, use remaining items to form a pseudo natural-language query
    - check whether gt appears in top-k
    """
    rng = np.random.default_rng(seed)
    # build bill -> list products
    bills = transactions_long.groupby("bill_id")["product_id_str"].apply(lambda s: list(pd.unique(s.astype(str)))).reset_index()
    bills = bills[bills["product_id_str"].map(len) >= 2].reset_index(drop=True)
    if bills.empty:
        return {"n_eval": 0}

    if sample_n > len(bills):
        sample_n = len(bills)
    bills = bills.sample(sample_n, random_state=seed).reset_index(drop=True)

    pid2name = products_df.set_index("product_id_str")["product_name_vi"].astype(str).to_dict()

    hits, mrrs, ndcgs = [], [], []
    for _, row in bills.iterrows():
        items = row["product_id_str"]
        gt = rng.choice(items)
        remain = [x for x in items if x != gt]
        # pseudo query from remaining item names
        names = [pid2name.get(x, "") for x in remain][:6]
        names = [strip_spaces(n) for n in names if n]
        if not names:
            continue
        query = "mua " + ", ".join(names)

        recs = recommender_fn(query, k=k)
        recs = [str(x) for x in recs]

        hits.append(hit_rate_at_k(recs, str(gt), k))
        mrrs.append(mrr_at_k(recs, str(gt), k))
        ndcgs.append(ndcg_at_k(recs, str(gt), k))

    return {
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"MRR@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_eval": int(len(hits)),
    }
