# -*- coding: utf-8 -*-
# ============================================================
# app.py 量子神託（試作）— 縁の球体（QUBO × アート）
# 仕様（ご要望反映）:
# - QUOTESワークシート入りExcelをアップロードして読み込む（st.file_uploader）
# - 既定Excel（ローカル/同梱）がある場合はそれも読める
# - BGMは assets/bgm.mp4（st.audio + audio/mp4）
# - 点滅（フラッシュ）を排除：自動更新なし、星屑固定、同条件なら配置固定（seed固定）
# - Plotlyの線トレースを集約して軽量化（None区切り）
# - 「QUBOを使っている感」を右側で可視化（Q行列ヒートマップ/強い結びつき）
# - 「気になる単語→格言候補」導線
# ============================================================

import os
import re
import io
import zlib
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# pandas（Excel読み込み用）
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False
    pd = None


# ============================================================
# 0) ページ設定 + CSS
# ============================================================
st.set_page_config(page_title="量子神託 - 縁の球体", layout="wide")

SPACE_CSS = """
<style>
.stApp{
  background:
    radial-gradient(circle at 18% 24%, rgba(110,150,255,0.12), transparent 38%),
    radial-gradient(circle at 78% 68%, rgba(255,160,220,0.08), transparent 44%),
    radial-gradient(circle at 50% 50%, rgba(255,255,255,0.03), transparent 55%),
    linear-gradient(180deg, rgba(6,8,18,1), rgba(10,12,26,1));
}
.block-container{ padding-top: 1.2rem; }
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li{
  font-family: "Hiragino Mincho ProN", "Yu Mincho", "Noto Serif JP", serif;
  letter-spacing: 0.02em;
  color: rgba(245,245,255,0.92);
}
h1,h2,h3{
  font-family: "Hiragino Mincho ProN", "Yu Mincho", "Noto Serif JP", serif !important;
  font-weight: 600 !important;
  color: rgba(245,245,255,0.95);
}
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.08);
  border-right: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(10px);
}
div[data-testid="stPlotlyChart"] > div{
  position: relative;
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 18px 60px rgba(0,0,0,0.30);
}
div[data-testid="stPlotlyChart"] > div::after{
  content:"";
  position:absolute;
  inset:0;
  background:
    radial-gradient(circle at 30% 25%, rgba(120,160,255,0.10), transparent 45%),
    radial-gradient(circle at 70% 65%, rgba(255,180,220,0.06), transparent 52%),
    radial-gradient(circle at 50% 50%, rgba(0,0,0,0.00), rgba(0,0,0,0.38));
  pointer-events:none;
}
.smallnote{opacity:0.78; font-size:0.92rem;}
.card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 16px 10px 16px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.18);
}
</style>
"""
st.markdown(SPACE_CSS, unsafe_allow_html=True)


# ============================================================
# 0.5) セッション状態 初期化
# ============================================================
def init_session_state():
    defaults = {
        "bgm_on": False,
        "last_params_hash": "",
        "network": None,
        "pos": None,
        "keywords": [],
        "center_set": set(),
        "famous_quotes": None,          # Excel統合後の格言DB
        "quotes_meta": {"loaded": 0},   # 表示用
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ============================================================
# 1) グローバル単語DB（他の人の言葉）
# ============================================================
GLOBAL_WORDS_DATABASE = [
    "世界平和","貢献","成長","学び","挑戦","夢","希望","未来",
    "感謝","愛","幸せ","喜び","安心","充実","満足","平和",
    "努力","継続","忍耐","誠実","正直","優しさ","思いやり","共感",
    "調和","バランス","自然","美","真実","自由","正義","道",
    "絆","つながり","家族","友人","仲間","信頼","尊敬","協力",
    "今","瞬間","過程","変化","進化","発展","循環","流れ",
    "静けさ","集中","覚悟","決意","勇気","強さ","柔軟性","寛容",
]

CATEGORIES = {
    "願い": ["世界平和","貢献","成長","夢","希望","未来"],
    "感情": ["感謝","愛","幸せ","喜び","安心","満足","平和"],
    "行動": ["努力","継続","忍耐","誠実","正直"],
    "哲学": ["調和","バランス","自然","美","道","真実","自由","正義"],
    "関係": ["絆","つながり","家族","友人","仲間","信頼","尊敬","協力"],
    "内的": ["静けさ","集中","覚悟","決意","勇気","強さ","柔軟性","寛容"],
    "時間": ["今","瞬間","過程","変化","進化","発展","循環","流れ"],
}


# ============================================================
# 2) 格言DB（ベース + Excel QUOTES）
# ============================================================
BASE_FAMOUS_QUOTES = [
    {
        "keywords": ["平和","世界","貢献","希望"],
        "quote": "雪の下で種は春を待っている。焦るべからず、時満ちるを待て。",
        "source": "量子神託 試作（福田雅彦）—創作/寓話調",
        "note": "公開用に典拠付き格言へ差し替え可"
    },
    {
        "keywords": ["成長","努力","継続","挑戦"],
        "quote": "千里の道も一歩から。歩みを止めず、続けることに意味がある。",
        "source": "故事成語（要典拠確認）—暫定",
        "note": "短文化した定型句。公開前に典拠精査推奨"
    },
    {
        "keywords": ["感謝","愛","絆","つながり"],
        "quote": "一期一会。今この瞬間を大切に。すべては縁で繋がっている。",
        "source": "一期一会（茶道思想）＋量子神託 試作（編集/意訳）",
        "note": ""
    },
    {
        "keywords": ["自然","調和","バランス","流れ"],
        "quote": "水は、争わない。形にこだわらず、流れるがままに。",
        "source": "老子『道徳経』（上善若水）—意訳/編集",
        "note": "厳密な原文引用ではなく意訳"
    },
    {
        "keywords": ["静けさ","集中","今","瞬間"],
        "quote": "止まることで、流れが見える。動の中に静がある。",
        "source": "量子神託 試作（福田雅彦）—創作",
        "note": ""
    },
    {
        "keywords": ["勇気","決意","挑戦","道"],
        "quote": "道が分かれていたら、念のない方へ行け。",
        "source": "出典要確認（流通句）—暫定",
        "note": "公開前に典拠を確定推奨"
    },
    {
        "keywords": ["思いやり","優しさ","共感","信頼"],
        "quote": "人の心に寄り添う。それが真の強さである。",
        "source": "量子神託 試作（福田雅彦）—創作",
        "note": ""
    },
    {
        "keywords": ["変化","進化","発展","未来"],
        "quote": "無為にして為す。動くことが静である。",
        "source": "東洋思想（無為自然）—意訳/編集",
        "note": ""
    },
    {
        "keywords": ["美","真実","自然","調和"],
        "quote": "間こそが答えである。余白にこそ本質がある。",
        "source": "美学（間/余白）＋量子神託 試作（編集）",
        "note": ""
    },
    {
        "keywords": ["自由","正義","道","誠実"],
        "quote": "己に誠実であること。それが自由への道である。",
        "source": "量子神託 試作（福田雅彦）—創作",
        "note": ""
    },
]

# 既定Excel（ローカル/同梱向け）
# ※Streamlit Cloudでは存在しないので、基本はアップロード推奨
EXCEL_DEFAULT_PATH = "quantum_shintaku_pack_v3_with_sense_20260213_oposite_modify (2).xlsx"

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@st.cache_data(show_spinner=False)
def load_quotes_from_excel_bytes(excel_bytes: bytes, file_hash: str) -> List[Dict]:
    """Excelバイナリから QUOTES シートを読む（キャッシュ安定のため file_hash を引数に含める）"""
    if not PANDAS_AVAILABLE or not excel_bytes:
        return []
    try:
        bio = io.BytesIO(excel_bytes)
        df = pd.read_excel(bio, sheet_name="QUOTES", engine="openpyxl")
    except Exception:
        return []

    quotes: List[Dict] = []

    def pick_text(row, candidates):
        for col in candidates:
            if col in df.columns:
                v = str(row.get(col, "")).strip()
                if v and v.lower() not in ("nan", "none"):
                    return v
        return ""

    for _, row in df.iterrows():
        quote_text = pick_text(row, ["格言", "QUOTE", "Quote", "quote", "テキスト", "文", "言葉"])
        if not quote_text:
            continue

        kw_str = pick_text(row, ["キーワード", "KEYWORDS", "Keywords", "keywords", "タグ", "TAG", "Tag"])
        keywords = [k.strip() for k in kw_str.replace("、", ",").split(",") if k.strip()] if kw_str else []

        source = pick_text(row, ["出典", "SOURCE", "Source", "source", "出所", "典拠", "作者"]) or "伝統的な教え"
        note   = pick_text(row, ["備考", "NOTE", "Note", "note", "注", "メモ"])

        quotes.append({"quote": quote_text, "keywords": keywords, "source": source, "note": note})

    return quotes

def build_famous_quotes(excel_bytes: Optional[bytes] = None) -> Tuple[List[Dict], int]:
    """BASE +（任意で）ExcelのQUOTESを統合"""
    fam = list(BASE_FAMOUS_QUOTES)
    excel_quotes: List[Dict] = []

    if excel_bytes:
        h = _hash_bytes(excel_bytes)
        excel_quotes = load_quotes_from_excel_bytes(excel_bytes, file_hash=h)

    if excel_quotes:
        existing = {q.get("quote", "") for q in fam}
        for q in excel_quotes:
            qt = q.get("quote", "")
            if qt and qt not in existing:
                fam.append(q)
                existing.add(qt)

    return fam, len(excel_quotes)


# ============================================================
# 3) テキスト→キーワード抽出（簡易）
# ============================================================
STOP_TOKENS = set([
    "した","たい","いる","い","こと","それ","これ","ため","よう","ので","から",
    "です","ます","ある","ない","そして","でも","しかし","また",
    "自分","私","あなた","もの","感じ","気持ち","今","今日"
])

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    text = (text or "").strip()
    if not text:
        return ["静けさ", "迷い"]

    # まずはDB語の直接ヒット（優先）
    found = [w for w in GLOBAL_WORDS_DATABASE if w in text]
    if found:
        return found[:top_n]

    # ざっくり分割（依存増やさない方針）
    text_clean = re.sub(r"[0-9０-９、。．,.!！?？\(\)\[\]{}「」『』\"'：:;／/\\\n\r\t]+", " ", text)
    tokens = [t.strip() for t in re.split(r"\s+", text_clean) if t.strip()]
    tokens = [t for t in tokens if (len(t) >= 2 and t not in STOP_TOKENS)]

    if not tokens:
        return ["静けさ", "迷い"]

    tokens = sorted(tokens, key=lambda s: (-len(s), s))
    return tokens[:top_n]


# ============================================================
# 4) “エネルギー”計算（QUBO的相互作用）
# ============================================================
def calculate_semantic_similarity(word1: str, word2: str) -> float:
    if word1 == word2:
        return 1.0

    common_chars = set(word1) & set(word2)
    char_sim = len(common_chars) / max(len(set(word1)), len(set(word2)), 1)

    category_sim = 0.0
    for _, ws in CATEGORIES.items():
        w1_in = word1 in ws
        w2_in = word2 in ws
        if w1_in and w2_in:
            category_sim = 1.0
            break
        elif w1_in or w2_in:
            category_sim = max(category_sim, 0.3)

    len_sim = 1.0 - abs(len(word1) - len(word2)) / max(len(word1), len(word2), 1)
    similarity = 0.4 * char_sim + 0.4 * category_sim + 0.2 * len_sim
    return float(np.clip(similarity, 0.0, 1.0))

def calculate_energy_between_words(word1: str, word2: str, rng: np.random.Generator, jitter: float) -> float:
    similarity = calculate_semantic_similarity(word1, word2)
    # 似てるほどエネルギーが低い（結びつく）
    energy = -2.0 * similarity + 0.5

    common = set(word1) & set(word2)
    if common:
        energy -= 0.20 * len(common) / max(len(word1), len(word2), 1)

    for _, ws in CATEGORIES.items():
        if (word1 in ws) and (word2 in ws):
            energy -= 0.60
            break

    if jitter > 0:
        energy += rng.normal(0, jitter)
    return float(energy)

def build_qubo_matrix_for_words(words: List[str], rng: np.random.Generator, jitter: float) -> np.ndarray:
    n = len(words)
    Q = np.zeros((n, n), dtype=float)
    np.fill_diagonal(Q, -0.5)
    for i in range(n):
        for j in range(i + 1, n):
            e = calculate_energy_between_words(words[i], words[j], rng, jitter)
            Q[i, j] = e
            Q[j, i] = e
    return Q

def solve_qubo_placement(
    Q: np.ndarray,
    words: List[str],
    center_indices: List[int],
    energies: Dict[str, float],
    rng: np.random.Generator,
    n_iterations: int = 100,
    progress_callback=None,
) -> np.ndarray:
    n = len(words)
    pos = np.zeros((n, 3), dtype=float)
    for idx in center_indices:
        if idx < n:
            pos[idx] = [0.0, 0.0, 0.0]

    ev = list(energies.values()) if energies else []
    if ev:
        mn, mx = min(ev), max(ev)
        er = (mx - mn) if mx != mn else 1.0
    else:
        mn, er = -3.0, 3.0

    golden_angle = np.pi * (3 - np.sqrt(5))
    k = 0
    for i in range(n):
        if i in center_indices:
            continue
        w = words[i]
        e = energies.get(w, 0.0)
        norm = (e - mn) / er
        dist = 0.3 + (1.0 - norm) * 2.2

        theta = golden_angle * k
        y = 1 - (k / float(max(1, n - len(center_indices) - 1))) * 2
        r = np.sqrt(max(0.0, 1 - y * y))
        x = np.cos(theta) * r * dist
        z = np.sin(theta) * r * dist
        pos[i] = [x, y * dist * 0.6, z]
        k += 1

    for it in range(n_iterations):
        for i in range(n):
            if i in center_indices:
                continue
            force = np.zeros(3, dtype=float)

            # 中心との距離を保つ
            for cidx in center_indices:
                vec = pos[cidx] - pos[i]
                d = np.linalg.norm(vec)
                if d > 0.01:
                    w = words[i]
                    e = energies.get(w, 0.0)
                    norm = (e - mn) / er if er > 0 else 0.5
                    target = 0.3 + (1.0 - norm) * 2.2
                    if d < target * 0.9:
                        force -= vec / d * 0.05
                    elif d > target * 1.1:
                        force += vec / d * 0.10

            # Q相互作用
            for j in range(n):
                if i == j or j in center_indices:
                    continue
                eij = Q[i, j]
                if eij < -0.3:
                    vec = pos[j] - pos[i]
                    d = np.linalg.norm(vec)
                    if d > 0.01:
                        force += vec / d * (abs(eij) * 0.08)
                elif eij > 0.2:
                    vec = pos[i] - pos[j]
                    d = np.linalg.norm(vec)
                    if d > 0.01:
                        force += vec / d * (abs(eij) * 0.03)

            pos[i] += force * 0.15

        if progress_callback:
            progress_callback(it + 1, n_iterations)

    return pos

def build_word_network(center_words: List[str], database: List[str], n_total: int,
                       rng: np.random.Generator, jitter: float) -> Dict:
    all_words = list(dict.fromkeys(center_words + database))  # 順序維持
    energies: Dict[str, float] = {}

    for w in all_words:
        if w in center_words:
            energies[w] = -3.0
        else:
            e_list = [calculate_energy_between_words(c, w, rng, jitter) for c in center_words]
            energies[w] = float(np.mean(e_list))

    sorted_words = sorted(energies.items(), key=lambda x: x[1])  # 低いほど中心に近い
    selected: List[str] = []
    for w, _ in sorted_words:
        if w in center_words and w not in selected:
            selected.append(w)
    for w, _ in sorted_words:
        if w not in selected:
            selected.append(w)
        if len(selected) >= n_total:
            break

    Q = build_qubo_matrix_for_words(selected, rng, jitter)
    center_indices = [i for i, w in enumerate(selected) if w in center_words]

    edges: List[Tuple[int, int, float]] = []
    n = len(selected)
    for i in range(n):
        for j in range(i + 1, n):
            e = Q[i, j]
            if e < -0.25:
                edges.append((i, j, float(e)))

    return {
        "words": selected,
        "energies": {w: energies[w] for w in selected},
        "edges": edges,
        "Q": Q,
        "center_indices": center_indices,
    }


# ============================================================
# 5) 格言選択（出所つき）
# ============================================================
def select_relevant_quote(keywords: List[str], famous_quotes: List[Dict]) -> Dict[str, str]:
    if not keywords:
        keywords = ["今"]

    ks = set()
    for kw in keywords:
        k = kw.strip().lower()
        ks.add(k)
        if len(k) > 2:
            for i in range(len(k) - 1):
                ks.add(k[i:i+2])

    best = None
    best_score = -1.0

    for q in famous_quotes:
        qk = q.get("keywords", [])
        if not qk:
            continue

        qks = set()
        for k in qk:
            kk = k.strip().lower()
            qks.add(kk)
            if len(kk) > 2:
                for i in range(len(kk) - 1):
                    qks.add(kk[i:i+2])

        exact = len(ks & qks)
        partial = 0.0
        for a in ks:
            for b in qks:
                if a in b or b in a:
                    partial += 0.5

        text = q.get("quote", "").lower()
        text_match = 0.0
        for a in ks:
            if len(a) >= 2 and a in text:
                text_match += 0.3

        score = exact * 2.0 + partial + text_match
        if score > best_score:
            best_score = score
            best = q

    if best is None or best_score < 0.1:
        return {"quote": "あなたの観測が、この世界線を確定させました。", "source": "量子神託 試作（福田雅彦）—創作", "note": ""}

    return {"quote": best.get("quote", ""), "source": best.get("source", "伝統的な教え"), "note": best.get("note", "")}

def quote_candidates_for_word(word: str, famous_quotes: List[Dict], max_n: int = 6) -> List[Dict]:
    if not word:
        return []
    w = word.strip().lower()
    scored = []
    for q in famous_quotes:
        ks = [k.strip().lower() for k in q.get("keywords", [])]
        score = 0.0
        if w in ks:
            score += 3.0
        else:
            for k in ks:
                if w in k or k in w:
                    score += 1.0
        if w in (q.get("quote","").lower()):
            score += 0.5
        if score > 0:
            scored.append((score, q))
    scored.sort(key=lambda x: (-x[0], x[1].get("quote","")))
    return [q for _, q in scored[:max_n]]


# ============================================================
# 6) seed固定（同条件なら同配置）
# ============================================================
def make_seed(s: str) -> int:
    return int(zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF)


# ============================================================
# 7) UI（サイドバー）
# ============================================================
st.title("量子神託（試作）— 縁の球体（QUBO × アート）")

BGM_PATH = Path("assets/bgm.mp4")
BGM_FORMAT = "audio/mp4"

with st.sidebar:
    st.markdown("### 📘 格言データ（Excel/QUOTES）")

    if not PANDAS_AVAILABLE:
        st.error("pandas が利用できないため、Excel読み込みが無効です。requirements.txt に pandas と openpyxl を入れてください。")

    uploaded_excel = st.file_uploader(
        "QUOTESワークシート入りExcelを選択",
        type=["xlsx"],
        accept_multiple_files=False
    )

    excel_bytes: Optional[bytes] = None
    source_label = "BASEのみ"

    if uploaded_excel is not None:
        excel_bytes = uploaded_excel.getvalue()
        source_label = "アップロード"
        st.success("Excelを読み込み対象に設定しました（アップロード）")
    else:
        # 既定ファイルが同梱されていれば読む（任意）
        if os.path.exists(EXCEL_DEFAULT_PATH):
            try:
                with open(EXCEL_DEFAULT_PATH, "rb") as f:
                    excel_bytes = f.read()
                source_label = f"同梱: {EXCEL_DEFAULT_PATH}"
                st.info("Excelを読み込み対象に設定しました（同梱ファイル）")
            except Exception:
                excel_bytes = None
        else:
            st.caption("Excel未指定：BASE_FAMOUS_QUOTESのみで動作します。")

    # Famous quotes をセッションに保持（アップロード変更時のみ更新される）
    # ※uploaded_excel が変わったら st が再実行されるので自然に更新されます
    famous_quotes, excel_loaded = build_famous_quotes(excel_bytes)
    st.session_state["famous_quotes"] = famous_quotes
    st.session_state["quotes_meta"] = {"loaded": excel_loaded, "source": source_label}

import mimetypes

st.markdown("---")
st.markdown("### 🎵 音楽")

st.session_state["bgm_on"] = st.toggle(
    "BGMを再生（▶を押すと鳴ります）",
    value=st.session_state.get("bgm_on", False)
)

if st.session_state["bgm_on"]:
    if BGM_PATH.exists():
        bgm_bytes = BGM_PATH.read_bytes()

        # MIMEを自動判定
        mime, _ = mimetypes.guess_type(str(BGM_PATH))
        mime = mime or "audio/mp4"

        st.caption(f"形式: {mime} / サイズ: {len(bgm_bytes)/1024/1024:.2f} MB")

        try:
            # まず audio として再生
            st.audio(bgm_bytes, format=mime)

            # mp4が鳴らない環境向けフォールバック
            if str(BGM_PATH).lower().endswith(".mp4"):
                with st.expander("🔁 再生できない場合はこちら（videoフォールバック）"):
                    st.video(bgm_bytes, format="video/mp4")

        except Exception as e:
            st.error(f"再生エラー: {e}")
            st.caption("mp4で再生できない場合は mp3 形式への変換を推奨します。")
    else:
        st.error(f"⚠ BGMが見つかりません: {BGM_PATH}（assets/bgm.mp4 をGitHubに追加してください）")
    st.markdown("---")
    st.markdown("### 今の気持ち（入力）")
    user_input = st.text_area(
        "短い一文でOK（例：人との会話に疲れた。少し迷っている。）",
        value="人との会話に疲れた。少し迷っている。",
        height=90,
        key="user_input_text",
    )

    st.markdown("---")
    st.markdown("### パラメータ")
    top_n = st.slider("抽出キーワード数", 2, 10, 5, 1)
    n_total = st.slider("空間に出す単語数（中心＋周辺）", 15, 60, 30, 1)

    st.caption("※点滅防止のため、自動更新（ゆらぎ）は無効化しています。")

    noise = st.slider("位置のゆらぎ（再計算時のみ）", 0.00, 0.20, 0.06, 0.01)
    jitter = st.slider("エネルギー揺らぎ", 0.00, 0.25, 0.10, 0.01)
    qubo_iterations = st.slider("QUBO最適化の反復回数", 50, 200, 80, 10)

    st.markdown("---")
    st.markdown("### 宇宙の密度")
    star_count = st.slider("星屑の数", 200, 2200, 900, 50)
    st.caption("※星のまたたき（点滅）は無効化しています。")

    st.markdown("---")
    enable_zoom = st.toggle("マウスホイールでズーム", value=True)

    if st.button("🔄 再計算", use_container_width=True):
        st.session_state["last_params_hash"] = ""  # 強制再計算
        st.rerun()


# ============================================================
# 8) 再計算判定（静止表示）
# ============================================================
# Excelが変わった場合も配置を変える（格言DB由来の体験が変化するため）
quotes_meta = st.session_state.get("quotes_meta", {})
quotes_sig = f"{quotes_meta.get('source','')}_{quotes_meta.get('loaded',0)}"

params_hash = f"{user_input}|{top_n}|{n_total}|{noise}|{jitter}|{qubo_iterations}|{star_count}|{quotes_sig}"
needs_recalc = params_hash != st.session_state["last_params_hash"]


# ============================================================
# 9) 計算（ネットワーク / 配置）
# ============================================================
def compute_all():
    progress_placeholder = st.empty()

    seed = make_seed(params_hash)
    rng = np.random.default_rng(seed)

    with progress_placeholder.container():
        st.info("🔄 計算を開始します...")
        progress_bar = st.progress(0)
        status_text = st.empty()

    status_text.text("📝 キーワードを抽出中...")
    progress_bar.progress(10)
    keywords = extract_keywords(user_input, top_n=top_n)
    center_set = set(keywords)

    status_text.text("🔗 単語ネットワークを構築中...")
    progress_bar.progress(30)
    network = build_word_network(keywords, GLOBAL_WORDS_DATABASE, n_total=n_total, rng=rng, jitter=jitter)

    status_text.text("🌐 QUBO最適化で3D配置を計算中...")
    progress_bar.progress(50)

    def update_progress(current, total):
        p = 50 + int((current / total) * 40)
        progress_bar.progress(p)
        status_text.text(f"🌐 QUBO最適化中... ({current}/{total} 反復)")

    pos = solve_qubo_placement(
        network["Q"],
        network["words"],
        network["center_indices"],
        network["energies"],
        rng=rng,
        n_iterations=qubo_iterations,
        progress_callback=update_progress,
    )

    # 位置ゆらぎは再計算時だけ
    if noise > 0:
        pos = pos + rng.normal(0, noise, size=pos.shape)

    progress_bar.progress(100)
    status_text.text("✅ 計算完了！")
    time.sleep(0.08)
    progress_placeholder.empty()

    st.session_state["network"] = network
    st.session_state["pos"] = pos
    st.session_state["keywords"] = keywords
    st.session_state["center_set"] = center_set
    st.session_state["last_params_hash"] = params_hash

# 初回 or 変更時のみ計算（通常は静止）
if (st.session_state["network"] is None) or needs_recalc:
    compute_all()

network = st.session_state["network"]
pos = st.session_state["pos"]
keywords = st.session_state["keywords"]
center_set = st.session_state["center_set"]
FAMOUS_QUOTES = st.session_state.get("famous_quotes") or list(BASE_FAMOUS_QUOTES)

if network is None or pos is None or len(network.get("words", [])) == 0:
    st.warning("⚠️ データが不完全です。「🔄 再計算」を押してください。")
    st.stop()


# ============================================================
# 10) Plotly描画（星屑＋縁＋球体＋ラベル）
#     - 線を集約して軽量化（None区切り）
# ============================================================
fig = go.Figure()

# --- 星屑（点滅排除：完全固定） ---
star_rng = np.random.default_rng(12345)
sx = star_rng.uniform(-3.2, 3.2, star_count)
sy = star_rng.uniform(-2.4, 2.4, star_count)
sz = star_rng.uniform(-2.0, 2.0, star_count)
alpha = np.full(star_count, 0.22, dtype=float)
star_size = star_rng.uniform(1.0, 2.4, star_count)
star_colors = [f"rgba(255,255,255,{a})" for a in alpha]

fig.add_trace(go.Scatter3d(
    x=sx, y=sy, z=sz,
    mode="markers",
    marker=dict(size=star_size, color=star_colors),
    hoverinfo="skip",
    showlegend=False
))

words = network["words"]
energies = network.get("energies", {})
center_indices = network.get("center_indices", [])
edges = network.get("edges", [])

# --- 線：中心→周辺（集約） ---
xL, yL, zL, hoverL = [], [], [], []
for cidx in center_indices:
    if cidx >= len(words):
        continue
    cx, cy, cz = pos[cidx]
    cword = words[cidx]
    for i, w in enumerate(words):
        if i == cidx or i in center_indices:
            continue
        x, y, z = pos[i]
        e = energies.get(w, 0.0)
        d = float(np.linalg.norm(pos[i] - pos[cidx]))
        xL += [cx, x, None]
        yL += [cy, y, None]
        zL += [cz, z, None]
        hoverL += [f"{cword} → {w}<br>距離:{d:.2f}<br>エネルギー:{e:.2f}", "", ""]

fig.add_trace(go.Scatter3d(
    x=xL, y=yL, z=zL,
    mode="lines",
    line=dict(width=2, color="rgba(150,200,255,0.35)"),
    hoverinfo="text",
    text=hoverL,
    showlegend=False
))

# --- 線：単語間エッジ（集約） ---
xE, yE, zE, hoverE = [], [], [], []
for i, j, e in edges:
    if i in center_indices or j in center_indices:
        continue
    x0, y0, z0 = pos[i]
    x1, y1, z1 = pos[j]
    d = float(np.linalg.norm(pos[j] - pos[i]))
    xE += [x0, x1, None]
    yE += [y0, y1, None]
    zE += [z0, z1, None]
    hoverE += [f"{words[i]} ↔ {words[j]}<br>距離:{d:.2f}<br>エネルギー:{e:.2f}", "", ""]

fig.add_trace(go.Scatter3d(
    x=xE, y=yE, z=zE,
    mode="lines",
    line=dict(width=1, color="rgba(200,220,255,0.22)"),
    hoverinfo="text",
    text=hoverE,
    showlegend=False
))

# --- 球体（言葉） + ラベル色分け ---
sizes, colors, labels = [], [], []
for w in words:
    e = energies.get(w, 0.0)
    if w in center_set:
        sizes.append(28)
        colors.append("rgba(255,235,100,0.98)")
        labels.append(w)
    else:
        en = min(1.0, abs(e) / 3.0)
        sizes.append(12 + int(8 * en))
        if e < -1.5:
            colors.append("rgba(180,220,255,0.85)")
        elif e < -0.5:
            colors.append("rgba(220,240,255,0.75)")
        else:
            colors.append("rgba(255,255,255,0.60)")
        labels.append(w)

center_idx = [i for i, w in enumerate(labels) if w in center_set]
other_idx  = [i for i, w in enumerate(labels) if w not in center_set]

# ① それ以外（白文字）
if other_idx:
    oi = np.array(other_idx, dtype=int)
    fig.add_trace(go.Scatter3d(
        x=pos[oi, 0], y=pos[oi, 1], z=pos[oi, 2],
        mode="markers+text",
        text=[labels[i] for i in oi],
        textposition="top center",
        textfont=dict(size=18, color="rgba(255,255,255,1.0)"),
        marker=dict(
            size=[sizes[i] for i in oi],
            color=[colors[i] for i in oi],
            line=dict(width=1, color="rgba(0,0,0,0.10)")
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False
    ))

# ② 中心語（赤文字）
if center_idx:
    ci = np.array(center_idx, dtype=int)
    fig.add_trace(go.Scatter3d(
        x=pos[ci, 0], y=pos[ci, 1], z=pos[ci, 2],
        mode="markers+text",
        text=[labels[i] for i in ci],
        textposition="top center",
        textfont=dict(size=24, color="rgba(255,80,80,1.0)"),
        marker=dict(
            size=[sizes[i] for i in ci],
            color=[colors[i] for i in ci],
            line=dict(width=2, color="rgba(255,80,80,0.8)")
        ),
        hovertemplate="<b>%{text}</b><br>中心語<extra></extra>",
        showlegend=False
    ))

# 中心語の“光の層”（固定）
for cidx in center_indices:
    if cidx >= len(words):
        continue
    cx, cy, cz = pos[cidx]
    for layer, mult in enumerate([1.0, 1.3, 1.6], 1):
        opacity = 0.15 / layer
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode="markers",
            marker=dict(size=[35 * mult], color=f"rgba(150,200,255,{opacity})", line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False
        ))

fig.update_layout(
    paper_bgcolor="rgba(6,8,18,1)",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="rgba(6,8,18,1)",
        camera=dict(
            eye=dict(x=1.6, y=1.15, z=1.05),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0),
        ),
        dragmode="orbit",
    ),
    margin=dict(l=0, r=0, t=0, b=0),
)

plotly_config = {
    "displayModeBar": True,
    "scrollZoom": bool(enable_zoom),
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "filename": "quantum_oracle", "height": 800, "width": 1200, "scale": 1},
    "doubleClick": "reset",
}


# ============================================================
# 11) レイアウト（左：宇宙 / 右：格言+QUBO可視化）
# ============================================================
left, right = st.columns([2.0, 1.0], gap="large")

with left:
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)
    st.caption("単語（球体）と縁（線）。マウスで回転・ズームできます。（点滅なし・静止表示）")

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("### 📊 現在の状態")
    st.markdown(f"**核（推定キーワード）**：`{', '.join(keywords)}`")
    st.markdown(f"**計算済み単語数**: {len(words)}語")
    st.markdown(f"**接続数**: {len(edges)}本")
    if energies:
        mn = min(energies.values())
        mx = max(energies.values())
        st.markdown(f"**エネルギー範囲**: {mn:.2f} ～ {mx:.2f}")

    qm = st.session_state.get("quotes_meta", {})
    st.markdown(f"**格言DB**: {len(FAMOUS_QUOTES)}件（Excel読込: {qm.get('loaded',0)} / {qm.get('source','')}）")

    st.markdown("---")
    st.markdown("### 🧠 QUBO（第三者向け説明）")
    st.markdown(
        "- 各単語をノード、単語間の相互作用を **Q行列** に置きます。  \n"
        "- **似ているほどエネルギーが低い**（結びつく）ように設計しています。  \n"
        "- 配置は Q に基づく疑似最適化で “縁” が強い単語同士が近づくように調整します。"
    )

    with st.expander("QUBOの形（概念）", expanded=False):
        st.latex(r"E(\mathbf{x})=\sum_i Q_{ii}x_i + \sum_{i<j} Q_{ij}x_i x_j")
        st.markdown("<div class='smallnote'>※ 本アプリは「Q相互作用」を可視化・配置に利用しています（点滅なし・静止）。</div>", unsafe_allow_html=True)

    # Qヒートマップ（小さめ）
    Q = network["Q"]
    hm = go.Figure(data=go.Heatmap(z=Q, showscale=True))
    hm.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=220)
    st.plotly_chart(hm, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    # 強い結びつき（Qが低いペア）
    pairs = []
    n = Q.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((Q[i, j], i, j))
    pairs.sort(key=lambda x: x[0])  # 低いほど強い
    top_pairs = pairs[:8]

    with st.expander("強い結びつき（Qが低いペア）", expanded=False):
        for val, i, j in top_pairs:
            st.write(f"- {words[i]} ↔ {words[j]} : Q={val:.2f}")

    st.markdown("---")
    st.markdown("### 先人のことば")
    q = select_relevant_quote(keywords, FAMOUS_QUOTES)
    st.markdown(f"#### 「{q['quote']}」")
    st.markdown(f"**出所：** {q.get('source','—') if q.get('source') else '—'}")
    if q.get("note"):
        st.markdown(f"<div class='smallnote'>※ {q['note']}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👉 気になる単語から深掘り")
    default_word = keywords[0] if keywords else (words[0] if words else "")
    try:
        default_index = words.index(default_word) if default_word in words else 0
    except Exception:
        default_index = 0

    selected_word = st.selectbox("単語を選ぶ", options=words, index=default_index)
    cands = quote_candidates_for_word(selected_word, FAMOUS_QUOTES)

    if cands:
        st.markdown(f"**「{selected_word}」に関連する格言候補**")
        for qq in cands:
            st.markdown(
                f"- **{qq.get('quote','')}**  \n"
                f"  <span class='smallnote'>出所：{qq.get('source','—')}</span>",
                unsafe_allow_html=True
            )
    else:
        st.markdown("<div class='smallnote'>この単語に直接ヒットする格言は未登録です（ExcelのQUOTESを増やすと強化されます）。</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
