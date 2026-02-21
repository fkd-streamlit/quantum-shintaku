# -*- coding: utf-8 -*-
# ============================================================
# QUBO × 量子神託 UI（Streamlit + Plotly）
# 目的：点滅（フラッシュ）を排除し、静的で見やすい表示へ
# - 自動更新（st_autorefresh）を廃止
# - 星屑のまたたきを固定（time seedを使わない）
# - 位置のゆらぎは再計算時のみ（通常は静止）
# ============================================================

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
</style>
"""
st.markdown(SPACE_CSS, unsafe_allow_html=True)

# ============================================================
# 0.5) セッション状態 初期化（落ちないための基盤）
# ============================================================
def init_session_state():
    defaults = {
        "bgm_on": True,
        "last_user_input": "",
        "last_params_hash": "",
        "network": None,
        "pos": None,
        "keywords": [],
        "center_set": set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ============================================================
# 0.6) BGM（サイドバーのみ）
# ============================================================
from pathlib import Path
import streamlit as st

BGM_PATH = Path("assets/bgm.mp4")
BGM_FORMAT = "audio/mpeg"  # mp3はこれが安定

if "bgm_on" not in st.session_state:
    st.session_state["bgm_on"] = False  # 初期はOFF推奨（自動再生と誤解されるため）

with st.sidebar:
    st.markdown("### 🎵 音楽")
    st.session_state["bgm_on"] = st.toggle("BGMを再生（▶を押すと鳴ります）", value=st.session_state["bgm_on"])

    if st.session_state["bgm_on"]:
        if BGM_PATH.exists():
            audio_bytes = BGM_PATH.read_bytes()
            st.audio(audio_bytes, format=BGM_FORMAT)
            st.caption("※ブラウザ制限により自動再生はできません。▶ を押してください。")
        else:
            st.error(f"⚠ BGMが見つかりません: {BGM_PATH}（assets/bgm.mp3 をGitHubに追加してください）")

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

# ============================================================
# 2) 格言DB（出所も持たせる）
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
        "source": "故事成語（老子/荀子等に類する表現として流通）—要典拠確認",
        "note": "短文化した定型句。公開前に典拠精査推奨"
    },
    {
        "keywords": ["感謝","愛","絆","つながり"],
        "quote": "一期一会。今この瞬間を大切に。すべては縁で繋がっている。",
        "source": "一期一会（茶道思想）＋量子神託 試作（福田雅彦）—編集/意訳",
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
        "source": "美学（間/余白）＋量子神託 試作（福田雅彦）—編集",
        "note": ""
    },
    {
        "keywords": ["自由","正義","道","誠実"],
        "quote": "己に誠実であること。それが自由への道である。",
        "source": "量子神託 試作（福田雅彦）—創作",
        "note": ""
    },
]

EXCEL_DEFAULT = "quantum_shintaku_pack_v3_with_sense_20260213_oposite_modify_with_lr022101.xlsx"

@st.cache_data(show_spinner=False)
def load_quotes_from_excel_cached(excel_path: str) -> List[Dict]:
    if not PANDAS_AVAILABLE:
        return []
    if not excel_path or not os.path.exists(excel_path):
        return []
    try:
        df = pd.read_excel(excel_path, sheet_name="QUOTES", engine="openpyxl")
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
        note = pick_text(row, ["備考", "NOTE", "Note", "note", "注", "メモ"])

        quotes.append({"quote": quote_text, "keywords": keywords, "source": source, "note": note})

    return quotes

def build_famous_quotes() -> List[Dict]:
    fam = list(BASE_FAMOUS_QUOTES)
    excel_quotes = load_quotes_from_excel_cached(EXCEL_DEFAULT)
    if excel_quotes:
        existing = {q.get("quote", "") for q in fam}
        for q in excel_quotes:
            qt = q.get("quote", "")
            if qt and qt not in existing:
                fam.append(q)
                existing.add(qt)
    return fam

FAMOUS_QUOTES = build_famous_quotes()

# ============================================================
# 3) テキスト→キーワード抽出（簡易）
# ============================================================
def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    text = (text or "").strip()
    text_clean = re.sub(r"[0-9０-９\W]+", " ", text)

    found = [w for w in GLOBAL_WORDS_DATABASE if w in text_clean]
    if found:
        return found[:top_n]

    tokens = [t for t in text_clean.split() if len(t) >= 2]
    if not tokens:
        return ["静けさ", "迷い"]
    return tokens[:top_n]

# ============================================================
# 4) “エネルギー”計算（QUBO的相互作用）
# ============================================================
CATEGORIES = {
    "願い": ["世界平和","貢献","成長","夢","希望","未来"],
    "感情": ["感謝","愛","幸せ","喜び","安心","満足","平和"],
    "行動": ["努力","継続","忍耐","誠実","正直"],
    "哲学": ["調和","バランス","自然","美","道","真実","自由","正義"],
    "関係": ["絆","つながり","家族","友人","仲間","信頼","尊敬","協力"],
    "内的": ["静けさ","集中","覚悟","決意","勇気","強さ","柔軟性","寛容"],
    "時間": ["今","瞬間","過程","変化","進化","発展","循環","流れ"],
}

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
            category_sim = 0.3

    len_sim = 1.0 - abs(len(word1) - len(word2)) / max(len(word1), len(word2), 1)
    similarity = 0.4 * char_sim + 0.4 * category_sim + 0.2 * len_sim
    return float(np.clip(similarity, 0.0, 1.0))

def calculate_energy_between_words(word1: str, word2: str, rng: np.random.Generator, jitter: float) -> float:
    similarity = calculate_semantic_similarity(word1, word2)
    energy = -2.0 * similarity + 0.5

    common = set(word1) & set(word2)
    if common:
        energy -= 0.20 * len(common) / max(len(word1), len(word2), 1)

    for _, ws in CATEGORIES.items():
        if (word1 in ws) and (word2 in ws):
            energy -= 0.60
            break

    energy += rng.normal(0, jitter)
    return float(energy)

def build_qubo_matrix_for_words(words: List[str], rng: np.random.Generator, jitter: float) -> Dict[Tuple[int, int], float]:
    n = len(words)
    Q: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        Q[(i, i)] = -0.5
    for i in range(n):
        for j in range(i + 1, n):
            e = calculate_energy_between_words(words[i], words[j], rng, jitter)
            Q[(i, j)] = e
            Q[(j, i)] = e
    return Q

def solve_qubo_placement(
    Q: Dict[Tuple[int, int], float],
    n_words: int,
    center_indices: List[int],
    rng: np.random.Generator,
    n_iterations: int = 100,
    progress_callback=None,
    energies_dict: Dict[str, float] | None = None,
    words_list: List[str] | None = None,
) -> np.ndarray:
    pos = np.zeros((n_words, 3), dtype=float)
    for idx in center_indices:
        if idx < n_words:
            pos[idx] = [0.0, 0.0, 0.0]

    energies_dict = energies_dict or {}
    words_list = words_list or []

    energy_values = list(energies_dict.values()) if energies_dict else []
    if energy_values:
        min_energy = min(energy_values)
        max_energy = max(energy_values)
        energy_range = max_energy - min_energy if max_energy != min_energy else 1.0
    else:
        min_energy = -3.0
        energy_range = 3.0

    golden_angle = np.pi * (3 - np.sqrt(5))
    word_idx = 0

    for i in range(n_words):
        if i in center_indices:
            continue

        if i < len(words_list):
            w = words_list[i]
            e = energies_dict.get(w, 0.0)
        else:
            e = 0.0

        normalized = (e - min_energy) / energy_range if energy_range > 0 else 0.5
        distance = 0.3 + (1.0 - normalized) * 2.2

        theta = golden_angle * word_idx
        y = 1 - (word_idx / float(max(1, n_words - len(center_indices) - 1))) * 2
        radius_at_y = np.sqrt(max(0.0, 1 - y * y))

        x = np.cos(theta) * radius_at_y * distance
        z = np.sin(theta) * radius_at_y * distance
        pos[i] = [x, y * distance * 0.6, z]
        word_idx += 1

    for it in range(n_iterations):
        for i in range(n_words):
            if i in center_indices:
                continue

            force = np.zeros(3, dtype=float)

            for cidx in center_indices:
                vec = pos[cidx] - pos[i]
                dist = np.linalg.norm(vec)
                if dist > 0.01:
                    if i < len(words_list):
                        w = words_list[i]
                        e = energies_dict.get(w, 0.0)
                    else:
                        e = 0.0
                    target = 0.3 + (1.0 - (e - min_energy) / energy_range) * 2.2 if energy_range > 0 else 1.5

                    if dist < target * 0.9:
                        force -= vec / dist * 0.05
                    elif dist > target * 1.1:
                        force += vec / dist * 0.1

            for j in range(n_words):
                if i == j or j in center_indices:
                    continue
                eij = Q.get((i, j), 0.0)
                if eij < -0.3:
                    vec = pos[j] - pos[i]
                    dist = np.linalg.norm(vec)
                    if dist > 0.01:
                        force += vec / dist * (abs(eij) * 0.08)
                elif eij > 0.2:
                    vec = pos[i] - pos[j]
                    dist = np.linalg.norm(vec)
                    if dist > 0.01:
                        force += vec / dist * (abs(eij) * 0.03)

            pos[i] += force * 0.15

        if progress_callback:
            progress_callback(it + 1, n_iterations)

    return pos

def build_word_network(center_words: List[str], database: List[str], n_total: int,
                       rng: np.random.Generator, jitter: float) -> Dict:
    all_words = list(set(center_words + database))
    energies = {}

    for w in all_words:
        if w in center_words:
            energies[w] = -3.0
        else:
            e_list = [calculate_energy_between_words(c, w, rng, jitter) for c in center_words]
            energies[w] = float(np.mean(e_list))

    sorted_words = sorted(energies.items(), key=lambda x: x[1])

    selected = []
    for w, _ in sorted_words:
        if w in center_words:
            selected.append(w)
    for w, _ in sorted_words:
        if w not in selected:
            selected.append(w)
        if len(selected) >= n_total:
            break

    Q = build_qubo_matrix_for_words(selected, rng, jitter)

    edges = []
    center_indices = [i for i, w in enumerate(selected) if w in center_words]
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            e = Q.get((i, j), 0.0)
            if e < -0.25:
                edges.append((i, j, e))

    return {
        "words": selected,
        "energies": {w: energies[w] for w in selected},
        "edges": edges,
        "qubo_matrix": Q,
        "center_indices": center_indices
    }

def place_words_3d(words: List[str], center_set: set, rng: np.random.Generator,
                   noise: float, network: Dict, n_iterations: int,
                   progress_callback=None) -> np.ndarray:
    n = len(words)
    Q = network["qubo_matrix"]
    center_indices = network["center_indices"]
    energies_dict = network.get("energies", {})
    pos = solve_qubo_placement(
        Q, n, center_indices, rng,
        n_iterations=n_iterations,
        progress_callback=progress_callback,
        energies_dict=energies_dict,
        words_list=words
    )
    # 位置ゆらぎは「再計算時だけ」加える（通常表示は静止）
    if noise > 0:
        pos += rng.normal(0, noise, size=pos.shape)
    return pos

# ============================================================
# 6) 格言選択（出所つき）
# ============================================================
def select_relevant_quote(keywords: List[str]) -> Dict[str, str]:
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

    for q in FAMOUS_QUOTES:
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

# ============================================================
# 7) UI（サイドバー）
# ============================================================
st.title("量子神託（試作）— 縁の球体（QUBO × アート）")

with st.sidebar:
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

    # 点滅を排除するため、自動更新は廃止（トグルも撤去）
    st.caption("※点滅防止のため、自動更新（ゆらぎ）は無効化しています。")

    noise = st.slider("位置のゆらぎ（再計算時のみ）", 0.00, 0.20, 0.06, 0.01)
    jitter = st.slider("エネルギー揺らぎ", 0.00, 0.25, 0.10, 0.01)

    qubo_iterations = st.slider(
        "QUBO最適化の反復回数", 50, 200, 80, 10,
        help="少ないほど速いが、配置の精度は下がります",
    )

    st.markdown("---")
    st.markdown("### 宇宙の密度")
    star_count = st.slider("星屑の数", 200, 2200, 900, 50)
    # 点滅防止：またたきは固定（スライダーを撤去）
    st.caption("※星のまたたき（点滅）は無効化しています。")

    st.markdown("---")
    enable_zoom = st.toggle("マウスホイールでズーム", value=True)

    if st.button("🔄 再計算", use_container_width=True):
        st.session_state["last_user_input"] = ""
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎵 音楽")
    st.session_state["bgm_on"] = st.toggle("BGMを再生", value=st.session_state["bgm_on"])
    if st.session_state["bgm_on"]:
        if BGM_PATH.exists():
            st.audio(BGM_PATH.read_bytes(), format=BGM_FORMAT)
        else:
            st.caption("⚠ assets/bgm.mp3 が見つかりません（GitHubに追加してください）")

# ============================================================
# 7.5) 再計算判定（静止表示）
# ============================================================
params_hash = f"{user_input}_{top_n}_{n_total}_{noise}_{jitter}_{qubo_iterations}_{star_count}"
input_changed = user_input != st.session_state["last_user_input"]
params_changed = params_hash != st.session_state["last_params_hash"]
needs_recalc = input_changed or params_changed

if needs_recalc:
    st.session_state["last_user_input"] = user_input
    st.session_state["last_params_hash"] = params_hash

# ============================================================
# 8) 計算（ネットワーク / 配置）
# ============================================================
def compute_all():
    progress_placeholder = st.empty()
    rng = np.random.default_rng(int(time.time() * 1000) % (2**32 - 1))

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

    pos = place_words_3d(
        network["words"],
        center_set=center_set,
        rng=rng,
        noise=noise,
        network=network,
        n_iterations=qubo_iterations,
        progress_callback=update_progress,
    )

    progress_bar.progress(100)
    status_text.text("✅ 計算完了！")
    time.sleep(0.15)
    progress_placeholder.empty()

    st.session_state["network"] = network
    st.session_state["pos"] = pos
    st.session_state["keywords"] = keywords
    st.session_state["center_set"] = center_set

# 初回 or 変更時のみ計算（通常は静止）
if (st.session_state["network"] is None) or needs_recalc:
    compute_all()

network = st.session_state["network"]
pos = st.session_state["pos"]
keywords = st.session_state["keywords"]
center_set = st.session_state["center_set"]

# ============================================================
# 9) Plotly描画（星屑＋縁＋球体＋ラベル）
# ============================================================
if network is None or pos is None or len(network.get("words", [])) == 0:
    st.warning("⚠️ データが不完全です。「🔄 再計算」を押してください。")
    st.stop()

fig = go.Figure()

# --- 星屑（点滅排除：完全固定） ---
# 固定seedで配置も透明度もサイズも固定
star_rng = np.random.default_rng(12345)
sx = star_rng.uniform(-3.2, 3.2, star_count)
sy = star_rng.uniform(-2.4, 2.4, star_count)
sz = star_rng.uniform(-2.0, 2.0, star_count)

# 透明度固定（点滅しない）
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
energies_dict = network.get("energies", {})
center_indices = network.get("center_indices", [])

# --- 中心語→各単語の線 ---
for cidx in center_indices:
    if cidx >= len(words):
        continue
    center_word = words[cidx]
    cx, cy, cz = pos[cidx]

    for i, w in enumerate(words):
        if i == cidx or i in center_indices:
            continue
        x, y, z = pos[i]
        energy = energies_dict.get(w, 0.0)
        distance = float(np.linalg.norm(pos[i] - pos[cidx]))

        en = min(1.0, abs(energy) / 3.0)
        lw = 1.0 + 3.0 * en
        a = 0.3 + 0.5 * en

        if distance < 1.0:
            color = f"rgba(100,200,255,{a})"
        elif distance < 1.8:
            color = f"rgba(150,200,255,{a * 0.7})"
        else:
            color = f"rgba(200,220,255,{a * 0.4})"

        fig.add_trace(go.Scatter3d(
            x=[cx, x], y=[cy, y], z=[cz, z],
            mode="lines",
            line=dict(width=lw, color=color),
            hovertemplate=f"<b>{center_word}</b> → <b>{w}</b><br>距離: {distance:.2f}<br>エネルギー: {energy:.2f}<extra></extra>",
            showlegend=False
        ))

# --- 単語間エッジ ---
for i, j, e in network["edges"]:
    if i in center_indices or j in center_indices:
        continue
    x0, y0, z0 = pos[i]
    x1, y1, z1 = pos[j]
    distance = float(np.linalg.norm(pos[j] - pos[i]))

    strength = min(1.0, abs(e) / 2.0)
    lw = 0.5 + 2.0 * strength
    a = min(0.70, 0.20 + 0.40 * strength)

    if e < -1.0:
        color = f"rgba(120,180,255,{a})"
    elif e < -0.5:
        color = f"rgba(160,200,255,{a})"
    else:
        color = f"rgba(200,200,255,{a})"

    fig.add_trace(go.Scatter3d(
        x=[x0, x1], y=[y0, y1], z=[z0, z1],
        mode="lines",
        line=dict(width=lw, color=color),
        hovertemplate=f"<b>{words[i]}</b> ↔ <b>{words[j]}</b><br>距離: {distance:.2f}<br>エネルギー: {e:.2f}<extra></extra>",
        showlegend=False
    ))

# --- 球体（言葉）---
sizes, colors, labels = [], [], []
for w in words:
    energy = energies_dict.get(w, 0.0)
    if w in center_set:
        sizes.append(28)
        colors.append("rgba(255,235,100,0.98)")
        labels.append(w)
    else:
        en = min(1.0, abs(energy) / 3.0)
        sizes.append(12 + int(8 * en))
        if energy < -1.5:
            colors.append("rgba(180,220,255,0.85)")
        elif energy < -0.5:
            colors.append("rgba(220,240,255,0.75)")
        else:
            colors.append("rgba(255,255,255,0.60)")
        labels.append(w)

# --- 球体（言葉）---
sizes, colors, labels = [], [], []
for w in words:
    energy = energies_dict.get(w, 0.0)
    if w in center_set:
        sizes.append(28)
        colors.append("rgba(255,235,100,0.98)")
        labels.append(w)
    else:
        en = min(1.0, abs(energy) / 3.0)
        sizes.append(12 + int(8 * en))
        if energy < -1.5:
            colors.append("rgba(180,220,255,0.85)")
        elif energy < -0.5:
            colors.append("rgba(220,240,255,0.75)")
        else:
            colors.append("rgba(255,255,255,0.60)")
        labels.append(w)

# ★中心語とそれ以外で分ける（中心語ラベルを赤にする）
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
        textfont=dict(size=24, color="rgba(255,80,80,1.0)"),  # ★赤
        marker=dict(
            size=[sizes[i] for i in ci],
            color=[colors[i] for i in ci],
            line=dict(width=2, color="rgba(255,80,80,0.8)")
        ),
        hovertemplate="<b>%{text}</b><br>中心語<extra></extra>",
        showlegend=False
    ))


# 中心語の“光の層”（固定：点滅しない）
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
# 10) レイアウト（左：宇宙 / 右：格言+出所）
# ============================================================
left, right = st.columns([2.0, 1.0], gap="large")

with left:
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)
    st.caption("単語（球体）と縁（線）。マウスで回転・ズームできます。（点滅なし）")

with right:
    st.markdown("### 📊 現在の状態")
    st.markdown(f"**計算済み単語数**: {len(network.get('words', []))}語")
    st.markdown(f"**接続数**: {len(network.get('edges', []))}本")
    if network.get("energies"):
        mn = min(network["energies"].values())
        mx = max(network["energies"].values())
        st.markdown(f"**エネルギー範囲**: {mn:.2f} ～ {mx:.2f}")
    st.markdown("---")

    st.markdown(
        """
        <div style="
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.10);
          border-radius: 18px;
          padding: 16px 16px 10px 16px;
          box-shadow: 0 18px 60px rgba(0,0,0,0.18);
        ">
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 先人のことば")
    st.markdown(f"**いまの核（推定キーワード）**：`{', '.join(keywords)}`")
    st.markdown("---")

    q = select_relevant_quote(keywords)

    with st.expander("🔍 デバッグ情報（格言選択）", expanded=False):
        st.write(f"**抽出キーワード**: {keywords}")
        st.write(f"**利用可能な格言数**: {len(FAMOUS_QUOTES)}件")
        st.write(f"**Excel格言（読み込み）**: {len(load_quotes_from_excel_cached(EXCEL_DEFAULT))}件")
        st.write(f"**出所**: {q.get('source', '—')}")

    st.markdown(f"#### 「{q['quote']}」")
    st.markdown("---")
    st.markdown(f"**出所：** {q.get('source','—') if q.get('source') else '—'}")
    if q.get("note"):
        st.markdown(f"<div style='opacity:0.80; font-size:0.92rem;'>※ {q['note']}</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 抽出キーワード（確認）")
    for k in keywords:
        st.markdown(f"- {k}")

    st.markdown("</div>", unsafe_allow_html=True)
