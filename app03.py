# -*- coding: utf-8 -*-
""" 
Q-Quest 量子神託 - Streamlitアプリケーション（Streamlit Community Cloud向け / 正式版 v2）

- 直観アンケートで「12神」を最初に選択 → その神を固定してQUBOを解く
- おみくじ短文は毎回必ず表示（LLM成功時はLLM文、失敗時は自然なフォールバック文）
- Optunaで最適化の進捗/可視化（履歴等）を表示
- 固定神のもとで階層QUBO（感覚8bit×誓願12=3072）を全列挙し、エネルギー地形を安定表示
- キーワード球体（エネルギー近さ）可視化
- キャラクター画像：assets/images/characters/character_01.png ... character_12.png を表示
- Hugging Face APIキーは st.secrets / 環境変数から安全に読み込み（コードやUIに表示しない）
"""

from __future__ import annotations

import io
import os
import re
import time
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

# ========================================================================================
# 量子乱数関数（新規追加）
# ========================================================================================

def get_quantum_random_bytes(n_bytes: int = 32) -> bytes:
    try:
        response = requests.get(
            f"https://qrng.anu.edu.au/API/jsonI.php?length={n_bytes}&type=uint8",
            timeout=3
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return bytes(data['data'][:n_bytes])
    except Exception:
        pass
    return os.urandom(n_bytes)

def quantum_seed() -> int:
    qbytes = get_quantum_random_bytes(8)
    return int.from_bytes(qbytes, byteorder='big') % (2**32)

def quantum_float(low: float = 0.0, high: float = 1.0) -> float:
    qbytes = get_quantum_random_bytes(8)
    uint64 = int.from_bytes(qbytes, byteorder='big')
    norm = uint64 / (2**64 - 1)
    return low + (high - low) * norm

# -------------------------
# Optional deps
# -------------------------
try:
    from janome.tokenizer import Tokenizer
    JANOME_AVAILABLE = True
except Exception:
    JANOME_AVAILABLE = False
    Tokenizer = None

try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_contour,
        plot_slice,
        plot_timeline,
    )
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False
    optuna = None
    plot_optimization_history = None
    plot_param_importances = None
    plot_parallel_coordinate = None
    plot_contour = None
    plot_slice = None
    plot_timeline = None


# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(
    page_title="Q-Quest 量子神託",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------
# Randomness control (session)
# -------------------------

def _get_session_seed() -> int:
    if "seed" not in st.session_state:
        st.session_state.seed = int(time.time() * 1000) % 1_000_000
    return int(st.session_state.seed)


def _rng() -> np.random.Generator:
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng(_get_session_seed())
    return st.session_state.rng


# -------------------------
# Character images
# -------------------------

def get_character_image_path(god_id: int) -> Optional[str]:
    """assets/images/characters/character_01.png ... character_12.png"""
    fn = f"character_{god_id+1:02d}.png"
    path = os.path.join("assets", "images", "characters", fn)
    return path if os.path.exists(path) else None


# -------------------------
# String utilities (Excel)
# -------------------------

def _split_multi_text(cell_value: str) -> List[str]:
    if cell_value is None:
        return []
    s = str(cell_value).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    parts: List[str] = []
    for chunk in s.split("\n\n"):
        parts.extend([p.strip() for p in chunk.split("\n") if p.strip()])
    return [p for p in parts if p]


def _parse_tagged_quote(line: str) -> Dict[str, object]:
    raw = (line or "").strip()
    if "::" in raw:
        tag_part, quote_part = raw.split("::", 1)
        tags = [t.strip() for t in tag_part.split(",") if t.strip()]
        quote = quote_part.strip()
        return {"text": quote, "tags": tags}
    return {"text": raw, "tags": []}


# -------------------------
# Default data (12 gods)
# -------------------------
TWELVE_GODS = [
    {"id": 0, "name": "秋葉三尺坊", "name_en": "Akiba Sanjakubo", "attribute": "火", "emoji": "🔥",
     "vows": {"vow01": -0.4, "vow02": 0.2, "vow03": -0.2, "vow04": 0.0, "vow05": 0.0,
              "vow06": 0.0, "vow07": 0.0, "vow08": -0.4, "vow09": 0.0, "vow10": 0.0,
              "vow11": 0.0, "vow12": -0.2},
     "roles": {"stillness": 0.0, "flow": -0.2, "ma": 0.0, "sincerity": -0.4},
     "maxim": "勢いMAX: 情熱的な筆致に降臨。",
     "description": "秋葉原の守護神。火伏せ=「炎上回避」の神。"},
    {"id": 1, "name": "真空管大将軍", "name_en": "Vacuum Tube General", "attribute": "電", "emoji": "⚡",
     "vows": {"vow01": -0.2, "vow02": 0.2, "vow03": 0.0, "vow04": -0.4, "vow05": -0.2,
              "vow06": 0.0, "vow07": 0.0, "vow08": 0.0, "vow09": 0.0, "vow10": 0.0,
              "vow11": 0.0, "vow12": -0.4},
     "roles": {"stillness": 0.0, "flow": -0.4, "ma": 0.0, "sincerity": -0.2},
     "maxim": "線の太さ: 力強く、太い線に反応。",
     "description": "秋葉原の原点。増幅=「才能開花」の神。"},
    {"id": 2, "name": "LED弁財天", "name_en": "LED Benzaiten", "attribute": "光", "emoji": "💡",
     "vows": {"vow01": 0.0, "vow02": 0.2, "vow03": 0.0, "vow04": -0.4, "vow05": 0.0,
              "vow06": 0.0, "vow07": 0.0, "vow08": 0.0, "vow09": -0.4, "vow10": 0.0,
              "vow11": -0.2, "vow12": -0.2},
     "roles": {"stillness": 0.0, "flow": -0.4, "ma": -0.2, "sincerity": 0.0},
     "maxim": "丸み: 華やかで曲線的な筆跡。",
     "description": "イルミネーションと発光。「自己表現」の神。"},
    {"id": 3, "name": "磁気記録黒龍", "name_en": "Magnetic Recording Black Dragon", "attribute": "磁", "emoji": "🐉",
     "vows": {"vow01": 0.0, "vow02": 0.0, "vow03": -0.4, "vow04": 0.0, "vow05": -0.2,
              "vow06": 0.0, "vow07": 0.0, "vow08": 0.0, "vow09": 0.0, "vow10": -0.4,
              "vow11": -0.2, "vow12": 0.2},
     "roles": {"stillness": -0.2, "flow": 0.0, "ma": 0.0, "sincerity": -0.4},
     "maxim": "緻密さ: 細かく丁寧な書き込み。",
     "description": "HDDやテープ。記憶=「温故知新」の守護龍。"},
    {"id": 4, "name": "無線傍受観音", "name_en": "Wireless Interception Kannon", "attribute": "波", "emoji": "📡",
     "vows": {"vow01": -0.4, "vow02": 0.2, "vow03": 0.0, "vow04": -0.2, "vow05": -0.4,
              "vow06": 0.0, "vow07": 0.0, "vow08": 0.0, "vow09": 0.0, "vow10": 0.0,
              "vow11": 0.0, "vow12": -0.2},
     "roles": {"stillness": 0.0, "flow": -0.4, "ma": -0.2, "sincerity": 0.0},
     "maxim": "ゆらぎ: 震えや迷いがある筆跡に寄り添う。",
     "description": "電波と通信。縁結び=「マッチング」の神。"},
    {"id": 5, "name": "基板曼荼羅", "name_en": "Circuit Board Mandala", "attribute": "基", "emoji": "🔌",
     "vows": {"vow01": 0.0, "vow02": -0.2, "vow03": 0.0, "vow04": 0.0, "vow05": 0.0,
              "vow06": -0.4, "vow07": -0.4, "vow08": 0.0, "vow09": 0.2, "vow10": -0.2,
              "vow11": 0.0, "vow12": 0.0},
     "roles": {"stillness": -0.4, "flow": 0.0, "ma": 0.0, "sincerity": -0.2},
     "maxim": "直線的: 迷いのない、カクカクした線。",
     "description": "回路設計。秩序=「論理的思考」の神。"},
    {"id": 6, "name": "絶対零度明王", "name_en": "Absolute Zero Myo-o", "attribute": "冷", "emoji": "❄️",
     "vows": {"vow01": 0.0, "vow02": -0.4, "vow03": -0.2, "vow04": 0.0, "vow05": 0.0,
              "vow06": 0.0, "vow07": -0.4, "vow08": 0.0, "vow09": 0.0, "vow10": 0.0,
              "vow11": -0.2, "vow12": 0.2},
     "roles": {"stillness": -0.4, "flow": 0.0, "ma": -0.2, "sincerity": 0.0},
     "maxim": "筆圧弱め: クールで淡々とした筆跡。",
     "description": "冷却ファン・超電導。冷静=「沈着冷静」の神。"},
    {"id": 7, "name": "ジャンク再生童子", "name_en": "Junk Regeneration Child", "attribute": "壊", "emoji": "🔧",
     "vows": {"vow01": -0.2, "vow02": 0.0, "vow03": 0.0, "vow04": -0.2, "vow05": 0.0,
              "vow06": 0.0, "vow07": 0.2, "vow08": -0.4, "vow09": 0.0, "vow10": 0.0,
              "vow11": 0.0, "vow12": -0.4},
     "roles": {"stillness": 0.0, "flow": -0.4, "ma": 0.0, "sincerity": -0.2},
     "maxim": "かすれ: 荒々しい、または掠れた線。",
     "description": "秋葉原のジャンク品。復活=「再起・リトヲ」の神。"},
    {"id": 8, "name": "真空オーディオ如来", "name_en": "Vacuum Audio Nyorai", "attribute": "音", "emoji": "🎧",
     "vows": {"vow01": 0.0, "vow02": 0.0, "vow03": 0.0, "vow04": -0.2, "vow05": -0.4,
              "vow06": 0.0, "vow07": 0.2, "vow08": 0.0, "vow09": -0.2, "vow10": 0.0,
              "vow11": -0.4, "vow12": 0.0},
     "roles": {"stillness": 0.0, "flow": -0.4, "ma": -0.2, "sincerity": 0.0},
     "maxim": "調和: 文字全体のバランスが良い。",
     "description": "高音質・共鳴。「本質を見極める」神。"},
    {"id": 9, "name": "ハンダ付け結び神", "name_en": "Soldering Connection Deity", "attribute": "結", "emoji": "🔗",
     "vows": {"vow01": 0.0, "vow02": -0.4, "vow03": -0.2, "vow04": 0.0, "vow05": -0.4,
              "vow06": 0.0, "vow07": -0.2, "vow08": 0.0, "vow09": 0.0, "vow10": 0.0,
              "vow11": 0.0, "vow12": 0.2},
     "roles": {"stillness": -0.2, "flow": 0.0, "ma": -0.4, "sincerity": 0.0},
     "maxim": "トメ・ハネ: 繋ぎ部分がしっかりしている。",
     "description": "接点と結合。協力=「チームワーク」の神。"},
    {"id": 10, "name": "光速通信韋駄天", "name_en": "Light-speed Communication Idaten", "attribute": "速", "emoji": "🚀",
     "vows": {"vow01": 0.0, "vow02": 0.2, "vow03": 0.0, "vow04": -0.2, "vow05": -0.4,
              "vow06": 0.0, "vow07": 0.0, "vow08": 0.0, "vow09": -0.2, "vow10": 0.0,
              "vow11": 0.0, "vow12": -0.4},
     "roles": {"stillness": 0.0, "flow": -0.4, "ma": 0.0, "sincerity": -0.2},
     "maxim": "書き速度: サッと短時間で書いた線。",
     "description": "5G・光回線。爆速=「即断即決」の神。"},
    {"id": 11, "name": "半導体文殊", "name_en": "Semiconductor Manjushri", "attribute": "智", "emoji": "🧠",
     "vows": {"vow01": 0.0, "vow02": 0.0, "vow03": -0.2, "vow04": 0.0, "vow05": 0.0,
              "vow06": -0.4, "vow07": -0.2, "vow08": 0.0, "vow09": 0.0, "vow10": -0.4,
              "vow11": 0.0, "vow12": 0.2},
     "roles": {"stillness": -0.4, "flow": 0.0, "ma": 0.0, "sincerity": -0.2},
     "maxim": "規則性: 等間隔で整理された筆跡。",
     "description": "CPU・AI。計算=「合格・知略」の神。"},
]

SEASONS = ["薄氷", "立春", "春霞", "若葉", "夕立", "秋声", "木枯らし", "雪明り"]
MAXIM_SOURCES = {g["maxim"]: {"source": g["name"], "origin": g["name_en"], "reference": g["description"]} for g in TWELVE_GODS}

NEXT_STEPS_BY_MOOD = {
    "fatigue": ["一つだけ、今日やることを減らしなさい。", "遠回りを選びなさい。答えは道の途中にある。", "決めなくてよい。保留は、立派な選択である。"],
    "anxiety": ["話すなら『結論』より『気配』を渡しなさい。", "境界（しきい）を越えるのは、静かな一歩でよい。", "水のように流れるがままに。形にこだわらない。"],
    "curiosity": ["千里の道も一歩から。歩みを止めず、続けることに意味がある。", "成長は過程にあり。今この瞬間を大切に。", "挑戦する勇気こそが、未来を開く鍵である。"],
    "loneliness": ["一期一会。今この瞬間を大切に。すべては縁で繋がっている。", "人の心に寄り添う。それが真の強さである。", "絆は見えなくても、そこにある。"],
    "decisiveness": ["決めなくてよい。保留は、立派な選択である。", "己に誠実であること。それが自由への道である。", "道が分れていたら、念がない方へ行け。"],
    "default": ["一つだけ、今日やることを減らしなさい。", "遠回りを選びなさい。答えは道の途中にある。", "話すなら『結論』より『気配』を渡しなさい。", "決めなくてよい。保留は、立派な選択である。", "境界（しきい）を越えるのは、静かな一歩でよい。"],
}

GLOBAL_WORDS_DATABASE = [
    "世界平和", "貢献", "成長", "学び", "挑戦", "夢", "希望", "未来",
    "感謝", "愛", "幸せ", "喜び", "安心", "充実", "満足", "平和",
    "努力", "継続", "忍耐", "誠実", "正直", "優しさ", "思いやり", "共感",
    "調和", "バランス", "自然", "美", "真実", "自由", "正義", "道",
    "絆", "つながり", "家族", "友人", "仲間", "信頼", "尊敬", "協力", "夫婦", "生活", "円満",
    "今", "瞬間", "過程", "変化", "進化", "発展", "循環", "流れ",
    "静けさ", "集中", "覚悟", "決意", "勇気", "強さ", "柔軟性", "寛容",
]

FAMOUS_QUOTES = [
    {"keywords": ["平和", "世界", "貢献", "希望"], "quote": "雪の下で種は春を待っている。焦るべからず、時満ちるを待て。", "source": "日本の古語・ことわざ", "origin": "自然の摂理", "reference": "忍耐"},
    {"keywords": ["成長", "努力", "継続", "挑戦"], "quote": "千里の道も一歩から。歩みを止めず、続けることに意味がある。", "source": "老子『道徳経』", "origin": "第六十四章", "reference": "一歩"},
    {"keywords": ["感謝", "愛", "絆", "つながり"], "quote": "一期一会。今この瞬間を大切に。すべては縁で繋がっている。", "source": "茶道精神", "origin": "一期一会", "reference": "縁"},
]


# -------------------------
# Mood inference
# -------------------------

@dataclass
class Mood:
    fatigue: float
    anxiety: float
    curiosity: float
    loneliness: float
    decisiveness: float


KEYWORDS = {
    "fatigue": ["疲", "しんど", "眠", "だる", "消耗", "限界", "体調", "重", "動けない"],
    "anxiety": ["不安", "焦", "怖", "心配", "迷", "落ち着か", "緊張", "気になる", "自信", "持てない", "失敗", "間違い", "否定", "批判"],
    "curiosity": ["やってみ", "興味", "面白", "学び", "試", "挑戦", "ワクワク", "知りたい", "探索", "成長", "向上", "改善", "発展", "前進"],
    "loneliness": ["孤独", "一人", "寂", "誰にも", "分かって", "話せ", "孤立", "疎外"],
    "decisiveness": ["決め", "結論", "選", "判断", "断", "方針", "期限", "決断", "躊躇", "ためら", "優柔不断"],
}


def score_from_text(text: str, keys: List[str]) -> float:
    s = 0.0
    tl = text.lower()
    for k in keys:
        matches = len(re.findall(re.escape(k.lower()), tl))
        if matches > 0:
            base = matches * 0.5
            if len(k) >= 3:
                base += 0.5
            if len(k) >= 4:
                base += 0.3
            s += base
    return float(s)


def infer_mood(text: str) -> Mood:
    t = (text or "").strip()
    if not t:
        return Mood(0.0, 0.0, 0.0, 0.0, 0.0)

    raw = {k: score_from_text(t, v) for k, v in KEYWORDS.items()}
    max_raw = max(raw.values()) if max(raw.values()) > 0 else 1.0

    def norm(x: float, scale: float) -> float:
        if x == 0.0:
            return 0.0
        relative = x / max_raw if max_raw > 0 else 1.0
        absolute = min(1.0, x / scale)
        combined = (relative * 0.6 + absolute * 0.4)
        return float(max(0.15, min(1.0, combined)))

    return Mood(
        fatigue=norm(raw["fatigue"], 1.2),
        anxiety=norm(raw["anxiety"], 1.0),
        curiosity=norm(raw["curiosity"], 1.3),
        loneliness=norm(raw["loneliness"], 1.2),
        decisiveness=norm(raw["decisiveness"], 1.1),
    )


def mood_to_sensation_vector(m: Mood, binary: bool = False, scale: float = 5.0) -> np.ndarray:
    x = np.zeros(8)
    x[0] = m.anxiety * (1.0 - m.decisiveness)
    x[1] = m.anxiety
    x[2] = (m.fatigue + m.loneliness) / 2.0
    x[3] = (m.loneliness + m.fatigue) / 2.0
    x[4] = (m.curiosity + m.decisiveness) / 2.0
    x[5] = (1.0 - m.loneliness) * m.curiosity
    x[6] = m.curiosity * m.decisiveness
    x[7] = m.fatigue * (1.0 - m.decisiveness)

    x = x * scale
    if binary:
        return (x >= 0.3 * scale).astype(float)
    return x


# -------------------------
# HF API key secure loading
# -------------------------

def get_hf_api_key() -> str:
    """st.secrets から取得（Cloud推奨）。無ければ環境変数。"""
    # st.secrets は無いと例外になる場合があるので安全に
    try:
        if "HUGGINGFACE_API_KEY" in st.secrets:
            return str(st.secrets["HUGGINGFACE_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("HUGGINGFACE_API_KEY", "").strip()


# -------------------------
# Excel loading (optional)
# -------------------------
SENSE_TO_VOW_MATRIX: Optional[np.ndarray] = None
K_MATRIX: Optional[np.ndarray] = None
L_MATRIX: Optional[np.ndarray] = None
LOADED_GODS: Optional[List[Dict]] = None
MAXIMS_DATABASE: Optional[List[Dict]] = None


def rebuild_globals_from_gods(gods_list: List[Dict]) -> None:
    global MAXIM_SOURCES
    # sources: include maxims list
    sources: Dict[str, Dict] = {}
    for g in gods_list:
        if g.get("maxim"):
            sources[g["maxim"]] = {"source": g.get("name", "神託"), "origin": g.get("name_en", ""), "reference": g.get("description", "")}
        for item in g.get("maxims", []) or []:
            if isinstance(item, dict) and item.get("text"):
                t = item["text"].strip()
                if t:
                    sources[t] = {"source": g.get("name", "神託"), "origin": g.get("name_en", ""), "reference": g.get("description", "")}
    if sources:
        MAXIM_SOURCES = sources


def load_maxims_from_excel(maxim_file: io.BytesIO) -> List[Dict]:
    global MAXIMS_DATABASE
    maxim_file.seek(0)
    df = pd.read_excel(maxim_file, engine="openpyxl", header=0)
    out = []
    for _, row in df.iterrows():
        txt = str(row.get("格言", "")).strip()
        src = str(row.get("出典", "")).strip()
        if not txt or txt.lower() in ("nan", "none"):
            continue
        tags = []
        if "タグ" in df.columns:
            tag_str = str(row.get("タグ", "")).strip()
            if tag_str and tag_str.lower() not in ("nan", "none"):
                tags = [t.strip() for t in tag_str.split(",") if t.strip()]
        out.append({"text": txt, "source": src or "伝統的な教え", "tags": tags})
    MAXIMS_DATABASE = out
    return out


def load_sense_to_vow_matrix(sense_to_vow_file: io.BytesIO) -> np.ndarray:
    sense_to_vow_file.seek(0)
    df = pd.read_excel(sense_to_vow_file, engine="openpyxl", header=0, index_col=0).iloc[:8, :12]
    return df.values.astype(float)


def load_gods_from_separate_files(character_file: io.BytesIO, k_file: io.BytesIO, l_file: io.BytesIO) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    k_file.seek(0)
    df_k = pd.read_excel(k_file, engine="openpyxl", header=0, index_col=0).iloc[:12, :12]
    k_matrix = df_k.values.astype(float)

    l_file.seek(0)
    df_l = pd.read_excel(l_file, engine="openpyxl", header=0, index_col=0).iloc[:12, :4]
    l_matrix = df_l.values.astype(float)

    if character_file is not None:
        character_file.seek(0)
        df_g = pd.read_excel(character_file, engine="openpyxl")
    else:
        # fallback
        names = [n for n in df_k.index.tolist() if n in df_l.index.tolist()]
        df_g = pd.DataFrame({"ID": range(12), "名前": names, "名前(英語)": [f"God {i+1}" for i in range(12)],
                             "属性": [""] * 12, "絵文字": ["🔮"] * 12, "説明": [""] * 12, "格言": [""] * 12})

    gods_list: List[Dict] = []
    role_names = ["stillness", "flow", "ma", "sincerity"]
    for idx, row in df_g.iterrows():
        god_id = int(row.get("ID", idx))
        name = str(row.get("名前", "")).strip()
        name_en = str(row.get("名前(英語)", "")).strip()
        attr = str(row.get("属性", "")).strip()
        emoji = str(row.get("絵文字", "🔮")).strip()
        desc = str(row.get("説明", "")).strip()

        maxim_cells: List[str] = []
        maxim_cells.extend(_split_multi_text(row.get("格言", "")))
        for col in row.index:
            if isinstance(col, str) and col.startswith("格言") and col != "格言":
                maxim_cells.extend(_split_multi_text(row.get(col, "")))
        maxims_parsed = [_parse_tagged_quote(m) for m in maxim_cells if str(m).strip()]
        maxim = maxims_parsed[0]["text"] if maxims_parsed else str(row.get("格言", "")).strip()

        vows = {}
        if name in df_k.index:
            ridx = df_k.index.get_loc(name)
            for j in range(12):
                vows[f"vow{j+1:02d}"] = float(k_matrix[ridx, j])
        else:
            for j in range(12):
                vows[f"vow{j+1:02d}"] = float(k_matrix[god_id, j])

        roles = {}
        if name in df_l.index:
            ridx = df_l.index.get_loc(name)
            for j, rn in enumerate(role_names):
                roles[rn] = float(l_matrix[ridx, j])
        else:
            for j, rn in enumerate(role_names):
                roles[rn] = float(l_matrix[god_id, j])

        gods_list.append({
            "id": god_id,
            "name": name,
            "name_en": name_en,
            "attribute": attr,
            "emoji": emoji,
            "vows": vows,
            "roles": roles,
            "maxim": maxim,
            "maxims": maxims_parsed,
            "description": desc,
        })

    return gods_list, k_matrix, l_matrix


def load_excel_config(character_file, maxim_file, sense_to_vow_file, k_file, l_file) -> bool:
    global SENSE_TO_VOW_MATRIX, K_MATRIX, L_MATRIX, LOADED_GODS, TWELVE_GODS
    try:
        if k_file is None or l_file is None:
            st.sidebar.error("k行列とl行列は必須です")
            return False
        gods_list, k_matrix, l_matrix = load_gods_from_separate_files(character_file, k_file, l_file)
        if sense_to_vow_file is not None:
            SENSE_TO_VOW_MATRIX = load_sense_to_vow_matrix(sense_to_vow_file)
        else:
            SENSE_TO_VOW_MATRIX = None
        K_MATRIX, L_MATRIX = k_matrix, l_matrix
        LOADED_GODS = gods_list
        TWELVE_GODS = gods_list
        rebuild_globals_from_gods(gods_list)
        if maxim_file is not None:
            load_maxims_from_excel(maxim_file)
        return True
    except Exception as e:
        st.sidebar.error(f"設定の読み込みに失敗: {e}")
        return False


# -------------------------
# QUBO
# -------------------------

def qubo_energy(x: np.ndarray, Q: Dict[Tuple[int, int], float]) -> float:
    e = 0.0
    n = len(x)
    for i in range(n):
        e += Q.get((i, i), 0.0) * x[i]
    for i in range(n):
        for j in range(i + 1, n):
            e += Q.get((i, j), 0.0) * x[i] * x[j]
    return float(e)


def build_qubo_with_quantum_fluctuation(
    x_bin: np.ndarray,
    mood: Mood,
    K_MATRIX: np.ndarray,
    L_MATRIX: np.ndarray,
    SENSE_TO_VOW_MATRIX: Optional[np.ndarray] = None,
    lambda_v: float = 5.0,
    lambda_c: float = 5.0,
    quantum_noise_level: float = 0.6,
) -> Tuple[Dict[Tuple[int, int], float], Dict]:
    Q: Dict[Tuple[int, int], float] = {}
    n_sense, n_vows, n_chars = 8, 12, 12
    v_start, c_start = n_sense, n_sense + n_vows
    
    metadata = {
        "quantum_seed": quantum_seed(),
        "noise_injections": [],
        "energy_shifts": {}
    }
    
    # 量子的基底バイアス
    for i in range(32):
        quantum_bias = quantum_float(-0.3, 0.3) * quantum_noise_level
        Q[(i, i)] = Q.get((i, i), 0.0) + quantum_bias
        metadata["noise_injections"].append({"bit": i, "bias": quantum_bias})
    
    # 感覚ビット
    x_cont = mood_to_sensation_vector(mood, binary=False, scale=5.0)
    for i in range(n_sense):
        if x_bin[i] > 0:
            strength = float(np.clip(x_cont[i] / 5.0, 0.0, 1.0))
            bias = -1.0 * strength + quantum_float(-0.2, 0.2) * quantum_noise_level
            Q[(i, i)] = Q.get((i, i), 0.0) + bias
    
    # 誓願 one-hot
    for j in range(n_vows):
        vj = v_start + j
        Q[(vj, vj)] = Q.get((vj, vj), 0.0) - 2.0 * lambda_v
        for k in range(j + 1, n_vows):
            vk = v_start + k
            Q[(vj, vk)] = Q.get((vj, vk), 0.0) + 2.0 * lambda_v
    
    # 神 one-hot（★固定ペナルティなし★）
    for k in range(n_chars):
        ck = c_start + k
        Q[(ck, ck)] = Q.get((ck, ck), 0.0) - 2.0 * lambda_c
        for l in range(k + 1, n_chars):
            cl = c_start + l
            Q[(ck, cl)] = Q.get((ck, cl), 0.0) + 2.0 * lambda_c
    
    # 感覚-誓願
    if SENSE_TO_VOW_MATRIX is not None:
        for i in range(n_sense):
            if x_bin[i] > 0:
                strength = float(np.clip(x_cont[i] / 5.0, 0.0, 1.0))
                for j in range(n_vows):
                    coupling = float(SENSE_TO_VOW_MATRIX[i, j]) * strength
                    coupling *= (1.0 + quantum_float(-0.2, 0.2) * quantum_noise_level)
                    Q[(i, v_start + j)] = Q.get((i, v_start + j), 0.0) + coupling
    
    # 誓願-神（★量子揺らぎ付き★）
    for j in range(n_vows):
        for k in range(n_chars):
            coupling = float(K_MATRIX[k, j])
            coupling *= (1.0 + quantum_float(-0.15, 0.15) * quantum_noise_level)
            Q[(v_start + j, c_start + k)] = Q.get((v_start + j, c_start + k), 0.0) + coupling
            metadata["energy_shifts"][f"vow{j}_god{k}"] = coupling - K_MATRIX[k, j]
    
    # 感覚-神
    role_mapping = {0: 0, 1: 1, 2: 0, 3: 2, 4: 1, 5: 2, 6: 1, 7: 3}
    for i in range(n_sense):
        if x_bin[i] > 0:
            rc = role_mapping.get(i, 0)
            for k in range(n_chars):
                coupling = float(L_MATRIX[k, rc])
                coupling *= (1.0 + quantum_float(-0.1, 0.1) * quantum_noise_level)
                Q[(i, c_start + k)] = Q.get((i, c_start + k), 0.0) + coupling
    
    # 矛盾
    if x_bin[0] > 0 and x_bin[4] > 0:
        penalty = 3.0 * (1.0 + quantum_float(-0.3, 0.3) * quantum_noise_level)
        Q[(0, 4)] = Q.get((0, 4), 0.0) + penalty
    if x_bin[1] > 0 and x_bin[7] > 0:
        penalty = 3.0 * (1.0 + quantum_float(-0.3, 0.3) * quantum_noise_level)
        Q[(1, 7)] = Q.get((1, 7), 0.0) + penalty
    
    return Q, metadata

def solve_exact_fixed_char(Q: Dict[Tuple[int, int], float], fixed_god_id: int) -> List[Tuple[float, np.ndarray]]:
    sols: List[Tuple[float, np.ndarray]] = []
    n = 32
    v_start, c_start = 8, 20
    for bits in range(256):
        sense = np.array([(bits >> i) & 1 for i in range(8)], dtype=int)
        for vow_idx in range(12):
            x = np.zeros(n, dtype=int)
            x[:8] = sense
            x[v_start + vow_idx] = 1
            x[c_start + fixed_god_id] = 1
            sols.append((qubo_energy(x, Q), x))
    sols.sort(key=lambda t: t[0])
    return sols

def calculate_quantum_temperature(mood: Mood) -> float:
    """気分から最適な温度（揺らぎの強さ）を決定"""
    base_temp = 0.5
    temp = base_temp + 0.3 * mood.curiosity - 0.2 * mood.fatigue + 0.1 * mood.anxiety
    temp += quantum_float(-0.1, 0.1)
    return float(np.clip(temp, 0.2, 1.0))


def boltzmann_sample(
    solutions: List[Tuple[float, np.ndarray]],
    temperature: float = 0.5,
    use_quantum_random: bool = True
) -> Tuple[float, np.ndarray]:
    """Boltzmann分布に従って解を確率的に選択"""
    if not solutions:
        raise ValueError("解のリストが空です")
    if len(solutions) == 1:
        return solutions[0]
    
    energies = np.array([e for e, _ in solutions])
    E_min = energies.min()
    E_shifted = energies - E_min
    
    if temperature <= 0:
        temperature = 1e-6
    
    weights = np.exp(-E_shifted / temperature)
    probs = weights / weights.sum()
    
    if use_quantum_random:
        r = quantum_float(0.0, 1.0)
    else:
        r = np.random.random()
    
    cumsum = np.cumsum(probs)
    idx = np.searchsorted(cumsum, r)
    idx = min(idx, len(solutions) - 1)
    
    return solutions[idx]


def solve_exact_all_gods(Q: Dict[Tuple[int, int], float]) -> List[Tuple[float, np.ndarray]]:
    """全ての神×全ての誓願を全列挙（固定なし）"""
    sols: List[Tuple[float, np.ndarray]] = []
    n = 32
    v_start, c_start = 8, 20
    
    # 全12神を対象に
    for god_id in range(12):
        for bits in range(256):  # 感覚8bit = 256通り
            sense = np.array([(bits >> i) & 1 for i in range(8)], dtype=int)
            for vow_idx in range(12):  # 誓願12通り
                x = np.zeros(n, dtype=int)
                x[:8] = sense
                x[v_start + vow_idx] = 1
                x[c_start + god_id] = 1
                sols.append((qubo_energy(x, Q), x))
    
    sols.sort(key=lambda t: t[0])
    return sols

def solve_optuna_fixed_char(Q: Dict[Tuple[int, int], float], fixed_god_id: int, n_trials: int, progress_container=None):
    if not OPTUNA_AVAILABLE:
        return None
    sampler = optuna.samplers.TPESampler(seed=_get_session_seed())
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial):
        vow_idx = trial.suggest_int("vow_idx", 0, 11)
        x = np.zeros(32, dtype=int)
        x[8 + vow_idx] = 1
        x[20 + fixed_god_id] = 1
        for i in range(8):
            x[i] = trial.suggest_int(f"sense_{i}", 0, 1)
        return qubo_energy(x, Q)

    if progress_container is not None:
        with progress_container:
            bar = st.progress(0)
        for i in range(n_trials):
            study.optimize(objective, n_trials=1, show_progress_bar=False)
            with progress_container:
                bar.progress(int(((i + 1) / n_trials) * 100))
    else:
        study.optimize(objective, n_trials=n_trials)

    return study


# -------------------------
# Omikuji core
# -------------------------

def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    if not text or not text.strip():
        return []
    t = text.strip()
    found: List[str] = []

    tl = t.lower()
    for kws in KEYWORDS.values():
        for kw in kws:
            if kw.lower() in tl and kw not in found:
                found.append(kw)

    tmp = t
    for w in sorted(GLOBAL_WORDS_DATABASE, key=len, reverse=True):
        if w in tmp and w not in found:
            found.append(w)
            tmp = tmp.replace(w, " ")

    chunks = re.findall(r"[ぁ-んァ-ヶ一-龠]{2,8}", tmp)
    stop = {"こと", "もの", "ため", "それ", "これ", "よう", "です", "ます"}
    for c in chunks:
        if c not in stop and c not in found:
            found.append(c)

    if JANOME_AVAILABLE:
        try:
            tok = Tokenizer()
            for token in tok.tokenize(t):
                pos = token.part_of_speech.split(",")[0]
                if pos in ["名詞", "動詞", "形容詞"]:
                    s = token.surface
                    if 2 <= len(s) <= 8 and s not in stop and s not in found:
                        found.append(s)
        except Exception:
            pass

    return list(dict.fromkeys(found))[:top_n]


def get_maxim_source(maxim: str) -> Dict:
    if maxim in MAXIM_SOURCES:
        return MAXIM_SOURCES[maxim]
    for q in FAMOUS_QUOTES:
        if q.get("quote") == maxim:
            return {"source": q.get("source", "引用"), "origin": q.get("origin", ""), "reference": q.get("reference", "")}
    return {"source": "伝統的な教え", "origin": "古来より伝わる智慧", "reference": ""}


def select_relevant_quote(keywords: List[str]) -> str:
    ks = set(keywords)
    scored = []
    for q in FAMOUS_QUOTES:
        score = len(ks & set(q["keywords"])) + float(_rng().uniform(-0.2, 0.2))
        scored.append((score, q["quote"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else "あなたの観測が、この世界線を確定させました。"


def select_maxims_from_database(keywords: List[str], top_k: int = 3, exclude: Optional[List[str]] = None) -> List[Dict]:
    if not MAXIMS_DATABASE:
        return []
    exclude_set = set(exclude or [])
    keyset = set([k.lower() for k in keywords])

    scored = []
    for m in MAXIMS_DATABASE:
        txt = m.get("text", "")
        if not txt or txt in exclude_set:
            continue
        low = txt.lower()
        tags = [t.lower() for t in (m.get("tags") or [])]
        score = 0.0
        for kw in keyset:
            if kw in low:
                score += 5.0
            if any(kw in tg for tg in tags):
                score += 3.0
        if score > 0:
            scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]


def select_picks_for_god(god: Dict, context_text: str, top_k: int = 3) -> List[str]:
    exclude = st.session_state.get("recent_maxims", [])[-10:]
    picks: List[str] = []

    god_maxims = []
    if god.get("maxims"):
        for it in god["maxims"]:
            if isinstance(it, dict) and it.get("text"):
                god_maxims.append(it["text"].strip())
    if not god_maxims and god.get("maxim"):
        god_maxims = [god["maxim"].strip()]

    for gm in god_maxims:
        if gm and gm not in picks and gm not in exclude:
            picks.append(gm)
        if len(picks) >= top_k:
            break

    kws = extract_keywords(context_text, top_n=8) if context_text else []
    if MAXIMS_DATABASE and kws and len(picks) < top_k:
        dbs = select_maxims_from_database(kws, top_k=top_k, exclude=exclude + picks)
        for m in dbs:
            t = m.get("text", "")
            if t and t not in picks:
                picks.append(t)
            if len(picks) >= top_k:
                break

    if not picks:
        picks = [select_relevant_quote(kws or ["今"])]

    st.session_state.setdefault("recent_maxims", [])
    for p in picks:
        if p and p not in st.session_state.recent_maxims:
            st.session_state.recent_maxims.append(p)
    st.session_state.recent_maxims = st.session_state.recent_maxims[-20:]

    return picks[:top_k]


def compose_poem_and_hint(picks: List[str], mood: Mood) -> Tuple[str, str]:
    season = random.choice(SEASONS)
    head = picks[0] if picks else "今この瞬間を大切に。"
    if len(head) > 30:
        head = head[:30] + "..."
    poem = f"{season}／{head}"

    mood_scores = {
        "fatigue": mood.fatigue,
        "anxiety": mood.anxiety,
        "curiosity": mood.curiosity,
        "loneliness": mood.loneliness,
        "decisiveness": mood.decisiveness,
    }
    k, v = max(mood_scores.items(), key=lambda x: x[1])
    hints = NEXT_STEPS_BY_MOOD.get(k, NEXT_STEPS_BY_MOOD["default"]) if v > 0.3 else NEXT_STEPS_BY_MOOD["default"]
    return poem, random.choice(hints)


# -------------------------
# LLM + fallback (always show)
# -------------------------

@dataclass
class LLMResult:
    text: str
    ok: bool
    reason: str = ""
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


def _pick_rate_headers(headers: Dict[str, str]) -> Dict[str, str]:
    keep = {}
    for k, v in headers.items():
        kl = k.lower()
        if "ratelimit" in kl or kl in ("retry-after", "x-ratelimit-remaining", "x-ratelimit-limit"):
            keep[k] = v
    return keep


def build_omikuji_prompt(user_text: str, god: Dict, picks: List[str], mood: Mood) -> str:
    maxims = "\n".join([f"- {p}" for p in picks[:2]])
    return f"""あなたは『{god.get('name','神')}』として日本語で話します。

ユーザーの願い/悩み：
「{user_text}」

参考格言：
{maxims}

気分：疲れ={mood.fatigue:.2f}, 不安={mood.anxiety:.2f}, 好奇心={mood.curiosity:.2f}, 孤独={mood.loneliness:.2f}, 決断={mood.decisiveness:.2f}

出力条件：
- おみくじ風
- 50〜100文字
- やさしく、最後は前向きに

神託：
"""


def hf_generate(prompt: str, model: str, api_key: str,
               max_new_tokens: int = 120, temperature: float = 0.7, top_p: float = 0.9,
               timeout: int = 30, retries: int = 2, backoff: float = 1.5) -> LLMResult:
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "return_full_text": False,
        }
    }

    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            hdrs = _pick_rate_headers(dict(r.headers))
            if r.status_code == 200:
                data = r.json()
                gen = ""
                if isinstance(data, list) and data:
                    gen = data[0].get("generated_text") or data[0].get("text") or ""
                elif isinstance(data, dict):
                    gen = data.get("generated_text") or data.get("text") or ""
                gen = (gen or "").strip()
                if gen:
                    return LLMResult(text=gen, ok=True, reason="ok", status_code=200, headers=hdrs)
                return LLMResult(text="", ok=False, reason="empty", status_code=200, headers=hdrs)

            if r.status_code in (429, 502, 503, 504):
                if attempt < retries:
                    time.sleep(backoff ** attempt)
                    continue
                return LLMResult(text="", ok=False, reason="busy_or_rate_limited", status_code=r.status_code, headers=hdrs)

            if r.status_code in (401, 403):
                return LLMResult(text="", ok=False, reason="auth_error", status_code=r.status_code, headers=hdrs)

            return LLMResult(text="", ok=False, reason="http_error", status_code=r.status_code, headers=hdrs)

        except Exception as e:
            if attempt < retries:
                time.sleep(backoff ** attempt)
                continue
            return LLMResult(text="", ok=False, reason=f"exception:{type(e).__name__}")

    return LLMResult(text="", ok=False, reason="unknown")


def fallback_short_oracle(user_text: str, god: Dict, picks: List[str], mood: Mood) -> str:
    kws = extract_keywords(user_text, top_n=6)
    core = picks[0] if picks else (god.get("maxim") or "今を大切に")

    mood_scores = {"疲れ": mood.fatigue, "不安": mood.anxiety, "好奇心": mood.curiosity, "孤独": mood.loneliness, "決断": mood.decisiveness}
    main_mood = max(mood_scores.items(), key=lambda x: x[1])[0]

    templates = [
        "{main}の気配が強い。{kw}を一つだけ守り、{end}。",
        "{kw}に目を向けよ。{core}。{end}",
        "焦らず、{kw}を整えよ。{core}。{end}",
        "{main}の時は小さく。{kw}から始めよ。{end}",
    ]

    kw = random.choice(kws) if kws else random.choice(GLOBAL_WORDS_DATABASE)
    ending = random.choice(["今日の一歩は必ず実る。", "大丈夫、道は続いている。", "縁は静かに結ばれる。", "あなたの観測が、世界線を整える。"])

    text = random.choice(templates).format(main=main_mood, kw=kw, core=core, end=ending)
    if len(text) > 110:
        text = text[:108] + "…"
    return text


def explain_llm_issue(meta: Dict) -> str:
    provider = meta.get("provider")
    status = meta.get("status")
    reason = meta.get("reason", "")
    if provider == "huggingface":
        if status == 429:
            return "推定原因：レート制限（無料枠/混雑/短時間の集中アクセス）。"
        if status in (502, 503, 504):
            return "推定原因：Hugging Face側の混雑/一時障害。"
        if status in (401, 403):
            return "推定原因：APIキー未設定/権限不足。"
        if reason == "empty":
            return "推定原因：200だが生成文が空（モデル/応答形式の相性）。"
        return f"推定原因：HTTP {status} / {reason}"
    return f"推定原因：{reason}"


def generate_short_oracle_always(user_text: str, god: Dict, picks: List[str], mood: Mood,
                                llm_enabled: bool, hf_model: str,
                                hf_max_new_tokens: int, hf_temperature: float, hf_top_p: float) -> Tuple[str, Dict]:
    meta = {
        "provider": "huggingface",
        "llm_ok": False,
        "reason": "",
        "status": None,
        "headers": {},
        "fallback": False,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": hf_model,
    }

    api_key = get_hf_api_key()
    if llm_enabled and user_text.strip():
        prompt = build_omikuji_prompt(user_text, god, picks, mood)
        res = hf_generate(prompt, model=hf_model, api_key=api_key,
                          max_new_tokens=hf_max_new_tokens,
                          temperature=hf_temperature,
                          top_p=hf_top_p)
        meta.update({"llm_ok": bool(res.ok and (res.text or "").strip()), "reason": res.reason, "status": res.status_code, "headers": res.headers or {}})
        if meta["llm_ok"]:
            return res.text.strip(), meta

    meta["fallback"] = True
    return fallback_short_oracle(user_text, god, picks, mood), meta


# -------------------------
# Word sphere
# -------------------------

def calculate_energy_between_words(word1: str, word2: str) -> float:
    energy = 0.0
    common = set(word1) & set(word2)
    if common:
        energy -= len(common) * 0.25
    energy += float(_rng().normal(0, 0.12))
    return energy


def build_word_network(center_words: List[str], database: List[str], n_neighbors: int = 20) -> Dict:
    all_words = list(set(center_words + database))
    energies: Dict[str, float] = {}

    for w in all_words:
        if w in center_words:
            e = -2.0
        else:
            es = [calculate_energy_between_words(cw, w) for cw in center_words]
            e = float(np.mean(es) + _rng().normal(0, 0.08))
        energies[w] = e

    sorted_words = sorted(energies.items(), key=lambda x: (x[1], float(_rng().random())))
    selected = center_words.copy()
    for w, _e in sorted_words:
        if w not in selected and len(selected) < n_neighbors:
            selected.append(w)

    edges = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            e = calculate_energy_between_words(selected[i], selected[j])
            if e < -0.25:
                edges.append((i, j, e))

    return {"words": selected, "energies": {w: energies[w] for w in selected}, "edges": edges}


def place_words_on_sphere(n_words: int, center_indices: List[int]) -> np.ndarray:
    pos = np.zeros((n_words, 3))
    golden = np.pi * (3 - np.sqrt(5))
    for i in range(n_words):
        r = 0.35 + float(_rng().random()) * 0.15 if i in center_indices else 0.85 + float(_rng().random()) * 0.35
        theta = golden * i
        y = 1 - (i / float(max(1, n_words - 1))) * 2
        rad = math.sqrt(max(1e-9, 1 - y * y))
        x = math.cos(theta) * rad * r
        z = math.sin(theta) * rad * r
        pos[i] = [x, y, z]
    return pos


def create_3d_network_plot(network: Dict, positions: np.ndarray, center_indices: List[int]) -> go.Figure:
    fig = go.Figure()

    for i, j, e in network["edges"]:
        fig.add_trace(go.Scatter3d(
            x=[positions[i, 0], positions[j, 0]],
            y=[positions[i, 1], positions[j, 1]],
            z=[positions[i, 2], positions[j, 2]],
            mode="lines",
            line=dict(color="#4a9eff" if e < -0.5 else "#ff6b6b", width=0.6 + abs(e) * 1.6),
            showlegend=False,
            hoverinfo="skip",
        ))

    for i, w in enumerate(network["words"]):
        is_center = i in center_indices
        fig.add_trace(go.Scatter3d(
            x=[positions[i, 0]],
            y=[positions[i, 1]],
            z=[positions[i, 2]],
            mode="markers+text",
            marker=dict(
                size=14 if is_center else 8,
                color="#ffd700" if is_center else "#ffffff",
                line=dict(width=2, color="white"),
                opacity=0.9 if is_center else 0.65,
            ),
            text=[w],
            textposition="middle center",
            textfont=dict(size=20 if is_center else 16, color="#ffd700" if is_center else "#ffffff"),
            hovertemplate=f"<b>{w}</b><br>エネルギー: {network['energies'].get(w,0):.2f}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text="言葉のエネルギー球体（Quantum Word Sphere）", x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, showticklabels=False, title=""),
            bgcolor="#0a0a1a",
        ),
        plot_bgcolor="#0a0a1a",
        paper_bgcolor="#0a0a1a",
        margin=dict(l=0, r=0, t=50, b=0),
        height=650,
    )

    return fig


# -------------------------
# Sidebar diagnostic panel
# -------------------------

def sidebar_diagnostic_panel():
    with st.sidebar.expander("🛠 診断情報（開発者向け）", expanded=False):
        meta = st.session_state.get("last_llm_meta")
        if not meta:
            st.caption("まだ診断情報はありません")
            return
        st.write(explain_llm_issue(meta))
        st.json(meta)
        st.download_button(
            "診断情報をJSONで保存",
            data=json.dumps(meta, ensure_ascii=False, indent=2),
            file_name="llm_diagnostic.json",
            mime="application/json",
        )


# -------------------------
# Main UI
# -------------------------

def main():
    st.title("🔮 Q-Quest 量子神託（画像表示＋神固定＋診断）")
    st.caption("12神アンケートで世界線を固定し、QUBO最適化と神託を表示します")
    st.markdown("---")

    gods = LOADED_GODS if LOADED_GODS else TWELVE_GODS
    options = [f"{g.get('emoji','🔮')} {g.get('name','')}" for g in gods]

    # Sidebar: survey
    st.sidebar.header("🗳️ 直観アンケート（必須）")
    idx0 = int(st.session_state.get("selected_god_index", 0))
    idx0 = min(max(idx0, 0), len(options) - 1)
    selected_label = st.sidebar.radio("まず神を選んでください", options, index=idx0)
    fixed_god_id = options.index(selected_label)
    st.session_state.selected_god_index = fixed_god_id

    # Show image in sidebar
    img = get_character_image_path(fixed_god_id)
    if img:
        st.sidebar.image(img, caption=options[fixed_god_id], use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.header("📊 設定ファイル（任意）")
    with st.sidebar.expander("Excel設定を読み込む（k/l必須）", expanded=False):
        character_file = st.file_uploader("1) 12神基本情報", type=["xlsx", "xls"], key="char")
        maxim_file = st.file_uploader("2) 格言DB（任意）", type=["xlsx", "xls"], key="maxim")
        sense_to_vow_file = st.file_uploader("3) sense_to_vow（任意）", type=["xlsx", "xls"], key="sv")
        k_file = st.file_uploader("4) k行列（必須）", type=["xlsx", "xls"], key="k")
        l_file = st.file_uploader("5) l行列（必須）", type=["xlsx", "xls"], key="l")
        if st.button("読み込み", use_container_width=True):
            ok = load_excel_config(character_file, maxim_file, sense_to_vow_file, k_file, l_file)
            st.sidebar.success("✅ 読み込み完了" if ok else "❌ 読み込み失敗")
            st.rerun()

    # Sidebar: LLM settings (no key shown)
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 短文生成（毎回必ず表示）")
    llm_enabled = st.sidebar.checkbox("Hugging Faceで生成を試す", value=True)
    hf_model = st.sidebar.text_input("モデル名", value="microsoft/DialoGPT-medium")
    hf_max_new_tokens = st.sidebar.slider("最大生成トークン", 60, 180, 120, 10)
    hf_temperature = st.sidebar.slider("temperature", 0.1, 1.2, 0.7, 0.1)
    hf_top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.9, 0.05)
    st.sidebar.caption("APIキーは st.secrets / 環境変数から読み込みます（UIには表示しません）。")

    # Sidebar: optuna
    st.sidebar.markdown("---")
    st.sidebar.header("📈 Optuna（進捗可視化）")
    run_optuna = st.sidebar.checkbox("Optunaを実行", value=True, disabled=not OPTUNA_AVAILABLE)
    n_trials = st.sidebar.slider("試行回数", 30, 200, 80, 10)
    if not OPTUNA_AVAILABLE:
        st.sidebar.info("optuna未導入：requirements.txtに optuna を追加してください")

    # Sidebar: mode
    st.sidebar.markdown("---")
    mode = st.sidebar.selectbox("モード", ["対話型量子神託", "言葉の球体可視化"])

    sidebar_diagnostic_panel()

    if mode == "言葉の球体可視化":
        st.header("🪐 言葉のエネルギー球体")
        user_input = st.text_input("願いを入力", value="世界平和に貢献できる人間になる")
        if st.button("可視化", use_container_width=True):
            kws = extract_keywords(user_input, top_n=8)
            if not kws:
                st.warning("キーワードが抽出できませんでした")
                return
            net = build_word_network(kws, GLOBAL_WORDS_DATABASE, n_neighbors=20)
            centers = [i for i, w in enumerate(net["words"]) if w in kws]
            pos = place_words_on_sphere(len(net["words"]), centers)
            fig = create_3d_network_plot(net, pos, centers)
            st.plotly_chart(fig, use_container_width=True)
        return

# Main: oracle
    st.header("🔮 量子神託")
    st.caption(f"固定された神：{options[fixed_god_id]}")

    god = (LOADED_GODS if LOADED_GODS else TWELVE_GODS)[fixed_god_id]
    img_main = get_character_image_path(fixed_god_id)
    if img_main:
        st.image(img_main, width=320)

    user_text = st.text_area("今日の願い・気持ちを一文で", placeholder="例：疲れていて決断ができない…", height=120)

    if st.button("神託を求める", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("テキストを入力してください")
            return

        mood = infer_mood(user_text)
        st.session_state["last_mood"] = mood
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("疲れ", f"{mood.fatigue:.2f}")
        c2.metric("不安", f"{mood.anxiety:.2f}")
        c3.metric("好奇心", f"{mood.curiosity:.2f}")
        c4.metric("孤独", f"{mood.loneliness:.2f}")
        c5.metric("決断", f"{mood.decisiveness:.2f}")

        # 感覚ベクトル生成
        x_cont = mood_to_sensation_vector(mood, binary=False, scale=5.0)
        x_bin = (x_cont >= 1.5).astype(float)

        # K/L行列準備
        gods_list = LOADED_GODS if LOADED_GODS else TWELVE_GODS
        k_matrix = K_MATRIX
        l_matrix = L_MATRIX

        if k_matrix is None:
            k_matrix = np.zeros((12, 12))
            for k, god_temp in enumerate(gods_list):
                for j in range(12):
                    k_matrix[k, j] = float(god_temp["vows"][f"vow{j+1:02d}"])

        if l_matrix is None:
            l_matrix = np.zeros((12, 4))
            roles = ["stillness", "flow", "ma", "sincerity"]
            for k, god_temp in enumerate(gods_list):
                for j, rn in enumerate(roles):
                    l_matrix[k, j] = float(god_temp["roles"][rn])

        # ★QUBO構築（量子揺らぎ付き）
        Q, metadata = build_qubo_with_quantum_fluctuation(
            x_bin=x_bin,
            mood=mood,
            K_MATRIX=k_matrix,
            L_MATRIX=l_matrix,
            SENSE_TO_VOW_MATRIX=SENSE_TO_VOW_MATRIX,
            quantum_noise_level=0.6
        )

        # ★★★ 量子的揺らぎの可視化 ★★★
        with st.expander("🌀 量子的揺らぎの詳細（この瞬間だけのユニークな値）", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("量子シード", f"{metadata['quantum_seed']}", 
                         help="この瞬間の宇宙の状態を表す数値（真正乱数）")
                st.metric("ノイズ注入箇所", f"{len(metadata['noise_injections'])}ビット",
                         help="QUBOに注入された量子的揺らぎの数")
            
            with col2:
                st.metric("エネルギー変動箇所", f"{len(metadata['energy_shifts'])}",
                         help="神と誓願の結びつきが量子的に変化した箇所")
                
                shifts = list(metadata['energy_shifts'].values())
                if shifts:
                    avg_shift = np.mean(np.abs(shifts))
                    st.metric("平均揺らぎ強度", f"{avg_shift:.4f}",
                             help="エネルギーの変動幅（大きいほど意外な結果も）")
            
            if shifts:
                st.markdown("#### エネルギー揺らぎの分布")
                fig_shift = px.histogram(
                    shifts, 
                    nbins=30, 
                    title="量子的エネルギーシフト（この瞬間のユニークな分布）",
                    labels={'value': 'エネルギー変化量', 'count': '頻度'}
                )
                fig_shift.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_shift, use_container_width=True)
                
                st.caption("💡 この分布は毎回異なります。同じ願いでも、量子的な揺らぎにより結果が変わります。")

        # Optuna progress
        if run_optuna and OPTUNA_AVAILABLE:
            st.subheader("📈 Optuna最適化（進捗）")
            container = st.empty()
            study = solve_optuna_fixed_char(Q, fixed_god_id=fixed_god_id, n_trials=n_trials, progress_container=container)
            if study is not None:
                with st.expander("Optuna可視化", expanded=False):
                    tabs = st.tabs(["履歴", "重要度", "パラレル", "等高線", "スライス", "タイムライン"])
                    try:
                        with tabs[0]:
                            st.plotly_chart(plot_optimization_history(study), use_container_width=True)
                        with tabs[1]:
                            st.plotly_chart(plot_param_importances(study), use_container_width=True)
                        with tabs[2]:
                            st.plotly_chart(plot_parallel_coordinate(study), use_container_width=True)
                        with tabs[3]:
                            params = list(study.best_params.keys())
                            if len(params) >= 2:
                                st.plotly_chart(plot_contour(study, params=[params[0], params[1]]), use_container_width=True)
                            else:
                                st.info("等高線表示には2パラメータ以上が必要です")
                        with tabs[4]:
                            st.plotly_chart(plot_slice(study), use_container_width=True)
                        with tabs[5]:
                            st.plotly_chart(plot_timeline(study), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Optuna可視化の一部に失敗: {e}")

        # exact landscape（★全神対象に変更★）
        sols = solve_exact_all_gods(Q)
        topN = 20
        energies = [e for e, _ in sols[:topN]]
        st.subheader("🗺️ エネルギー地形（上位候補・★全12神が対象★）")
        fig_bar = px.bar(x=[f"候補{i+1}" for i in range(topN)], y=energies,
                         labels={"x": "候補", "y": "エネルギー"},
                         title="Energy landscape（低いほど縁が結ばれやすい）")
        fig_bar.update_xaxes(tickangle=-60)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ★Boltzmann sampling（量子的確率選択）
        pool = sols[:20]
        T = calculate_quantum_temperature(mood)

        st.write(f"**量子温度（揺らぎの強さ）**: {T:.3f}")
        st.caption("温度が高いほど多様な選択、低いほど最適解に集中")

        e_pick, x_pick = boltzmann_sample(pool, temperature=T, use_quantum_random=True)

        # 選ばれた神を特定
        c_start = 20
        selected_god_id = int(np.argmax(x_pick[c_start:c_start+12]))
        god = (LOADED_GODS if LOADED_GODS else TWELVE_GODS)[selected_god_id]
        
        # picks / short oracle
        picks = select_picks_for_god(god, user_text, top_k=3)
        poem, hint = compose_poem_and_hint(picks, mood)
        short_text, meta = generate_short_oracle_always(user_text, god, picks, mood,
                                                        llm_enabled, hf_model,
                                                        hf_max_new_tokens, hf_temperature, hf_top_p)
        st.session_state["last_llm_meta"] = meta

        st.markdown("---")
        st.subheader(f"🎴 {god['emoji']} {god['name']} からの神託")
        st.write(f"**選ばれた神**: {god['name']} ({god['name_en']})")
        st.write(f"**エネルギー**: {e_pick:.3f} / **温度**: {T:.3f}")

        # キャラクター画像表示
        img = get_character_image_path(selected_god_id)
        if img:
            st.image(img, width=320)
            
        st.markdown("### ✨ 神託（短文）")
        st.success(short_text)

        st.markdown("### 📜 選ばれた縁（格言）")
        for p in picks:
            src = get_maxim_source(p)
            st.markdown(f"- **{p}** *(出典: {src['source']})*")

        st.markdown(f"### 🍃 ことば（短句）\n「{poem}」")
        st.markdown(f"### 👣 次の一歩\n{hint}")

        st.markdown("---")
        st.subheader("🪐 キーワードのエネルギー球体（補助可視化）")
        kws = extract_keywords(user_text, top_n=8)
        if kws:
            st.caption("抽出キーワード: " + ", ".join(kws))
            net = build_word_network(kws, GLOBAL_WORDS_DATABASE, n_neighbors=20)
            centers = [i for i, w in enumerate(net["words"]) if w in kws]
            pos = place_words_on_sphere(len(net["words"]), centers)
            fig = create_3d_network_plot(net, pos, centers)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()