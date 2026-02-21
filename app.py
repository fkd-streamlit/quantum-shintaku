# -*- coding: utf-8 -*-
# ============================================================
# QUBO × 量子神託 UI（Streamlit + Plotly）
# - 入力文からキーワード抽出
# - キーワード中心に「エネルギーが近い単語」が集まるネットワークを構築
# - 3Dで“球体（言葉）＋縁（線）＋星屑（宇宙）”を描画
# - 格言は「出所（典拠/作者/意訳/創作）」も表示
# - マウスで回転/ズーム/リセット可能
# ============================================================

import re
import time
import random
import os
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# pandasとopenpyxlをインポート（Excel読み込み用）
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# =========================
# 0) ページ設定 + CSS（宇宙）
# =========================
st.set_page_config(page_title="量子神託 - 縁の球体", layout="wide")
from pathlib import Path

# ======================
# BGM設定
# ======================
BGM_PATH = Path("assets/bgm.mp3")

if "bgm_on" not in st.session_state:
    st.session_state.bgm_on = True

# セッション状態の初期化（最初に実行）
if "excel_quotes_loaded" not in st.session_state:
    st.session_state.excel_quotes_loaded = False
    st.session_state.excel_quotes = []

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

# =========================
# 1) グローバル単語DB（他の人の言葉）
# =========================
GLOBAL_WORDS_DATABASE = [
    "世界平和","貢献","成長","学び","挑戦","夢","希望","未来",
    "感謝","愛","幸せ","喜び","安心","充実","満足","平和",
    "努力","継続","忍耐","誠実","正直","優しさ","思いやり","共感",
    "調和","バランス","自然","美","真実","自由","正義","道",
    "絆","つながり","家族","友人","仲間","信頼","尊敬","協力",
    "今","瞬間","過程","変化","進化","発展","循環","流れ",
    "静けさ","集中","覚悟","決意","勇気","強さ","柔軟性","寛容",
]

# =========================
# 2) 格言DB（出所も持たせる）
# =========================

def load_quotes_from_excel(excel_path: str = None) -> List[Dict]:
    """
    ExcelファイルのQUOTESワークシートから格言を読み込む
    """
    quotes = []
    
    if not PANDAS_AVAILABLE:
        return quotes
    
    # デフォルトのExcelファイルパス
    if excel_path is None:
        excel_path = "quantum_shintaku_pack_v3_with_sense_20260213_oposite_modify_with_lr022101.xlsx"
    
    if not os.path.exists(excel_path):
        return quotes
    
    try:
        # QUOTESワークシートを読み込む
        df = pd.read_excel(excel_path, sheet_name='QUOTES', engine='openpyxl')
        
        # デバッグ: 列名を表示
        print(f"Excelファイルの列名: {df.columns.tolist()}")
        print(f"Excelファイルの行数: {len(df)}")
        
        # 列名を確認して適切にマッピング
        for idx, row in df.iterrows():
            quote_dict = {}
            
            # 格言テキスト（様々な列名に対応）
            quote_text = None
            for col in ['格言', 'QUOTE', 'Quote', 'quote', 'テキスト', '文', '言葉']:
                if col in df.columns:
                    quote_text = str(row.get(col, "")).strip()
                    if quote_text and quote_text.lower() not in ("nan", "none", ""):
                        break
            
            if not quote_text:
                continue
            
            quote_dict["quote"] = quote_text
            
            # キーワード（様々な列名に対応）
            keywords = []
            for col in ['キーワード', 'KEYWORDS', 'Keywords', 'keywords', 'タグ', 'TAG', 'Tag']:
                if col in df.columns:
                    kw_str = str(row.get(col, "")).strip()
                    if kw_str and kw_str.lower() not in ("nan", "none", ""):
                        keywords = [k.strip() for k in kw_str.replace("、", ",").split(",") if k.strip()]
                        break
            
            quote_dict["keywords"] = keywords if keywords else []
            
            # 出典（様々な列名に対応）
            source = None
            for col in ['出典', 'SOURCE', 'Source', 'source', '出所', '典拠', '作者']:
                if col in df.columns:
                    source = str(row.get(col, "")).strip()
                    if source and source.lower() not in ("nan", "none", ""):
                        break
            
            quote_dict["source"] = source or "伝統的な教え"
            
            # 備考（様々な列名に対応）
            note = None
            for col in ['備考', 'NOTE', 'Note', 'note', '注', 'メモ']:
                if col in df.columns:
                    note = str(row.get(col, "")).strip()
                    if note and note.lower() not in ("nan", "none", ""):
                        break
            
            quote_dict["note"] = note or ""
            
            quotes.append(quote_dict)
        
        return quotes
    
    except Exception as e:
        # エラーが発生しても既存のFAMOUS_QUOTESを使用
        # st.warningはStreamlitコンテキスト外では使えないので、printで代用
        print(f"Excelファイルからの格言読み込みに失敗: {e}")
        import traceback
        traceback.print_exc()
        return []

# Excelファイルから格言を読み込む（存在する場合）
# 注意: Streamlitの実行時には毎回読み込まれるため、セッション状態に保存する
try:
    if not st.session_state.excel_quotes_loaded:
        st.session_state.excel_quotes = load_quotes_from_excel()
        st.session_state.excel_quotes_loaded = True
        if st.session_state.excel_quotes:
            print(f"Excelから{len(st.session_state.excel_quotes)}件の格言を読み込みました")
except Exception as e:
    # エラーが発生しても処理を続行
    print(f"Excel格言読み込みの初期化エラー: {e}")
    import traceback
    traceback.print_exc()
    st.session_state.excel_quotes_loaded = True
    st.session_state.excel_quotes = []

# 既存のFAMOUS_QUOTES（基本の格言）
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

# Excelから読み込んだ格言を追加（既存のものと重複しないように）
# FAMOUS_QUOTESを初期化（BASE_FAMOUS_QUOTESから開始）
FAMOUS_QUOTES = BASE_FAMOUS_QUOTES.copy()

try:
    excel_quotes = st.session_state.get("excel_quotes", [])
    if excel_quotes:
        # 既存の格言のテキストを取得
        existing_quotes = {q.get("quote", "") for q in FAMOUS_QUOTES}
        
        # 新しい格言を追加
        added_count = 0
        for excel_quote in excel_quotes:
            excel_quote_text = excel_quote.get("quote", "")
            if excel_quote_text and excel_quote_text not in existing_quotes:
                FAMOUS_QUOTES.append(excel_quote)
                existing_quotes.add(excel_quote_text)
                added_count += 1
        
        if added_count > 0:
            print(f"Excelから{added_count}件の新しい格言を追加しました（合計: {len(FAMOUS_QUOTES)}件）")
except Exception as e:
    # エラーが発生しても処理を続行（BASE_FAMOUS_QUOTESのみ使用）
    print(f"Excel格言の統合エラー: {e}")
    import traceback
    traceback.print_exc()

# =========================
# 3) テキスト→キーワード抽出（簡易）
# =========================
def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    text = (text or "").strip()
    # 記号/数字をざっくり除去
    text_clean = re.sub(r"[0-9０-９\W]+", " ", text)

    found = [w for w in GLOBAL_WORDS_DATABASE if w in text_clean]
    if found:
        # 入力に含まれたDB語を優先
        return found[:top_n]

    # fallback：スペース区切りや短文から拾う
    tokens = [t for t in text_clean.split() if len(t) >= 2]
    if not tokens:
        return ["静けさ", "迷い"]  # 何もなければ保険
    return tokens[:top_n]

# =========================
# 4) “エネルギー”計算（QUBO的相互作用の精密モデル）
# =========================
CATEGORIES = {
    "願い": ["世界平和","貢献","成長","夢","希望","未来"],
    "感情": ["感謝","愛","幸せ","喜び","安心","満足","平和"],
    "行動": ["努力","継続","忍耐","誠実","正直"],
    "哲学": ["調和","バランス","自然","美","道","真実","自由","正義"],
    "関係": ["絆","つながり","家族","友人","仲間","信頼","尊敬","協力"],
    "内的": ["静けさ","集中","覚悟","決意","勇気","強さ","柔軟性","寛容"],
    "時間": ["今","瞬間","過程","変化","進化","発展","循環","流れ"],
}

# 単語間の意味的類似度マトリックス（事前計算用）
WORD_SEMANTIC_WEIGHTS = {}

def calculate_semantic_similarity(word1: str, word2: str) -> float:
    """意味的類似度を計算（0-1の範囲、1に近いほど類似）"""
    if word1 == word2:
        return 1.0
    
    # 文字レベルの類似度
    common_chars = set(word1) & set(word2)
    char_sim = len(common_chars) / max(len(set(word1)), len(set(word2)), 1)
    
    # カテゴリの一致度
    category_sim = 0.0
    for _, ws in CATEGORIES.items():
        w1_in = word1 in ws
        w2_in = word2 in ws
        if w1_in and w2_in:
            category_sim = 1.0
            break
        elif w1_in or w2_in:
            category_sim = 0.3
    
    # 長さの類似度
    len_sim = 1.0 - abs(len(word1) - len(word2)) / max(len(word1), len(word2), 1)
    
    # 重み付き平均
    similarity = 0.4 * char_sim + 0.4 * category_sim + 0.2 * len_sim
    return float(np.clip(similarity, 0.0, 1.0))

def calculate_energy_between_words(word1: str, word2: str, rng: np.random.Generator, jitter: float) -> float:
    """
    より精密なエネルギー計算（QUBO相互作用）
    小さい（より負）ほど“近い”扱い。
    """
    # 意味的類似度からエネルギーを計算
    similarity = calculate_semantic_similarity(word1, word2)
    
    # 類似度が高いほど負のエネルギー（結合が強い）
    # エネルギー = -2.0 * similarity + 0.5（ベースライン）
    energy = -2.0 * similarity + 0.5
    
    # 文字の共通成分による補正
    common = set(word1) & set(word2)
    if common:
        energy -= 0.20 * len(common) / max(len(word1), len(word2), 1)
    
    # 同カテゴリならさらに近づく
    for _, ws in CATEGORIES.items():
        if (word1 in ws) and (word2 in ws):
            energy -= 0.60
            break
    
    # 量子的揺らぎ（QUBOの本質）
    energy += rng.normal(0, jitter)
    return float(energy)

def build_qubo_matrix_for_words(words: List[str], rng: np.random.Generator, jitter: float) -> Dict[Tuple[int, int], float]:
    """
    単語ネットワーク用のQUBOマトリックスを構築
    Q[i,j] = 単語iと単語jの相互作用エネルギー
    """
    n = len(words)
    Q: Dict[Tuple[int, int], float] = {}
    
    # 対角項（各単語のバイアス）
    for i in range(n):
        # 中心語はより低いエネルギー（選択されやすい）
        Q[(i, i)] = -0.5 if i < len(words) else 0.0
    
    # 非対角項（単語間の相互作用）
    for i in range(n):
        for j in range(i + 1, n):
            energy = calculate_energy_between_words(words[i], words[j], rng, jitter)
            # QUBO形式：x_i * x_j の係数
            Q[(i, j)] = energy
            Q[(j, i)] = energy  # 対称性
    
    return Q

def solve_qubo_placement(Q: Dict[Tuple[int, int], float], n_words: int, 
                         center_indices: List[int], rng: np.random.Generator,
                         n_iterations: int = 100, progress_callback=None,
                         energies_dict: Dict[str, float] = None,
                         words_list: List[str] = None) -> np.ndarray:
    """
    QUBO最適化を使って単語の3D配置を決定
    中心語を原点に配置し、エネルギーに基づいて距離を決定
    """
    pos = np.zeros((n_words, 3), dtype=float)
    
    # 中心語を原点（0,0,0）に配置
    for idx in center_indices:
        if idx < n_words:
            pos[idx] = [0.0, 0.0, 0.0]
    
    # エネルギーに基づいて各単語の中心語からの距離を決定
    if energies_dict is None:
        energies_dict = {}
    if words_list is None:
        words_list = []
    
    # エネルギーの範囲を取得（距離計算用）
    energy_values = list(energies_dict.values()) if energies_dict else []
    if energy_values:
        min_energy = min(energy_values)
        max_energy = max(energy_values)
        energy_range = max_energy - min_energy if max_energy != min_energy else 1.0
    else:
        min_energy = -3.0
        energy_range = 3.0
    
    # 各単語をエネルギーに基づいて配置
    golden_angle = np.pi * (3 - np.sqrt(5))
    word_idx = 0
    
    for i in range(n_words):
        if i in center_indices:
            continue  # 中心語は既に配置済み
        
        # 単語のインデックスからエネルギーを取得
        if i < len(words_list):
            word = words_list[i]
            energy = energies_dict.get(word, 0.0)
        else:
            energy = 0.0
        
        # エネルギーを距離に変換（負のエネルギーが大きいほど近く）
        # エネルギー範囲: -3.0 ～ 0.0 → 距離範囲: 0.3 ～ 2.5
        normalized_energy = (energy - min_energy) / energy_range if energy_range > 0 else 0.5
        distance = 0.3 + (1.0 - normalized_energy) * 2.2  # 0.3から2.5の範囲
        
        # 球面上に均等に配置（中心語からの距離を維持）
        theta = golden_angle * word_idx
        y = 1 - (word_idx / float(max(1, n_words - len(center_indices) - 1))) * 2
        radius_at_y = np.sqrt(max(0.0, 1 - y * y))
        
        # 距離を適用
        x = np.cos(theta) * radius_at_y * distance
        z = np.sin(theta) * radius_at_y * distance
        
        pos[i] = [x, y * distance * 0.6, z]  # y方向も距離に比例
        word_idx += 1
    
    # QUBOエネルギーに基づいて微調整（単語間の相互作用）
    for iteration in range(n_iterations):
        for i in range(n_words):
            if i in center_indices:
                continue  # 中心語は動かさない
            
            force = np.zeros(3, dtype=float)
            
            # 中心語からの引力（エネルギーに基づく）
            for center_idx in center_indices:
                vec_to_center = pos[center_idx] - pos[i]
                dist_to_center = np.linalg.norm(vec_to_center)
                if dist_to_center > 0.01:
                    # エネルギーが低いほど強く引き合う
                    if i < len(words_list):
                        word = words_list[i]
                        energy = energies_dict.get(word, 0.0)
                    else:
                        energy = 0.0
                    target_distance = 0.3 + (1.0 - (energy - min_energy) / energy_range) * 2.2 if energy_range > 0 else 1.5
                    
                    # 目標距離に向かう力
                    if dist_to_center < target_distance * 0.9:
                        # 近すぎる場合は少し離す
                        force -= vec_to_center / dist_to_center * 0.05
                    elif dist_to_center > target_distance * 1.1:
                        # 遠すぎる場合は近づける
                        force += vec_to_center / dist_to_center * 0.1
            
            # 他の単語との相互作用
            for j in range(n_words):
                if i == j or j in center_indices:
                    continue
                
                energy = Q.get((i, j), 0.0)
                if energy < -0.3:  # 強い負のエネルギー = 引き合う
                    vec = pos[j] - pos[i]
                    dist = np.linalg.norm(vec)
                    if dist > 0.01:
                        strength = abs(energy) * 0.08
                        force += vec / dist * strength
                elif energy > 0.2:  # 正のエネルギー = 反発
                    vec = pos[i] - pos[j]
                    dist = np.linalg.norm(vec)
                    if dist > 0.01:
                        strength = abs(energy) * 0.03
                        force += vec / dist * strength
            
            # 位置を更新
            pos[i] += force * 0.15
        
        # 進捗コールバック
        if progress_callback:
            progress_callback(iteration + 1, n_iterations)
    
    return pos

def build_word_network(center_words: List[str], database: List[str], n_total: int,
                       rng: np.random.Generator, jitter: float) -> Dict:
    """
    より精密なQUBOベースの単語ネットワーク構築
    """
    all_words = list(set(center_words + database))
    energies = {}

    # 中心語とのエネルギーを計算
    for w in all_words:
        if w in center_words:
            energies[w] = -3.0  # 中心語は非常に低いエネルギー
        else:
            e_list = [calculate_energy_between_words(c, w, rng, jitter) for c in center_words]
            energies[w] = float(np.mean(e_list))

    # エネルギー順にソート
    sorted_words = sorted(energies.items(), key=lambda x: x[1])

    # 中心語を優先的に選択
    selected = []
    for w, _ in sorted_words:
        if w in center_words:
            selected.append(w)
    
    # エネルギーが低い順に追加
    for w, _ in sorted_words:
        if w not in selected:
            selected.append(w)
        if len(selected) >= n_total:
            break

    # QUBOマトリックスを構築
    Q = build_qubo_matrix_for_words(selected, rng, jitter)
    
    # エッジを計算（QUBOエネルギーに基づく）
    edges = []
    center_indices = [i for i, w in enumerate(selected) if w in center_words]
    
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            energy = Q.get((i, j), 0.0)
            # 負のエネルギー（結合が強い）のみエッジとして追加
            if energy < -0.25:
                edges.append((i, j, energy))

    return {
        "words": selected, 
        "energies": {w: energies[w] for w in selected}, 
        "edges": edges,
        "qubo_matrix": Q,
        "center_indices": center_indices
    }

# =========================
# 5) 3D配置（QUBO最適化ベース）
# =========================
def place_words_3d(words: List[str], center_set: set, rng: np.random.Generator, 
                   noise: float, network: Dict = None, n_iterations: int = 80,
                   progress_callback=None) -> np.ndarray:
    """
    QUBO最適化を使ってエネルギーに基づいた3D配置を生成
    """
    n = len(words)
    
    # QUBOマトリックスが提供されている場合はそれを使用
    if network and "qubo_matrix" in network and "center_indices" in network:
        Q = network["qubo_matrix"]
        center_indices = network["center_indices"]
        energies_dict = network.get("energies", {})
        pos = solve_qubo_placement(Q, n, center_indices, rng, n_iterations=n_iterations,
                                   progress_callback=progress_callback,
                                   energies_dict=energies_dict,
                                   words_list=words)
    else:
        # フォールバック：従来の方法
        pos = np.zeros((n, 3), dtype=float)
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            w = words[i]
            theta = golden_angle * i
            y = 1 - (i / float(max(1, n - 1))) * 2
            radius_at_y = np.sqrt(max(0.0, 1 - y * y))
            r = 1.0 + rng.uniform(-0.15, 0.20)
            x = np.cos(theta) * radius_at_y * r
            z = np.sin(theta) * radius_at_y * r
            if w in center_set:
                x *= 0.35
                y *= 0.35
                z += 1.10
            pos[i] = [x, y, z]
    
    # 最終的な揺らぎ
    pos += rng.normal(0, noise, size=pos.shape)
    return pos

# =========================
# 6) 格言選択（出所つき）
# =========================
def select_relevant_quote(keywords: List[str]) -> Dict[str, str]:
    """
    キーワードに基づいて最も関連性の高い格言を選択
    """
    if not keywords:
        keywords = ["今"]
    
    # キーワードを正規化（小文字化、部分文字列も考慮）
    ks_normalized = set()
    for kw in keywords:
        kw_clean = kw.strip().lower()
        ks_normalized.add(kw_clean)
        # 部分文字列も追加（例：「人との会話に疲れた」→「疲れた」「会話」など）
        if len(kw_clean) > 2:
            for i in range(len(kw_clean) - 1):
                if len(kw_clean[i:i+2]) >= 2:
                    ks_normalized.add(kw_clean[i:i+2])
    
    best = None
    best_score = -1.0

    for q in FAMOUS_QUOTES:
        quote_keywords = q.get("keywords", [])
        if not quote_keywords:
            continue
        
        # 格言のキーワードも正規化
        quote_kw_normalized = set()
        for qkw in quote_keywords:
            qkw_clean = qkw.strip().lower()
            quote_kw_normalized.add(qkw_clean)
            # 部分文字列も追加
            if len(qkw_clean) > 2:
                for i in range(len(qkw_clean) - 1):
                    if len(qkw_clean[i:i+2]) >= 2:
                        quote_kw_normalized.add(qkw_clean[i:i+2])
        
        # 完全一致のスコア
        exact_match = len(ks_normalized & quote_kw_normalized)
        
        # 部分一致のスコア（キーワードが格言のキーワードに含まれる、またはその逆）
        partial_match = 0.0
        for kw in ks_normalized:
            for qkw in quote_kw_normalized:
                if kw in qkw or qkw in kw:
                    partial_match += 0.5
        
        # 格言のテキスト内にキーワードが含まれる場合も加点
        quote_text = q.get("quote", "").lower()
        text_match = 0.0
        for kw in ks_normalized:
            if len(kw) >= 2 and kw in quote_text:
                text_match += 0.3
        
        # 総合スコア
        score = exact_match * 2.0 + partial_match + text_match
        
        if score > best_score:
            best_score = score
            best = q

    # デバッグ情報（必要に応じてコメントアウト）
    # if best:
    #     print(f"選択された格言: {best.get('quote', '')[:50]}... (スコア: {best_score:.2f})")
    
    if best is None or best_score < 0.1:
        return {
            "quote": "あなたの観測が、この世界線を確定させました。",
            "source": "量子神託 試作（福田雅彦）—創作",
            "note": ""
        }
    
    return {
        "quote": best.get("quote", ""),
        "source": best.get("source", "伝統的な教え"),
        "note": best.get("note", "")
    }

# =========================
# 7) UI
# =========================
st.title("量子神託（試作）— 縁の球体（QUBO × アート）")

# セッション状態の初期化
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "last_params_hash" not in st.session_state:
    st.session_state.last_params_hash = ""

with st.sidebar:
    st.markdown("### 今の気持ち（入力）")
    user_input = st.text_area(
        "短い一文でOK（例：人との会話に疲れた。少し迷っている。）",
        value="人との会話に疲れた。少し迷っている。",
        height=90,
        key="user_input_text"
    )

    st.markdown("---")
    st.markdown("### パラメータ")
    top_n = st.slider("抽出キーワード数", 2, 10, 5, 1)
    n_total = st.slider("空間に出す単語数（中心＋周辺）", 15, 60, 30, 1)

    auto = st.toggle("ゆらぎ（自動更新）", value=True)
    refresh_ms = st.slider("更新間隔(ms)", 200, 1500, 650, 50)

    noise = st.slider("位置のゆらぎ", 0.00, 0.20, 0.06, 0.01)
    jitter = st.slider("エネルギー揺らぎ", 0.00, 0.25, 0.10, 0.01)
    
    # QUBO最適化の反復回数（計算時間を調整可能に）
    qubo_iterations = st.slider("QUBO最適化の反復回数", 50, 200, 80, 10, 
                                help="少ないほど速いが、配置の精度は下がります")

    st.markdown("---")
    st.markdown("### 宇宙の密度")
    star_count = st.slider("星屑の数", 200, 2200, 900, 50)
    star_twinkle = st.slider("星のまたたき", 0.00, 0.15, 0.04, 0.01)

    st.markdown("---")
    enable_zoom = st.toggle("マウスホイールでズーム", value=True)
    
    # 手動更新ボタン
    if st.button("🔄 再計算", use_container_width=True):
        st.session_state.last_user_input = ""  # 強制的に再計算
        st.rerun()

# 入力またはパラメータが変更されたかチェック
params_hash = f"{user_input}_{top_n}_{n_total}_{noise}_{jitter}_{qubo_iterations}"
input_changed = user_input != st.session_state.last_user_input
params_changed = params_hash != st.session_state.last_params_hash
needs_recalc = input_changed or params_changed

# 自動更新（入力変更時は一時停止）
if auto and not needs_recalc:
    # 自動更新中であることを示す（メインエリアに表示）
    if "network" in st.session_state:
        st.caption(f"🔄 自動更新中（{refresh_ms}ms間隔） - 球体がゆらぎます")
    st_autorefresh(interval=refresh_ms, key="refresh")

# 入力変更時は再計算を明示的に実行
if needs_recalc:
    st.session_state.last_user_input = user_input
    st.session_state.last_params_hash = params_hash

# 計算中表示
if needs_recalc:
    # 進捗バー用のプレースホルダー
    progress_placeholder = st.empty()
    
    # RNG（揺らぎを毎回変える）
    rng = np.random.default_rng(int(time.time() * 1000) % (2**32 - 1))

    # キーワード抽出
    with progress_placeholder.container():
        st.info("🔄 計算を開始します...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("📝 キーワードを抽出中...")
        progress_bar.progress(10)
    
    keywords = extract_keywords(user_input, top_n=top_n)
    center_set = set(keywords)

    # ネットワーク構築（QUBOベース）
    with progress_placeholder.container():
        status_text.text("🔗 単語ネットワークを構築中...")
        progress_bar.progress(30)
    
    network = build_word_network(keywords, GLOBAL_WORDS_DATABASE, n_total=n_total, rng=rng, jitter=jitter)

    # 3D位置（QUBO最適化ベース）- 反復回数を調整可能に
    with progress_placeholder.container():
        status_text.text("🌐 QUBO最適化で3D配置を計算中...")
        progress_bar.progress(50)
    
    # 進捗コールバック関数
    def update_progress(current, total):
        progress = 50 + int((current / total) * 40)  # 50%から90%まで
        with progress_placeholder.container():
            progress_bar.progress(progress)
            status_text.text(f"🌐 QUBO最適化中... ({current}/{total} 反復)")
    
    pos = place_words_3d(network["words"], center_set=center_set, rng=rng, noise=noise, 
                        network=network, n_iterations=qubo_iterations,
                        progress_callback=update_progress)
    
    # 完了
    with progress_placeholder.container():
        progress_bar.progress(100)
        status_text.text("✅ 計算完了！")
        time.sleep(0.2)
    
    # 進捗バーを削除（次のレンダリングで自動的に上書きされる）
    progress_placeholder.empty()
    
    # セッション状態に保存
    st.session_state.network = network
    st.session_state.pos = pos
    st.session_state.keywords = keywords
    st.session_state.center_set = center_set
else:
    # 前回の計算結果を使用（自動更新時の揺らぎのみ）
    if "network" in st.session_state and "pos" in st.session_state:
        network = st.session_state.network
        pos = st.session_state.pos.copy()  # コピーを作成
        keywords = st.session_state.keywords
        center_set = st.session_state.center_set
        
        # 自動更新時は位置に小さな揺らぎを追加（視覚的な動きを追加）
        if auto:
            rng = np.random.default_rng(int(time.time() * 1000) % (2**32 - 1))
            # より視覚的に分かる揺らぎを追加（自動更新時の動きを強調）
            pos = pos + rng.normal(0, noise * 0.6, size=pos.shape)
    else:
        # 初回実行時
        progress_container = st.container()
        with progress_container:
            st.info("🔄 計算を開始します...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        rng = np.random.default_rng(int(time.time() * 1000) % (2**32 - 1))
        
        with progress_container:
            status_text.text("📝 キーワードを抽出中...")
            progress_bar.progress(10)
        
        keywords = extract_keywords(user_input, top_n=top_n)
        center_set = set(keywords)
        
        with progress_container:
            status_text.text("🔗 単語ネットワークを構築中...")
            progress_bar.progress(30)
        
        network = build_word_network(keywords, GLOBAL_WORDS_DATABASE, n_total=n_total, rng=rng, jitter=jitter)
        
        with progress_container:
            status_text.text("🌐 QUBO最適化で3D配置を計算中...")
            progress_bar.progress(50)
        
        def update_progress(current, total):
            progress = 50 + int((current / total) * 40)
            progress_bar.progress(progress)
            status_text.text(f"🌐 QUBO最適化中... ({current}/{total} 反復)")
        
        pos = place_words_3d(network["words"], center_set=center_set, rng=rng, noise=noise, 
                            network=network, n_iterations=qubo_iterations,
                            progress_callback=update_progress)
        
        with progress_container:
            progress_bar.progress(100)
            status_text.text("✅ 計算完了！")
            time.sleep(0.3)
        
        # 進捗バーは次のレンダリングで自動的に上書きされる
        
        st.session_state.network = network
        st.session_state.pos = pos
        st.session_state.keywords = keywords
        st.session_state.center_set = center_set

# =========================
# 8) Plotly描画（星屑＋縁＋球体＋ラベル）
# =========================
# networkとposが確実に存在することを確認
try:
    # 変数が定義されているか確認
    _ = network
    _ = pos
    _ = keywords
    _ = center_set
except NameError:
    # セッション状態から取得を試みる
    if "network" in st.session_state and "pos" in st.session_state:
        network = st.session_state.network
        pos = st.session_state.pos
        keywords = st.session_state.keywords
        center_set = st.session_state.center_set
    else:
        # 初期化されていない場合はエラーメッセージを表示
        st.warning("⚠️ データが初期化されていません。計算中です...")
        st.info("入力テキストを変更するか、「🔄 再計算」ボタンを押してください。")
        st.stop()

# データの整合性チェック
if network is None or pos is None or len(network.get("words", [])) == 0:
    st.warning("⚠️ データが不完全です。再計算してください。")
    st.stop()

fig = go.Figure()

# 星屑（背景）
star_rng = np.random.default_rng(12345)  # 固定seedでチラつき抑制
sx = star_rng.uniform(-3.2, 3.2, star_count)
sy = star_rng.uniform(-2.4, 2.4, star_count)
sz = star_rng.uniform(-2.0, 2.0, star_count)

# RNG（星屑用）
star_rng_for_twinkle = np.random.default_rng(int(time.time() * 1000) % (2**32 - 1))
tw = np.clip(star_rng_for_twinkle.normal(0, star_twinkle, size=star_count), -0.15, 0.15)
alpha = np.clip(0.22 + tw, 0.10, 0.42)
star_size = star_rng.uniform(1.0, 2.4, star_count)
star_colors = [f"rgba(255,255,255,{a})" for a in alpha]

fig.add_trace(go.Scatter3d(
    x=sx, y=sy, z=sz,
    mode="markers",
    marker=dict(size=star_size, color=star_colors),
    hoverinfo="skip",
    showlegend=False
))

# 中心語から各単語への線（距離を視覚化）
center_indices = network.get("center_indices", [])
words = network["words"]
energies_dict = network.get("energies", {})

for center_idx in center_indices:
    if center_idx >= len(words):
        continue
    
    center_word = words[center_idx]
    cx, cy, cz = pos[center_idx]
    
    # 中心語から各単語への線を描画
    for i, word in enumerate(words):
        if i == center_idx or i in center_indices:
            continue
        
        x, y, z = pos[i]
        energy = energies_dict.get(word, 0.0)
        
        # 距離を計算
        distance = np.linalg.norm(pos[i] - pos[center_idx])
        
        # エネルギーに基づいて線の太さと色を決定
        # エネルギーが低い（近い）ほど太く明るく
        energy_normalized = min(1.0, abs(energy) / 3.0)
        lw = 1.0 + 3.0 * energy_normalized
        a = 0.3 + 0.5 * energy_normalized
        
        # 距離に応じて色を変える（近い=明るい青、遠い=薄い青）
        if distance < 1.0:
            color = f"rgba(100,200,255,{a})"  # 近い = 明るい青
        elif distance < 1.8:
            color = f"rgba(150,200,255,{a * 0.7})"  # 中距離 = 青
        else:
            color = f"rgba(200,220,255,{a * 0.4})"  # 遠い = 薄い青
        
        # 中心語から各単語への線
        fig.add_trace(go.Scatter3d(
            x=[cx, x], y=[cy, y], z=[cz, z],
            mode="lines",
            line=dict(width=lw, color=color),
            hovertemplate=f"<b>{center_word}</b> → <b>{word}</b><br>" +
                         f"距離: {distance:.2f}<br>" +
                         f"エネルギー: {energy:.2f}<extra></extra>",
            showlegend=False
        ))

# 単語間のエッジ（関連ワード同士を繋ぐ）
for i, j, e in network["edges"]:
    # 中心語は既に上で描画済みなのでスキップ
    if i in center_indices or j in center_indices:
        continue
    
    x0, y0, z0 = pos[i]
    x1, y1, z1 = pos[j]
    
    # 距離を計算
    distance = np.linalg.norm(pos[j] - pos[i])
    
    # エネルギーが低い（負の値が大きい）ほど強い結合
    energy_strength = abs(e)
    normalized_strength = min(1.0, energy_strength / 2.0)
    
    # 線の太さ（エネルギーが低いほど太く）
    lw = 0.5 + 2.0 * normalized_strength
    
    # 透明度（強い結合ほど明るく）
    a = min(0.70, 0.20 + 0.40 * normalized_strength)
    
    # 色（エネルギーに応じて変化）
    if e < -1.0:
        color = f"rgba(120,180,255,{a})"  # 強い結合 = 明るい青
    elif e < -0.5:
        color = f"rgba(160,200,255,{a})"  # 中程度 = 青
    else:
        color = f"rgba(200,200,255,{a})"  # 弱い結合 = 薄い青

    fig.add_trace(go.Scatter3d(
        x=[x0, x1], y=[y0, y1], z=[z0, z1],
        mode="lines",
        line=dict(width=lw, color=color),
        hovertemplate=f"<b>{words[i]}</b> ↔ <b>{words[j]}</b><br>" +
                     f"距離: {distance:.2f}<br>" +
                     f"エネルギー: {e:.2f}<extra></extra>",
        showlegend=False
    ))

# 球体（言葉）- エネルギーに基づいてサイズと色を変える
words = network["words"]
energies_dict = network.get("energies", {})

# エネルギーに基づいてサイズと色を決定
sizes = []
colors = []
label = []
for w in words:
    energy = energies_dict.get(w, 0.0)
    
    if w in center_set:
        # 中心語は大きく明るく
        sizes.append(28)
        colors.append("rgba(255,235,100,0.98)")  # 金色
        label.append(w)
    else:
        # エネルギーが低いほど大きく明るく
        energy_normalized = min(1.0, abs(energy) / 3.0)
        size = 12 + int(8 * energy_normalized)
        sizes.append(size)
        
        # エネルギーに応じて色を変える
        if energy < -1.5:
            colors.append("rgba(180,220,255,0.85)")  # 低エネルギー = 明るい青
        elif energy < -0.5:
            colors.append("rgba(220,240,255,0.75)")  # 中程度 = 薄い青
        else:
            colors.append("rgba(255,255,255,0.60)")  # 高エネルギー = 白
        
        label.append(w)

# 中心語とその他の単語を分けて描画（文字サイズを個別に制御）
center_texts = []
center_positions = []
other_texts = []
other_positions = []
other_sizes = []
other_colors = []

for i, (w, size, color) in enumerate(zip(label, sizes, colors)):
    if w in center_set:
        center_texts.append(w)
        center_positions.append(pos[i])
    else:
        other_texts.append(w)
        other_positions.append(pos[i])
        other_sizes.append(size)
        other_colors.append(color)

# その他の単語を描画（拡大縮小に応じてサイズが変わる）
if other_texts:
    other_positions = np.array(other_positions)
    fig.add_trace(go.Scatter3d(
        x=other_positions[:, 0], y=other_positions[:, 1], z=other_positions[:, 2],
        mode="markers+text",
        text=other_texts,
        textposition="top center",
        textfont=dict(size=18, color="rgba(255,255,255,1.0)"),
        marker=dict(size=other_sizes, color=other_colors, line=dict(width=1, color="rgba(0,0,0,0.10)")),
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False
    ))

# 中心語を個別に描画（拡大縮小に応じてサイズが変わる）
if center_texts:
    center_positions = np.array(center_positions)
    center_indices_in_label = [i for i, w in enumerate(label) if w in center_set]
    center_sizes = [sizes[i] for i in center_indices_in_label]
    center_colors_list = [colors[i] for i in center_indices_in_label]
    
    fig.add_trace(go.Scatter3d(
        x=center_positions[:, 0], y=center_positions[:, 1], z=center_positions[:, 2],
        mode="markers+text",
        text=center_texts,
        textposition="top center",
        textfont=dict(size=24, color="rgba(255,80,80,1.0)"),  # 中心語は赤色、サイズを大きく
        marker=dict(size=center_sizes, color=center_colors_list, line=dict(width=2, color="rgba(255,80,80,0.8)")),
        hovertemplate="<b>%{text}</b><br>中心語<extra></extra>",
        showlegend=False
    ))

# 中心語を球体として表示（薄い青色の球体）
for center_idx in center_indices:
    if center_idx >= len(words):
        continue
    
    center_word = words[center_idx]
    cx, cy, cz = pos[center_idx]
    
    # 中心語の球体（薄い青色、複数の層で立体感を出す）
    for layer, size_mult in enumerate([1.0, 1.3, 1.6], 1):
        opacity = 0.15 / layer  # 外側ほど薄く
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode="markers",
            marker=dict(
                size=[35 * size_mult],
                color=f"rgba(150,200,255,{opacity})",
                line=dict(width=0)
            ),
            hoverinfo="skip",
            showlegend=False
        ))
    
    # 【】付きのテキストは削除（892-908行目で既に描画済み）

fig.update_layout(
    paper_bgcolor="rgba(6,8,18,1)",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="rgba(6,8,18,1)",
        camera=dict(
            eye=dict(x=1.6, y=1.15, z=1.05),
            center=dict(x=0, y=0, z=0),  # 中心を原点に設定
            up=dict(x=0, y=1, z=0)
        ),
        # ドラッグモードを設定（左クリックで回転軸を変更可能に）
        dragmode="orbit"  # orbitモードで左クリック位置を中心に回転
    ),
    margin=dict(l=0, r=0, t=0, b=0),
)

plotly_config = {
    "displayModeBar": True,
    "scrollZoom": bool(enable_zoom),
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "quantum_oracle",
        "height": 800,
        "width": 1200,
        "scale": 1
    },
    # 拡大縮小に応じて文字サイズも変わるように
    "doubleClick": "reset",
    "modeBarButtonsToAdd": ["select2d", "lasso2d"],
}

# =========================
# 9) レイアウト（左：宇宙 / 右：格言+出所）
# =========================
left, right = st.columns([2.0, 1.0], gap="large")

with left:
    # 自動更新時のステータス表示
    if auto and not needs_recalc and "network" in st.session_state:
        st.caption(f"🔄 自動更新中（{refresh_ms}ms間隔） - 球体がゆらぎます")
    
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)
    st.caption("単語（球体）と縁（線）。マウスで回転・ズームできます。")

with right:
    # 右上の空欄を活用：現在の状態や統計情報を表示
    st.markdown("### 📊 現在の状態")
    st.markdown(f"**計算済み単語数**: {len(network.get('words', []))}語")
    st.markdown(f"**接続数**: {len(network.get('edges', []))}本")
    if "energies" in network:
        min_energy = min(network["energies"].values()) if network["energies"] else 0.0
        max_energy = max(network["energies"].values()) if network["energies"] else 0.0
        st.markdown(f"**エネルギー範囲**: {min_energy:.2f} ～ {max_energy:.2f}")
    st.markdown("---")
    
    # “カード”
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
    
    # デバッグ情報（展開可能）
    with st.expander("🔍 デバッグ情報（格言選択）", expanded=False):
        st.write(f"**抽出キーワード**: {keywords}")
        st.write(f"**利用可能な格言数**: {len(FAMOUS_QUOTES)}件")
        excel_quotes_count = len(st.session_state.get("excel_quotes", []))
        st.write(f"**Excelから読み込んだ格言数**: {excel_quotes_count}件")
        if keywords:
            st.write(f"**選択された格言**: {q.get('quote', '')[:100]}...")
            st.write(f"**出所**: {q.get('source', '—')}")

    st.markdown("### 🎵 音楽")
    st.session_state.bgm_on = st.toggle("BGMを再生", value=st.session_state.bgm_on)

    if st.session_state.bgm_on and BGM_PATH.exists():
        st.audio(str(BGM_PATH), format="audio/mp3")
    
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
