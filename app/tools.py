"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
from functools import cache
from pathlib import Path
from datetime import datetime, timezone
import random

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RESULTS = 3


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]

    # """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    # AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    # 答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    # """


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を１つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e
        for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e
        for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def save_learning_log(topic: str, summary: str) -> str:
    """利用者の学習内容を学習記録ファイルに保存する。

    学習した内容を記録してほしいと利用者から依頼された場合に使用する。
    保存する前に、必ず利用者の承認を得ること。

    Args:
        topic (str): 学習したテーマや用語。
        summary (str): 学習内容の要約。

    Returns:
        str: 保存結果を示すメッセージ。
    """
    log_path = DATA_DIR / "learning_log.json"

    # 既存の学習記録を読み込む
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []

    # 新しい学習記録を追加
    logs.append(
        {
            "topic": topic,
            "summary": summary,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    # JSONファイルに保存
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    return f"「{topic}」を学習記録に保存しました。"


def make_quiz(lecture: int) -> str:
    """指定した回の用語から1つを選び、確認問題と答えを返す。

    「第2回の確認問題を出して」のように、
    復習用の問題を求められたときに使う。
    """

    # 1. 用語集を読み込む
    terms = load_glossary()

    # 2. 指定した回の用語を集める
    lecture_terms = [entry for entry in terms if entry.get("lecture") == lecture]

    # 該当する用語がない場合
    if not lecture_terms:
        return f"第{lecture}回の用語は用語集に登録されていません。"

    # 3. 用語を1つランダムに選ぶ
    selected = random.choice(lecture_terms)

    # 4. 問題と答えを作成
    quiz = {
        "question": f"「{selected['term']}」の意味は？",
        "answer": selected["definition"],
    }

    # 5. JSON形式の文字列で返す
    return json.dumps(quiz, ensure_ascii=False, indent=2)
