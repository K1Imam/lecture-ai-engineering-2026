"""第3回: Pydantic AI で作る学習支援エージェント。

使い方:
    uv run python -m app.agent "AIE2026の講座で使う「lecNN-start タグ」とは何ですか？"
    uv run python -m app.agent --trace "LoRAとQLoRAの違いは？"   # ツール呼び出しを表示
    uv run python -m app.agent --mcp --trace "LoRAとは？"        # 第5回: ツールを MCP サーバーから受け取る
    uv run python -m app.agent --mcp --list-tools               # 第5回: モデルに見えるツール一覧（APIキー不要）

エージェント = モデル + ループ + ツール。
モデルは tools.py の関数を「ツール」として呼び出し、その結果を見て回答を作る。
第5回では、同じツールを MCP サーバー（app/mcp_server.py）経由で受け取る。
"""

import argparse
import inspect
import json
import time

from pathlib import Path

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.exceptions import ModelHTTPError
from fastmcp import Client
from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.messages import ToolCallPart, ToolReturnPart, ModelMessagesTypeAdapter
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.google import GoogleProvider

from datetime import datetime
from zoneinfo import ZoneInfo

from app import tools
from app.config import MODEL, api_error_message, get_api_key

# 通信エラー時の再試行設定
MAX_ATTEMPTS = 5  # 最大試行回数（初回を含む）
RETRY_WAIT_SECONDS = 1  # 再試行までの待ち時間

# INSTRUCTIONS = """\
# あなたはAIエンジニアリング講座（AIE2026）の学習支援エージェントです。
# - 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
# - 回の内容を聞かれたら get_lecture_summary を使う。
# - 用語集に見つからないことは、推測で断定せず「用語集にはない」と伝える。
# - 回答の最後に、関係する回（第N回）を示す。
# - 宿題の答えそのものは書かず、考え方のヒントで導く。
# - 最後にsave_learning_log を使用して学習ログを残す。
# """

INSTRUCTIONS = """\
あなたはAIエンジニアリング講座（AIE2026）の学習支援エージェントです。
- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
- 回の内容を聞かれたら get_lecture_summary を使う。
- 確認問題や復習問題を求められたら make_quiz を使う。
- 用語集に見つからないことは、推測で断定せず「用語集にはない」と伝える。
- 回答の最後に、関係する回（第N回）を示す。
- 宿題の答えそのものは書かず、考え方のヒントで導く。
- 最後にsave_learning_log を使用して学習ログを残す。
"""

# モデルは実行時に渡す（テストではAPIを呼ばない TestModel に差し替えるため）
agent = Agent(instructions=INSTRUCTIONS, output_type=[str, DeferredToolRequests])
agent.tool_plain(tools.search_glossary)
agent.tool_plain(tools.get_lecture_summary)
agent.tool_plain(tools.make_quiz)
agent.tool_plain(requires_approval=True)(tools.save_learning_log)

# TODO(第3回 演習): tools.py に make_quiz を作り、ここで登録しよう

# 第5回: 使う MCP サーバーの設定ファイル。Claude Desktop や Cursor と同じ "mcpServers" 形式
MCP_CONFIG = Path(__file__).resolve().parent / "mcp_servers.json"


def load_mcp_servers(config_path: Path = MCP_CONFIG) -> list[MCPToolset]:
    """設定ファイル（mcpServers 形式）を読み、サーバーごとに接続（MCPToolset）を作る。

    - "command" があれば stdio：サーバーをサブプロセスとして起動する
    - "url" があれば Streamable HTTP：起動済みのサーバーに接続する
    ツール名には「サーバー名_」を付ける（例: glossary_search_glossary）。複数のサーバーに
    同じ名前のツールがあっても区別できるようにするため。

    Pydantic AI には同じことをする load_mcp_toolsets() もある。ここで自分で組み立てているのは、
    古い SDK で作られたサーバー（fetch など）につなぐとき、接続方式の自動判定（"auto"）が
    警告を大量に出すため。"legacy" に固定すると新旧どちらのサーバーにも静かにつながる。
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))["mcpServers"]
    toolsets = []
    for name, spec in config.items():
        if "url" in spec:
            transport = spec["url"]
        else:
            transport = StdioTransport(
                spec["command"], spec.get("args", []), env=spec.get("env")
            )
        toolsets.append(MCPToolset(Client(transport, mode="legacy")).prefixed(name))
    return toolsets


def build_mcp_agent(config_path: Path = MCP_CONFIG) -> Agent:
    """第5回: 関数を直接登録する代わりに、MCP サーバーが公開するツールを使うエージェント。INSTRUCTIONS は同じ。"""
    return Agent(instructions=INSTRUCTIONS, toolsets=load_mcp_servers(config_path))


def build_model() -> GoogleModel:
    return GoogleModel(MODEL, provider=GoogleProvider(api_key=get_api_key()))


def print_tools(target: Agent) -> None:
    """モデルに見えているツールの一覧（名前・説明文・引数）を表示する。LLM は呼ばない。

    TestModel（テストで使う偽モデル）にツール一覧を渡し、そこから読み出す。
    MCP サーバーのツールも、第3回の関数ツールも、モデルから見れば同じ形をしている。
    """
    model = TestModel(call_tools=[])
    target.run_sync("ツール一覧", model=model)
    tools_seen = model.last_model_request_parameters.function_tools
    print(f"モデルに見えているツール: {len(tools_seen)} 個")
    for tool in tools_seen:
        print(f"\n[{tool.name}]")
        print("  " + inspect.cleandoc(tool.description or "").replace("\n", "\n  "))
        params = tool.parameters_json_schema.get("properties", {})
        print(f"  引数: {json.dumps(params, ensure_ascii=False)}")


def print_trace(result) -> None:
    """どのツールが、どの引数で呼ばれ、何を返したかを表示する。"""
    print("\n--- trace ---")
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                print(f"[tool call]   {part.tool_name}({part.args})")
            elif isinstance(part, ToolReturnPart):
                print(f"[tool return] {part.tool_name} -> {str(part.content)[:120]}…")
    print(f"[usage] {result.usage}")


def run_with_approval(target, question: str, model):
    # usageの累積用
    total_requests = 0
    total_tool_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0

    result = target.run_sync(question, model=model)

    while True:
        # 今回のrun_syncで発生したusageを加算
        usage = result.usage
        total_requests += usage.requests
        total_tool_calls += usage.tool_calls
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens

        if usage.cost is not None:
            total_cost += usage.cost

        print(
            f"[DEBUG usage] requests={usage.requests}, "
            f"tool_calls={usage.tool_calls}, "
            f"input_tokens={usage.input_tokens}, "
            f"output_tokens={usage.output_tokens}"
        )

        # 承認待ちがなければ終了
        if not isinstance(result.output, DeferredToolRequests):
            break

        approvals = {}

        # 承認待ちのツール呼び出しを取得
        for call in result.output.approvals:
            print("\n学習記録の保存要求があります。")
            print(f"ツール: {call.tool_name}")
            print(f"引数: {call.args}")

            answer = input("保存を承認しますか？ (y/N): ").strip().lower()

            if answer == "y":
                approvals[call.tool_call_id] = True
            else:
                approvals[call.tool_call_id] = ToolDenied(
                    "利用者が実行を拒否しました。"
                )

        print("[DEBUG] approvals:", approvals)

        result = target.run_sync(
            "承認結果を確認して、処理を続けてください。",
            model=model,
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals=approvals),
        )

    # 累積usageを別の属性として保存せず、戻り値をタプルにする
    total_usage = {
        "requests": total_requests,
        "tool_calls": total_tool_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "cost": total_cost,
    }

    return result, total_usage


def is_retryable_error(error: Exception) -> bool:
    """一時的な通信エラーかどうかを判定する。"""

    # HTTPエラーの場合
    if isinstance(error, ModelHTTPError):
        return error.status_code in (408, 429, 500, 502, 503, 504)

    # 例外が別の例外に包まれている場合も調べる
    current = error
    while current is not None:
        error_name = type(current).__name__.lower()

        if any(
            word in error_name
            for word in (
                "timeout",
                "connection",
                "network",
                "connecterror",
                "protocolerror",
            )
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


def run_with_retry(target, question: str, model):
    """一時的な通信エラーが発生した場合に再試行する。"""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result, total_usage = run_with_approval(target, question, model)
            return result, total_usage

        except Exception as e:
            # 再試行対象外のエラーなら、そのまま送出
            if not is_retryable_error(e):
                raise

            # 最大試行回数に達した場合
            if attempt >= MAX_ATTEMPTS:
                print(
                    f"\n[ERROR] {MAX_ATTEMPTS}回試行しましたが、" "処理に失敗しました。"
                )
                raise

            print(
                f"\n[RETRY] 通信エラーが発生しました。"
                f"{RETRY_WAIT_SECONDS}秒後に再試行します。"
                f"（{attempt}/{MAX_ATTEMPTS}回目）"
            )

            time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError("再試行処理が予期せず終了しました。")


def save_run_log(question: str, result, total_usage) -> None:
    """モデルへの入力・出力・TRACEをJSONL形式で保存する。"""

    log_path = tools.DATA_DIR / "agent_runs.jsonl"

    # 実行日時（日本時間）
    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()

    # TRACE情報を取得
    trace = ModelMessagesTypeAdapter.dump_python(
        result.all_messages(),
        mode="json",
    )

    # 回答と使用量を取得
    answer = result.output
    usage = {
        "requests": result.usage.requests,
        "tool_calls": result.usage.tool_calls,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "total_tokens": (result.usage.input_tokens + result.usage.output_tokens),
        "cost": str(result.usage.cost),
        "details": result.usage.details,
    }

    # 1回分の実行記録
    record = {
        "timestamp": timestamp,
        "model": MODEL,
        "question": question,
        "answer": answer,
        "trace": trace,
        "usage": usage,
        "total_usage": total_usage,
    }

    # JSONL形式で追記
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )

    print(f"[log] 実行記録を保存しました: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="学習支援エージェントに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--trace", action="store_true", help="ツール呼び出しを表示する")
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="第5回: ツールを MCP サーバー（app/mcp_servers.json）から受け取る",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="モデルに見えるツール一覧を表示して終了する（APIキー不要）",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        help="モデルとの入出力とTRACEをファイルに保存する",
    )
    args = parser.parse_args()

    target = build_mcp_agent() if args.mcp else agent
    if args.list_tools:
        print_tools(target)
        return
    if not args.question:
        parser.error("質問を指定してください")

    print(f"[model] {MODEL}")
    if args.mcp:
        print("[tools] MCP サーバー（app/mcp_servers.json）")
    try:
        result, total_usage = run_with_retry(target, args.question, build_model())

    except ModelHTTPError as e:
        # # エラーもログに残す
        # if args.save_log:
        #     save_run_log(
        #         question=args.question,
        #         error=e,
        #     )
        print(api_error_message(e.status_code, str(e.body)[:200]))
        raise SystemExit(1)

    # 正常終了時にログ保存
    if args.save_log:
        save_run_log(
            question=args.question,
            result=result,
            total_usage=total_usage,
        )

    print(result.output)
    if args.trace:
        print_trace(result)

    print(f"[total usage] {total_usage}")


if __name__ == "__main__":
    main()
