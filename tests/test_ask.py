from app import ask as ask_module
from google.genai import types


def test_ask_outputs_each_model_and_comparison(monkeypatch, capsys):
    calls = []

    def fake_ask(question, model, history=None):
        calls.append((question, model, history))
        if question.startswith("以下は同じ質問に対する複数モデル"):
            return ask_module.AskResult("回答の比較です。", 120, 40)
        if question.startswith("以下は同じ質問に対する3モデル"):
            return ask_module.AskResult("計測値の考察です。", 150, 50)
        model_index = list(ask_module.MODEL_NAMES).index(model)
        return ask_module.AskResult(f"{model}の回答", 100 + model_index, 30 + model_index)

    monkeypatch.setattr(ask_module, "ask", fake_ask)
    monkeypatch.setattr("sys.argv", ["app.ask", "LoRAとは？"])
    timer_values = iter([0.0, 0.12, 1.0, 1.34, 2.0, 2.56])
    monkeypatch.setattr(ask_module, "perf_counter", lambda: next(timer_values))

    ask_module.main()

    output = capsys.readouterr().out
    assert [model for _, model, _ in calls] == [
        *ask_module.MODEL_NAMES,
        ask_module.MODEL_NAMES[0],
        ask_module.MODEL_NAMES[0],
    ]
    assert "LoRAとは？" in calls[0][0]
    assert "===== 3モデルの比較・考察 =====" in output
    assert "回答の比較です。" in output
    assert "===== 応答時間・トークン数 =====" in output
    assert "===== 応答時間・トークン数と無料枠の考察 =====" in output
    assert "計測値の考察です。" in output
    for model in ask_module.MODEL_NAMES:
        assert f"===== {model} =====" in output
        assert f"{model}の回答" in output
    comparison_prompt = calls[-2][0]
    for model in ask_module.MODEL_NAMES:
        assert f"### {model}\n{model}の回答" in comparison_prompt
    analysis_prompt = calls[-1][0]
    assert "| gemini-3.5-flash-lite | 0.12 | 100 | 30 |" in analysis_prompt
    assert "RPM" in analysis_prompt and "TPM" in analysis_prompt and "RPD" in analysis_prompt
    assert "| gemini-3.1-flash-lite | 0.34 | 101 | 31 |" in output
    assert "| gemini-3.6-flash | 0.56 | 102 | 32 |" in output


def test_interactive_mode_keeps_separate_history_per_model(monkeypatch, capsys):
    calls = []

    def fake_ask(question, model, history=None):
        calls.append((question, model, tuple(history) if history else ()))
        if question.startswith("以下は同じ質問に対する"):
            return ask_module.AskResult("分析です。", None, None)
        return ask_module.AskResult(f"{model}: {question}", 10, 5)

    monkeypatch.setattr(ask_module, "ask", fake_ask)
    monkeypatch.setattr("sys.argv", ["app.ask", "--interactive"])
    questions = iter(["LoRAとは？", "それを短く説明して", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    ask_module.main()

    model_calls = [call for call in calls if not call[0].startswith("以下は同じ質問")]
    assert len(model_calls) == 2 * len(ask_module.MODEL_NAMES)
    for model_index, model in enumerate(ask_module.MODEL_NAMES):
        first_question, first_model, first_history = model_calls[model_index]
        second_question, second_model, second_history = model_calls[
            model_index + len(ask_module.MODEL_NAMES)
        ]
        assert (first_question, first_model, first_history) == ("LoRAとは？", model, ())
        assert second_question == "それを短く説明して"
        assert second_model == model
        assert [(turn.role, turn.parts[0].text) for turn in second_history] == [
            ("user", "LoRAとは？"),
            ("model", f"{model}: LoRAとは？"),
        ]


def test_ask_includes_history_in_gemini_request(monkeypatch):
    request = {}

    class FakeModels:
        def generate_content(self, *, model, contents):
            request["model"] = model
            request["contents"] = contents
            return type(
                "Response",
                (),
                {
                    "text": "続きの回答",
                    "usage_metadata": types.GenerateContentResponseUsageMetadata(
                        prompt_token_count=12,
                        candidates_token_count=3,
                    ),
                },
            )()

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "test-key"
            self.models = FakeModels()

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    history = [
        types.Content(role="user", parts=[types.Part.from_text(text="最初の質問")]),
        types.Content(role="model", parts=[types.Part.from_text(text="最初の回答")]),
    ]

    answer = ask_module.ask("続けて質問", model="test-model", history=history)

    assert answer == ask_module.AskResult("続きの回答", 12, 3)
    assert request["model"] == "test-model"
    contents = request["contents"]
    assert [(turn.role, turn.parts[0].text) for turn in contents] == [
        ("user", "最初の質問"),
        ("model", "最初の回答"),
        ("user", "続けて質問"),
    ]
