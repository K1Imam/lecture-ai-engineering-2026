from app import ask as ask_module
from google.genai import types


def test_ask_outputs_each_model_and_comparison(monkeypatch, capsys):
    calls = []

    def fake_ask(question, model, history=None):
        calls.append((question, model))
        if question.startswith("以下は同じ質問"):
            return "回答の比較です。"
        return f"{model}の回答"

    monkeypatch.setattr(ask_module, "ask", fake_ask)
    monkeypatch.setattr("sys.argv", ["app.ask", "LoRAとは？"])

    ask_module.main()

    output = capsys.readouterr().out
    assert [model for _, model in calls] == [
        *ask_module.MODEL_NAMES,
        ask_module.MODEL_NAMES[0],
    ]
    assert "LoRAとは？" in calls[0][0]
    assert "===== 3モデルの比較・考察 =====" in output
    assert "回答の比較です。" in output
    for model in ask_module.MODEL_NAMES:
        assert f"===== {model} =====" in output
        assert f"{model}の回答" in output
    comparison_prompt = calls[-1][0]
    for model in ask_module.MODEL_NAMES:
        assert f"### {model}\n{model}の回答" in comparison_prompt


def test_interactive_mode_keeps_separate_history_per_model(monkeypatch, capsys):
    calls = []

    def fake_ask(question, model, history=None):
        calls.append((question, model, tuple(history) if history else ()))
        if question.startswith("以下は同じ質問"):
            return "回答の比較です。"
        return f"{model}: {question}"

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
            return type("Response", (), {"text": "続きの回答", "usage_metadata": None})()

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

    assert answer == "続きの回答"
    assert request["model"] == "test-model"
    contents = request["contents"]
    assert [(turn.role, turn.parts[0].text) for turn in contents] == [
        ("user", "最初の質問"),
        ("model", "最初の回答"),
        ("user", "続けて質問"),
    ]
