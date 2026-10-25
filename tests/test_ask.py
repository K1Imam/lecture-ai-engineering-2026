from app import ask as ask_module


def test_ask_outputs_each_model_and_comparison(monkeypatch, capsys):
    calls = []

    def fake_ask(question, model):
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
