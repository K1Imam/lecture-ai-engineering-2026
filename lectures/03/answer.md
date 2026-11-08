# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
  　きちんとツールを使用して回答することから、正しい回答を得ることが可能
- `--trace` で見えたツール呼び出し（ツール名と引数）：
  "AIE2026の講座にある「3レベル」や「lecNN-start」とは何ですか？"
  という質問をしたところ、きちんと次のような呼び出しを行っていた
  [tool call] search_glossary({'query': '3レベル'})
  [tool call] search_glossary({'query': 'lecNN-start'})

また、save_learning_logを実装したところ、最後に以下の呼出しも行っていた
[tool call] save_learning_log({'summary': '「3レベル」と「lecNN-start」の用語について、用語集を参照してそれぞれの意味（宿題の難易度別3段階、演習開始用のGitタグ）を確認・説明した。', 'topic': '3レベル・lecNN-start'})

- 1つの質問での LLM 呼び出し回数（`requests`）：

- 追加したツールと、docstring に書いた説明：

## 宿題

- 取り組んだレベル：基礎 / 実装 / 発展
- やったこと：
- 結果：
- 気づき・疑問：
