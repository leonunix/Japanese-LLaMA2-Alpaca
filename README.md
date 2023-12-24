# Japanese-LLaMA2-Alpaca

本プロジェクトは、Meta（旧Facebook）からリリースした商業利用可能モデル「[Llama-2](https://github.com/facebookresearch/llama)」をベースにしており、日本語能力を拡張させるためのLLaMA&Alpaca大規模言語モデルの二次学習を行いました。第一段階の成果物として、Japanese-LLaMA-2（基盤モデル）とJapanese-Alpaca-2（指示実行モデル）をオープンソースでデモ公開しました。

#### プロジェクト内容

- Llama-2モデルをベースに日本語能力を拡張させ、日本語LLaMA-2及びAlpaca-2モデルを公開
- 事前学習用スクリプト、ファインチューニング用スクリプトを公開。必要に応じてモデルの二次学習が可能
- ローカルCPU/GPUを利用したモデル実装
- LLaMAエコシステムに対応：[llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [LangChain](https://github.com/hwchase17/langchain), [vLLM](https://github.com/vllm-project/vllm)

#### 配布中のモデル

- 基盤モデル (foundation model)：Japanese-LLaMA-2-13B
- 指示実行モデル (instruction-following model)：Japanese-Alpaca-2-13B

----

## 最新情報

[2023/12/??] Japanese-LLaMA-2-13Bをベースに、ファインチューニングを行ったJapanese-Alpaca-2-13B（指示実行モデル）を公開

[2023/12/20] 事前学習を行ったJapanese-LLaMA-2-13B（基盤モデル）公開

[2023/12/04] 🚀[日本語LLaMA-2、Alpaca-2オープンソースプロジェクト](https://github.com/leonunix/Japanese-LLaMA2-Alpaca)スタート

## 目次

| タイトル                                                          | 説明                                                                   |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [💁🏻‍♂️概要](#概要)                                                  | 本プロジェクトの特徴                                                   |
| [⏬ダウンロード](#ダウンロード)                                    | 配布中のモデルのダウンロード                                           |
| [💻推理と実装](#推理と実装)                                         | トレーニング後の量子化 (quantify)、ローカルCPU/GPUを利用したモデル実装 |
| [📝事前学習とファインチューニング](#事前学習とファインチューニング) | 日本語LLaMA-2、Alpaca-2モデルのトレーニング                            |
| [🙏謝辞](#謝辞)                                                     | 貢献した方々に感謝                                                     |

## 概要

本プロジェクトはLlama-2をベースに日本語LLaMA-2及びAlpaca-2モデルを公開しました。特徴として：

#### 📖 大きいトークンサイズ

- 大きいトークンサイズ（サイズ：60,105）を利用
- また、LLaMAとAlpacaのトークンを統一し、混在使用による問題を回避

#### ⚡ FlashAttention-2

- [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)は、Efficient Attentionメカニズムを実装したもので、第1世代と比較して高速化とメモリ使用量の最適化を実現
- 文脈が長くなると、メモリ使用量の爆増を防ぐため、Efficient Attentionの仕様が不可欠となる
- 本プロジェクトはFlashAttention-2を利用してトレーニングを行った

## ダウンロード

### モデル比較

日本語LLaMA-2及びAlpaca-2モデルの比較及び向いているシーンです。チャットが必要の場合はLLaMAではなくAlpacaを使ってください。

| 項目                         |                           Japanese-LLaMA-2                           |                          Japanese-Alpaca-2                          |
| :--------------------------- | :------------------------------------------------------------------: | :-----------------------------------------------------------------: |
| モデルタイプ                 |                            **基盤モデル**                            |                     **指示実行モデル（Chat）**                      |
| 配布中のモデルサイズ         |                                 13B                                  |                                 13B                                 |
| トレーシング方法             |                           Causal-LM (CLM)                            |                        ファインチューニング                         |
| トレーシングパート           |                          LoRA + emb/lm-head                          |                         LoRA + emb/lm-head                          |
| ベースモデル                 | [公式オリジナル版Llama-2](https://github.com/facebookresearch/llama) |                          Japanese-LLaMA-2                           |
| トレーニングデータ           |           ラベリングなし汎用データ（11Gプレーンテキスト）            |                 ラベリングありデータ（100万セット）                 |
| トークンサイズ               |                                60,105                                |                               60,105                                |
| 文脈サイズ（最大拡張サイズ） |                            4K（12K-18K）                             |                            4K（12K-18K）                            |
| 入力テンプレート             |                                 不要                                 |               Llama-2-Chatシリーズのテンプレート必要                |
| 向いている利用シーン         |              文章の続き：文脈を基に続きのテキストを生成              | プロンプトの理解：Q&A、ライティング、チャット、インタラクションなど |
| 向いていない利用シーン       |                  プロンプトの理解、連続チャットなど                  |                        無制限のテキスト生成                         |

### モデルダウンロード

| モデル                |  モデルタイプ  | ファイルサイズ |                      ダウンロードリング                      |                    GGUF形式ダウンロードリング                     |
| :-------------------- | :------------: | :------------: | :----------------------------------------------------------: | :---------------------------------------------------------------: |
| Japanese-LLaMA-2-13B  |   基盤モデル   |    26.6 GB     | [[🤗HF]](https://huggingface.co/owner203/japanese-llama-2-13b) | [[🤗HF]](https://huggingface.co/owner203/japanese-llama-2-13b-gguf) |
| Japanese-Alpaca-2-13B | 指示実行モデル |    26.6 GB     |                           [[🤗HF]]()                           |                             [[🤗HF]]()                              |

## 推理と実装

本プロジェクトのモデルは主に下記の量子化・推理・実装ツールをサポートしています。

| ツール名                                                                        | 機能                                                  | CPU | GPU | 量子化 | GUI | API | vLLM | 16K | 投機的サンプリング | チュートリアル |
| :------------------------------------------------------------------------------ | ----------------------------------------------------- | :-: | :-: | :----: | :-: | :-: | :--: | :-: | :----------------: | -------------- |
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp)                         | 豊富な量子化オプションと効率的なローカル推論          |  ✅  |  ✅  |   ✅    |  ❌  |  ✅  |  ❌   |  ✅  |         ✅          | [作成中]()     |
| [**OpenAI API Calls**](https://platform.openai.com/docs/api-reference)          | OpenAI APIに互換するサーバー実装                      |  ✅  |  ✅  |   ✅    |  ❌  |  ✅  |  ✅   |  ✅  |         ❌          | [作成中]()     |
| [**text-generation-webui**](https://github.com/oobabooga/text-generation-webui) | モデルをWeb UIとしてデプロイするツール                |  ✅  |  ✅  |   ✅    |  ✅  |  ✅  |  ❌   |  ✅  |         ❌          | [作成中]()     |
| [**LangChain**](https://github.com/hwchase17/langchain)                         | 二次開発に適したLLMアプリケーション開発フレームワーク |  ✅  |  ✅  |   ✅    |  ❌  |  ❌  |  ❌   |  ✅  |         ❌          | [作成中]()     |

## 事前学習とファインチューニング

### 事前学習 (pre-training)

- 公式オリジナル版Llama-2をベースに、大規模ラベリングなしデータを利用してトレーニングを行い、Japanese-LLaMA-2基盤モデルを作成
- ソースコードは🤗transformersの[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)を参考

### ファインチューニング (fine-tuning)

- Japanese-LLaMA-2をベースに、ラベリングありデータを利用してファインチューニングを行い、Japanese-Alpaca-2モデルを作成
- ソースコードは[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)プロジェクトを参考

## 謝辞

本プロジェクトは下記オープンソースプロジェクトをベースに二次開発を行いました。各プロジェクトに参加された方々に御礼申し上げます。

- [Llama-2 *by Meta*](https://github.com/facebookresearch/llama)
- [llama.cpp *by @ggerganov*](https://github.com/ggerganov/llama.cpp)
- [🤗transformers *by Hugging Face*](https://github.com/huggingface/transformers)
- [Stanford Alpaca *by Tatsu Lab*](https://github.com/tatsu-lab/stanford_alpaca)
- [FlashAttention-2 *by Dao-AILab*](https://github.com/Dao-AILab/flash-attention)
- [Chinese-LLaMA-Alpaca-2 *by @ymcui*](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)

本プロジェクトは[**GMOインターネットグループ株式会社**](https://www.gmo.jp/)が運営する[**ConoHa VPS (with NVIDIA H100 GPU)**](https://www.conoha.jp/vps/gpu/)上で開発及びテストを行いました。

## 免責事項

本プロジェクトは、Meta（旧Facebook）からリリースしたLlama-2モデルの利用を前提としています。モデルを利用する際は必ず**Llama-2モデルに関するオープンソースライセンス規約**に従ってください。

その他ソースコードを利用する際は、**それぞれのオープンソースライセンス規約**に従ってください。

配布中のモデルを利用して生成されたコンテンツの正確性は、計算方法、ランダム要素、量子化精度の潜在的な劣化のために変動する可能性があります。ご了承ください。

本プロジェクトは、モデルの出力の正確性に関して**いかなる保証も行いません**。関連リソースの使用及びその結果に起因する**いかなる損失についても、責任を負うことはできません**。

本プロジェクトに関連するモデルが商業目的で使用される場合、開発者は**現地の法律と規制に従い、モデルを利用して生成されたコンテンツの合法性を保証する**義務があります。

本プロジェクトは、モデルを利用して開発された製品やサービスに対し、**いかなる責任を負うことができません**。

<details>

<summary><b>制限事項</b></summary>

本プロジェクトのモデルは一定の日本語の理解及び生成能力が備わっているが、次の制限もあります：

- 予測できない有害なコンテンツや、人間の嗜好や価値観に従わないコンテンツが生成される可能性
- 計算能力とデータの問題により、関連モデルのトレーニングが十分ではなく、日本語理解能力をさらに向上させる必要
- 現在オンラインデモンストレーションを提供していない

</details>
