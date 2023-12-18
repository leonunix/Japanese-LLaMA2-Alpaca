[**🇨🇳中文**](./README.md) | [**🌐English**](./README_EN.md) | [**📖文档/Docs**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki) | [**❓提问/Issues**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/issues) | [**💬讨论/Discussions**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/discussions) | [**⚔️竞技场/Arena**](http://llm-arena.ymcui.com/)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="800"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca-2.svg?color=blue&style=flat-square">
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/ymcui/Chinese-LLaMA-Alpaca-2">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca-2">
    <a href="https://app.codacy.com/gh/ymcui/Chinese-LLaMA-Alpaca-2/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde"/></a>
</p>


本项目基于Meta发布的可商用大模型[Llama-2](https://github.com/facebookresearch/llama)开发，是[中文LLaMA&Alpaca大模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca)的第二期项目，开源了**中文LLaMA-2基座模型和Alpaca-2指令精调大模型**。这些模型**在原版Llama-2的基础上扩充并优化了中文词表**，使用了大规模中文数据进行增量预训练，进一步提升了中文基础语义和指令理解能力，相比一代相关模型获得了显著性能提升。相关模型**支持FlashAttention-2训练**。标准版模型支持4K上下文长度，**长上下文版模型支持16K上下文长度**，并可通过NTK方法最高扩展至24K+上下文长度。

#### 本项目主要内容

- 🚀 针对Llama-2模型扩充了**新版中文词表**，开源了中文LLaMA-2和Alpaca-2大模型
- 🚀 开源了预训练脚本、指令精调脚本，用户可根据需要进一步训练模型
- 🚀 使用个人电脑的CPU/GPU快速在本地进行大模型量化和部署体验
- 🚀 支持[🤗transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [LangChain](https://github.com/hwchase17/langchain), [privateGPT](https://github.com/imartinez/privateGPT), [vLLM](https://github.com/vllm-project/vllm)等LLaMA生态

#### 已开源的模型

- 基座模型：Chinese-LLaMA-2-1.3B, Chinese-LLaMA-2-7B, Chinese-LLaMA-2-13B
- 聊天模型：Chinese-Alpaca-2-1.3B, Chinese-Alpaca-2-7B, Chinese-Alpaca-2-13B
- 长上下文模型：Chinese-LLaMA-2-7B-16K, Chinese-LLaMA-2-13B-16K, Chinese-Alpaca-2-7B-16K, Chinese-Alpaca-2-13B-16K

![](./pics/screencast.gif)

----

[中文LLaMA&Alpaca大模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | [多模态中文LLaMA&Alpaca大模型](https://github.com/airaria/Visual-Chinese-LLaMA-Alpaca) | [多模态VLE](https://github.com/iflytek/VLE) | [中文MiniRBT](https://github.com/iflytek/MiniRBT) | [中文LERT](https://github.com/ymcui/LERT) | [中英文PERT](https://github.com/ymcui/PERT) | [中文MacBERT](https://github.com/ymcui/MacBERT) | [中文ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [中文XLNet](https://github.com/ymcui/Chinese-XLNet) | [中文BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer) | [模型裁剪工具TextPruner](https://github.com/airaria/TextPruner) | [蒸馏裁剪一体化GRAIN](https://github.com/airaria/GRAIN)


## 新闻

**[2023/09/01] 发布长上下文模型Chinese-Alpaca-2-7B-16K和Chinese-Alpaca-2-13B-16K，该模型可直接应用于下游任务，例如privateGPT等。详情查看[📚 v3.1版本发布日志](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v3.1)**

[2023/08/25] 发布长上下文模型Chinese-LLaMA-2-7B-16K和Chinese-LLaMA-2-13B-16K，支持16K上下文，并可通过NTK方法进一步扩展至24K+。详情查看[📚 v3.0版本发布日志](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v3.0)

[2023/08/14] 发布Chinese-LLaMA-2-13B和Chinese-Alpaca-2-13B，添加text-generation-webui/LangChain/privateGPT支持，添加CFG Sampling解码方法等。详情查看[📚 v2.0版本发布日志](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v2.0)

[2023/08/02] 添加FlashAttention-2训练支持，基于vLLM的推理加速支持，提供长回复系统提示语模板等。详情查看[📚 v1.1版本发布日志](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v1.1)

[2023/07/31] 正式发布Chinese-LLaMA-2-7B（基座模型），使用120G中文语料增量训练（与一代Plus系列相同）；进一步通过5M条指令数据精调（相比一代略微增加），得到Chinese-Alpaca-2-7B（指令/chat模型）。详情查看[📚 v1.0版本发布日志](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v1.0)

[2023/07/19] 🚀启动[中文LLaMA-2、Alpaca-2开源大模型项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)


## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| [💁🏻‍♂️模型简介](#模型简介) | 简要介绍本项目相关模型的技术特点 |
| [⏬模型下载](#模型下载)        | 中文LLaMA-2、Alpaca-2大模型下载地址          |
| [💻推理与部署](#推理与部署) | 介绍了如何对模型进行量化并使用个人电脑部署并体验大模型 |
| [💯系统效果](#系统效果) | 介绍了模型在部分任务上的效果    |
| [📝训练与精调](#训练与精调) | 介绍了如何训练和精调中文LLaMA-2、Alpaca-2大模型 |
| [❓常见问题](#常见问题) | 一些常见问题的回复 |


## 模型简介

本项目推出了基于Llama-2的中文LLaMA-2以及Alpaca-2系列模型，相比[一期项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)其主要特点如下：

#### 📖 经过优化的中文词表

- 我们针对一代LLaMA模型的32K词表扩展了中文字词（LLaMA：49953，Alpaca：49954）
- 在本项目中，我们**重新设计了新词表**（大小：60105），进一步提升了中文字词的覆盖程度，同时统一了LLaMA/Alpaca的词表，避免了因混用词表带来的问题，以期进一步提升模型对中文文本的编解码效率

#### ⚡ 基于FlashAttention-2的高效注意力

- [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)是高效注意力机制的一种实现，相比其一代技术具有**更快的速度和更优化的显存占用**
- 当上下文长度更长时，为了避免显存爆炸式的增长，使用此类高效注意力技术尤为重要
- 本项目的所有模型均使用了FlashAttention-2技术进行训练

#### 🚄 基于PI和NTK的超长上下文扩展技术

- 在[一期项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)中，我们实现了[基于NTK的上下文扩展技术](https://github.com/ymcui/Chinese-LLaMA-Alpaca/pull/743)，可在不继续训练模型的情况下支持更长的上下文
- 基于[位置插值PI](https://arxiv.org/abs/2306.15595)和NTK等方法推出了长上下文版模型，支持16K上下文，并可通过NTK方法最高扩展至24K-32K
- 进一步设计了**方便的自适应经验公式**，无需针对不同的上下文长度设置NTK超参，降低了使用难度

#### 🤖 简化的中英双语系统提示语

- 在[一期项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)中，中文Alpaca系列模型使用了[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)的指令模板和系统提示语
- 初步实验发现，Llama-2-Chat系列模型的默认系统提示语未能带来统计显著的性能提升，且其内容过于冗长
- 本项目中的Alpaca-2系列模型简化了系统提示语，同时遵循Llama-2-Chat指令模板，以便更好地适配相关生态

下图展示了本项目以及[一期项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)推出的所有大模型之间的关系。

![](./pics/models.png)



### 模型选择指引

以下是中文LLaMA-2和Alpaca-2模型的对比以及建议使用场景。**如需聊天交互，请选择Alpaca而不是LLaMA。**

| 对比项                | 中文LLaMA-2                                            | 中文Alpaca-2                                                 |
| :-------------------- | :----------------------------------------------------: | :----------------------------------------------------------: |
| 模型类型 | **基座模型** | **指令/Chat模型（类ChatGPT）** |
| 已开源大小 | 1.3B、7B、13B | 1.3B、7B、13B |
| 训练类型     | Causal-LM (CLM)           | 指令精调                                                     |
| 训练方式 | 7B、13B：LoRA + 全量emb/lm-head<br/>1.3B：全量 | 7B、13B：LoRA + 全量emb/lm-head<br/>1.3B：全量 |
| 基于什么模型训练 | [原版Llama-2](https://github.com/facebookresearch/llama)（非chat版） | 中文LLaMA-2 |
| 训练语料 | 无标注通用语料（120G纯文本） | 有标注指令数据（500万条） |
| 词表大小<sup>[1]</sup> | 55,296 | 55,296 |
| 上下文长度<sup>[2]</sup> | 标准版：4K（12K-18K）<br/>长上下文版：16K（24K-32K） | 标准版：4K（12K-18K）<br/>长上下文版：16K（24K-32K） |
| 输入模板              | 不需要                                                 | 需要套用特定模板<sup>[3]</sup>，类似Llama-2-Chat |
| 适用场景            | 文本续写：给定上文，让模型生成下文            | 指令理解：问答、写作、聊天、交互等 |
| 不适用场景          | 指令理解 、多轮聊天等                                  |  文本无限制自由生成                                                       |

> [!NOTE]
> [1] *本项目一代模型和二代模型的词表不同，请勿混用。二代LLaMA和Alpaca的词表相同。*</br>
> [2] *括号内表示基于NTK上下文扩展支持的最大长度。*</br>
> [3] *Alpaca-2采用了Llama-2-chat系列模板（格式相同，提示语不同），而不是一代Alpaca的模板，请勿混用。*</br>
> [4] *不建议单独使用1.3B模型，而是通过投机采样搭配更大的模型（7B、13B）使用。*</br>



## 训练与精调

### 预训练

- 在原版Llama-2的基础上，利用大规模无标注数据进行增量训练，得到Chinese-LLaMA-2系列基座模型
- 训练数据采用了一期项目中Plus版本模型一致的数据，其总量约120G纯文本文件
- 训练代码参考了🤗transformers中的[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)，使用方法见[📖预训练脚本Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/pt_scripts_zh)

### 指令精调

- 在Chinese-LLaMA-2的基础上，利用有标注指令数据进行进一步精调，得到Chinese-Alpaca-2系列模型
- 训练数据采用了一期项目中Pro版本模型使用的指令数据，其总量约500万条指令数据（相比一期略增加）
- 训练代码参考了[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)项目中数据集处理的相关部分，使用方法见[📖指令精调脚本Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh)


## 致谢

本项目主要基于以下开源项目二次开发，在此对相关项目和研究开发人员表示感谢。

- [Llama-2 *by Meta*](https://github.com/facebookresearch/llama)
- [llama.cpp *by @ggerganov*](https://github.com/ggerganov/llama.cpp)
- [FlashAttention-2 by *Dao-AILab*](https://github.com/Dao-AILab/flash-attention)

同时感谢Chinese-LLaMA-Alpaca（一期项目）的contributor以及[关联项目和人员](https://github.com/ymcui/Chinese-LLaMA-Alpaca#致谢)。


## 免责声明

本项目基于由Meta发布的Llama-2模型进行开发，使用过程中请严格遵守Llama-2的开源许可协议。如果涉及使用第三方代码，请务必遵从相关的开源许可协议。模型生成的内容可能会因为计算方法、随机因素以及量化精度损失等影响其准确性，因此，本项目不对模型输出的准确性提供任何保证，也不会对任何因使用相关资源和输出结果产生的损失承担责任。如果将本项目的相关模型用于商业用途，开发者应遵守当地的法律法规，确保模型输出内容的合规性，本项目不对任何由此衍生的产品或服务承担责任。

<details>
<summary><b>局限性声明</b></summary>

虽然本项目中的模型具备一定的中文理解和生成能力，但也存在局限性，包括但不限于：

- 可能会产生不可预测的有害内容以及不符合人类偏好和价值观的内容
- 由于算力和数据问题，相关模型的训练并不充分，中文理解能力有待进一步提升
- 暂时没有在线可互动的demo（注：用户仍然可以自行在本地部署和体验）

</details>


