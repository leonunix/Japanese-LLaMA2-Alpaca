import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', required=True, type=str)
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--gpu_id', default="0", type=str)
parser.add_argument('--chain_type', default="refine", type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
file_path = args.file_path
model_path = args.model_path

import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

prompt_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. あなたは助けを惜しまないアシスタントです。\n"
    "<</SYS>>\n\n"
    "请为以下文字写一段摘要:\n{text} [/INST]"
)
refine_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. あなたは助けを惜しまないアシスタントです。\n"
    "<</SYS>>\n\n"
    "既存の要約があります：{existing_answer}\n"
    "今、いくつかのテキストがありますが、必要であればそれらに基づいて既存の要約を洗練させることができます。"
    "\n"
    "{text}\n"
    "\n"
    "もしこのテキストが役に立たない場合は、元の要約に戻してください。最終的な要約を生成してください。"
    " [/INST]"
)


if __name__ == '__main__':
    load_type = torch.float16
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs are available.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, length_function=len)
    with open(file_path) as f:
        text = f.read()
    docs = text_splitter.create_documents([text])

    print("loading LLM...")
    model = HuggingFacePipeline.from_model_id(model_id=model_path,
            task="text-generation",
            device=0,
            pipeline_kwargs={
                "max_new_tokens": 400,
                "do_sample": True,
                "temperature": 0.2,
                "top_k": 40,
                "top_p": 0.9,
                "repetition_penalty": 1.1},
            model_kwargs={
                "torch_dtype" : load_type,
                "low_cpu_mem_usage" : True}
            )

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    REFINE_PROMPT = PromptTemplate(
        template=refine_template,input_variables=["existing_answer", "text"],
    )

    if args.chain_type == "stuff":
        chain = load_summarize_chain(model, chain_type="stuff", prompt=PROMPT)
    elif args.chain_type == "refine":
        chain = load_summarize_chain(model, chain_type="refine", question_prompt=PROMPT, refine_prompt=REFINE_PROMPT)
    print(chain.run(docs))
