from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import torch

tokenizers = AutoTokenizer.from_pretrained(
    r'/Users/avinzhang/git/avin-kit/llm_test/test03/model/uer/gpt2-distil-chinese-cluecorpussmall/models--uer--gpt2-distil-chinese-cluecorpussmall/snapshots/c98ef629a1ece266e9d9183add4cbe5d4b99c7d5')
model = AutoModelForCausalLM.from_pretrained(
    r'/Users/avinzhang/git/avin-kit/llm_test/test03/model/uer/gpt2-distil-chinese-cluecorpussmall/models--uer--gpt2-distil-chinese-cluecorpussmall/snapshots/c98ef629a1ece266e9d9183add4cbe5d4b99c7d5')

# 加载自己的训练权重
model.load_state_dict(torch.load(r"param/net.pt"))

# 使用系统自带的 pipeline 工具生成内容
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizers, device="cpu")

print(pipeline("白", max_length=24))
