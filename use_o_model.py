'''
原始模型测试
'''
from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline

model = GPT2LMHeadModel.from_pretrained(
    r"./model/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")
tokenizer = BertTokenizer.from_pretrained(
    r"./model/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")
# print(model)

text_generator = TextGenerationPipeline(model, tokenizer, device="cpu")

for i in range(2):
    print(text_generator("白", max_length=257, do_sample=True))
