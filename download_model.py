'''
模型下载
'''
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "uer/gpt2-chinese-cluecorpussmall"
cache_dir = "model/uer/gpt2-chinese-cluecorpussmall"

# model_name = "uer/gpt2-distil-chinese-cluecorpussmall"
# cache_dir = "model/uer/gpt2-distil-chinese-cluecorpussmall"

# model_name = "uer/gpt2-chinese-lyric"
# cache_dir = "model/uer/gpt2-chinese-lyric"

# model_name = "uer/gpt2-chinese-poem"
# cache_dir = "model/uer/gpt2-chinese-poem"

# model_name = "uer/gpt2-chinese-ancient"
# cache_dir = "model/uer/gpt2-chinese-ancient"

# model_name = "uer/gpt2-chinese-couplet"
# cache_dir = "model/uer/gpt2-chinese-couplet"

# model_name = "qgyd2021/chinese_porn_novel"
# cache_dir = "model/qgyd2021/chinese_porn_novel"

print(f"模型分词器下载中...")
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print(f"模型分词器下载到：{cache_dir}")
