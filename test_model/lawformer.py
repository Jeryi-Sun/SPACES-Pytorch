from transformers import AutoModel, AutoTokenizer
from transformers import LongformerForMaskedLM,RobertaForMaskedLM,AutoModelForMaskedLM,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir="/new_disk2/zhongxiang_sun/code/pretrain_model/lawformer/")
model = AutoModel.from_pretrained("thunlp/Lawformer", cache_dir="/new_disk2/zhongxiang_sun/code/pretrain_model/lawformer/")
inputs = tokenizer("任某提起诉讼，请求判令解除婚姻关系并对夫妻共同财产进行分割。", return_tensors="pt")
outputs = model(**inputs)
print(outputs)
print()
