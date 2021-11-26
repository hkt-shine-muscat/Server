from transformers import TFAutoModelForCausalLM
from preprocessing import Preprocesser

p = Preprocesser()

model = TFAutoModelForCausalLM.from_pretrained(p.PREMODEL_NAME, from_pt=True)
