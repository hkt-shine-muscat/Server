from transformers import TFAutoModelForCausalLM
from .preprocessing import Preprocesser

p = Preprocesser()

model = TFAutoModelForCausalLM.from_pretrained(p.PREMODEL_NAME, from_pt=True)


def use_model(text: str):
    text = p.encoding(text)
    output = model.generate(text,
                            max_length=1000,
                            num_beams=5,
                            top_k=20,
                            no_repeat_ngram_size=4,
                            length_penalty=0.65,
                            repetition_penalty=2.)
    return p.decoding(output[0][text.shape[-1]:])
