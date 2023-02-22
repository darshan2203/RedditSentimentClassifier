import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

import re
import collections
import contractions
import html

app = FastAPI()


class ModelInference:
    def __init__(self, args):
        if "model_dir" not in args:
            raise Exception("Please pass the valid arguments to load the model.")
        self.model_checkpoint = args["model_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint)
        self.id2label = self.model.config.id2label

    def predict(self, message):
        message = self.clean_text(msg=message)
        inputs = self.tokenizer(message, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs['logits'], dim=1)
            prediction = probabilities.detach().numpy().tolist()
        print("Message - {} Prediction - {}".format(message, prediction))
        return {self.id2label[idx]: x for resp in prediction for idx, x in enumerate(resp)}

    def clean_text(self, msg):
        def lower_case(text):
            return text.lower()

        def remove_urls(text):
            return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)

        def fix_html(x: str) -> str:
            "List of replacements from html strings in `x`."
            re1 = re.compile(r'  +')
            x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
                'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
                '<br />', "\n").replace('\\"', '"').replace('<unk>', "UNK").replace(' @.@ ', '.').replace(
                ' @-@ ', '-').replace(' @,@ ', ',').replace('\\', ' \\ ')
            return re1.sub(' ', html.unescape(x))

        def remove_extra_lines(text):
            return re.sub(r'\n|\r|\t', ' ', text)

        def replace_rep(t: str) -> str:
            "Replace repetitions at the character level in `t`."

            def _replace_rep(m: collections.Collection[str]) -> str:
                c, cc = m.groups()
                return f' {"TK_REP"} {len(cc) + 1} {c} '

            re_rep = re.compile(r'(\S)(\1{3,})')
            return re_rep.sub(_replace_rep, t)

        def expand_contractions(con_text):
            con_text = contractions.fix(con_text)
            return con_text

        msg = lower_case(msg)
        msg = remove_urls(msg)
        msg = fix_html(msg)
        msg = remove_extra_lines(msg)
        msg = replace_rep(msg)
        msg = expand_contractions(msg)
        return msg

class SimpleMessage(BaseModel):
    text: Optional[str] = 'test'


model_args = {
    "model_dir": "model/checkpoint-2400"
}

model_class = ModelInference(model_args)


@app.post("/prediction")
async def run_prediction(message: SimpleMessage):
    prediction = model_class.predict(message.text)
    return {'prediction': prediction}
