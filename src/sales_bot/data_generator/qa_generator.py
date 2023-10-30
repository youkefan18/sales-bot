

import os
import sys

from langchain.llms.base import LLM
from langchain.prompts import BasePromptTemplate, PromptTemplate
from pydantic import Field
from pydantic.dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_model import Api2dLLM


def promptFactory() -> BasePromptTemplate:
    return PromptTemplate.from_template(
        """You are a {role}. \n
        Now you are training freshman in sales in your domain industry and please provide
        {num_qa} sales talk Q&A examples. \n
        The Q&A examples are about a scenario where {scenario}. \n
        Please format question and answer examples as below format: \n
        Sequence Number, number only.
        [Customer Question]
        [Sales Answer]
        """
    )

def modelFactory() -> LLM:
    return Api2dLLM()
@dataclass
class QAGenerator():
    """
        Generate QA pairs based on domain industry of sales man and few shots.
        For good example shots in electronic device sales, refer to [sales skills](https://zhuanlan.zhihu.com/p/357487465)
    """

    _prompt: BasePromptTemplate = Field(default_factory=promptFactory) 
    _model: LLM = Field(default_factory=modelFactory)

    @property
    def model(self) -> LLM:
        return self._model
    
    @property
    def prompt(self) -> BasePromptTemplate:
        return self._prompt


if __name__ == "__main__":
    from langchain.chains import LLMChain
    qa = QAGenerator()
    chain = LLMChain(llm=qa.model, prompt=qa.prompt)
    text = chain.run(
        role="Senior sales person selling electronic devices",
        num_qa=2,
        scenario="customers are bargaining with sales man"
    )
    print(text)