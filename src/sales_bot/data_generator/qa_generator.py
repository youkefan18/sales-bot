

from langchain.llms.base import LLM
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic.dataclasses import dataclass
from sales_bot.langchain_model.api2d_model import Api2dLLM


@dataclass
class QAGenerator():
    """
        Generate QA pairs based on domain industry of sales man and few shots.
        For good example shots in electronic device sales, refer to [sales skills](https://zhuanlan.zhihu.com/p/357487465)
    """
    _prompt: BasePromptTemplate = ChatPromptTemplate.from_messages(
        [SystemMessagePromptTemplate.from_template(
            """You are a {role}. \n
            Now you are training freshman in sales in your domain industry and please provide
            {num_qa} of practical sales talk as examples. \n
            Please format question and answer examples as below format: \n
            [Customer Question]
            [Sales Answer]
            """
        ), 
        HumanMessagePromptTemplate.from_template(
            """
                haha
            """)
        ]
    )
    _model: LLM = Api2dLLM()