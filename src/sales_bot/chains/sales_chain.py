import os
import sys
from threading import Lock
from typing import Any, List, Optional

from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain, RetrievalQA
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory
from langchain.utilities import SerpAPIWrapper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import get_settings
from langchain.pydantic_v1 import BaseModel, Field
from langchain_model.api2d_model import Api2dLLM
from vectordbs.faissdb import FaissDb
from vectordbs.vectordb import VectorDb


class CustomerQuestion(BaseModel):
    #matching with the key in LLMChain
    query: str = Field()

class SalesChain:
    #For thread safe singleton example see [here](https://refactoring.guru/design-patterns/singleton/python/example#example-1)
    _instance = None
    _lock: Lock = Lock()

    _tools: List[Tool]
    _agent: AgentExecutor

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, /, tools: Optional[List[Tool]] = None, memory: Optional[BaseMemory] = None):
        vectordb = FaissDb()
        llm = Api2dLLM(temperature=0)
        if tools is not None:
            self._tools = tools
        else:
            self._tools = self._default_tools(vectordb, llm)
        #TODO fix here to make memory optional
        #memory = memory if memory is not None else vectordb.createMemory()
        memory = ConversationBufferMemory(memory_key="chat_history")
        self._agent = self._create_agent(memory, self._tools, llm)

    def _default_tools(self, vectordb: VectorDb, llm: LLM) -> List[Tool]:
        web_tool = Tool.from_function(
            #TODO Improve web search by switching to google shop searching with more input params
            func=SerpAPIWrapper(params = {
                "engine": "google",
                "location": "Austin, Texas, United States",
                "google_domain": "google.com",
                "gl": "cn",
                "hl": "zh-cn",
                "tbm": "shop"
            }).run,
            name="Web Search",
            description="""useful for when you didn't find proper answer from VectorDb QA Search \n
            and need to answer questions about product specifications and market price."""
            # coroutine= ... <- you can specify an async method if desired as well
        )
        vectorqa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.8}
            )
        )
        vectorqa_tool = Tool.from_function(
            func=vectorqa_chain.run,
            name="VectorDb QA Search",
            description="Use this first when trying to answer a customer's question", #Emphasize on priority
            args_schema=CustomerQuestion
            # coroutine= ... <- you can specify an async method if desired as well
        )
        return [vectorqa_tool, web_tool]

    def _create_agent(self, memory: BaseMemory, tools: List[Tool], llm: LLM) -> AgentExecutor:
        prefix = """你是一个有礼貌的老练的电器销售"""
        suffix = """Begin!"

        {chat_history}
        客户问题: {input}
        {agent_scratchpad}"""
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain)
        return AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory,
            handle_parsing_errors=True
        )
    
    @property
    def agent(self):
        return self._agent
    
if __name__ == "__main__":
    #TODO Fix "Observation: Invalid or incomplete response" causing infinit looping on ReAct
    text = SalesChain().agent.run(input = "请问A100 GPU 卡多少钱?")
    print(text)