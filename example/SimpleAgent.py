from typing import List
from zigent.agents import ABCAgent, BaseAgent
from zigent.llm.agent_llms import BaseLLM, LLMConfig, LangchainChatModel
from zigent.commons import TaskPackage
from zigent.logging.terminal_logger import AgentLogger
from zigent.actions.BaseAction import BaseAction
from duckduckgo_search import DDGS

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('WOWRAG_API_KEY')
base_url = "http://43.200.7.56:8008/v1"
chat_model = "glm-4-flash"

llm_config = LLMConfig(
    {
        "base_url": base_url,
        "api_key": api_key,
        "llm_name": chat_model,
        "temperature": "0.0",
    }
)

llm = LangchainChatModel(llm_config)
agent_logger = AgentLogger(PROMPT_DEBUG_FLAG=False)


class DuckSearchAction(BaseAction):
    def __init__(self) -> None:
        action_name = "DuckDuckGo_Search"
        action_desc = "Using this action to search online content."
        params_doc = {"query": "the search string. be simple."}
        self.ddgs = DDGS()
        super().__init__(
            action_name=action_name, action_desc=action_desc, params_doc=params_doc,
        )

    def __call__(self, query):
        results = self.ddgs.chat(query)
        return results


class DuckSearchAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        actions: List[BaseAction] = [DuckSearchAction()],
        manager: ABCAgent = None,
        **kwargs
    ):
        name = "duck_search_agent"
        role = "You can answer questions by using duck duck go search content."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=actions,
            manager=manager,
            logger=agent_logger,
        )

def test_search_agent():
    labor_agent = DuckSearchAgent(llm=llm)

    test_task = "what is the found date of microsoft"
    test_task_pack = TaskPackage(instruction=test_task)

    response = labor_agent(test_task_pack)

    print("response:", response)

if __name__ == "__main__":
    test_search_agent()
    search_action = DuckSearchAction()
    results = search_action("什么是agent")
    print(results)