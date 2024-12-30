import os
from dotenv import load_dotenv
from zigent.llm.agent_llms import BaseLLM, LLMConfig, LangchainChatModel

from typing import List
from zigent.actions.BaseAction import BaseAction
from zigent.agents import ABCAgent, BaseAgent

from zigent.commons import AgentAct, TaskPackage
from zigent.actions import ThinkAct, FinishAct
from zigent.actions.InnerActions import INNER_ACT_KEY
from zigent.agents.agent_utils import AGENT_CALL_ARG_KEY

from zigent.agents import ManagerAgent

load_dotenv()
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

class Philosopher(BaseAgent):
    def __init__(
        self,
        philosopher,
        llm: BaseLLM,
        actions: List[BaseAction] = [], 
        manager: ABCAgent = None,
        **kwargs
    ):
        name = philosopher
        role = f"""You are {philosopher}, the famous educator in history. You are very familiar with {philosopher}'s Book and Thought. Tell your opinion on behalf of {philosopher}."""
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=actions,
            manager=manager,
        )

Confucius = Philosopher(philosopher= "Confucius", llm = llm)
Socrates = Philosopher(philosopher="Socrates", llm = llm)
Aristotle = Philosopher(philosopher="Aristotle", llm = llm)

exp_task = "What do you think the meaning of life?"
exp_task_pack = TaskPackage(instruction=exp_task)

act_1 = AgentAct(
    name=ThinkAct.action_name,
    params={INNER_ACT_KEY: f"""Based on my thought, we are born to live a meaningful life, and it is in living a meaningful life that our existence gains value. Even if a life is brief, if it holds value, it is meaningful. A life without value is merely existence, a mere survival, a walking corpse."""
    },
)
obs_1 = "OK. I have finished my thought, I can pass it to the manager now."

act_2 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "I can summarize my thought now."})
obs_2 = "I finished my task, I think the meaning of life is to pursue value for the whold world."
exp_act_obs = [(act_1, obs_1), (act_2, obs_2)]

Confucius.prompt_gen.add_example(
    task = exp_task_pack, action_chain = exp_act_obs
)
Socrates.prompt_gen.add_example(
    task = exp_task_pack, action_chain = exp_act_obs
)
Aristotle.prompt_gen.add_example(
    task = exp_task_pack, action_chain = exp_act_obs
)

manager_agent_info = {
    "name": "manager_agent",
    "role": "you are managing Confucius, Socrates and Aristotle to discuss on questions. Ask their opinion one by one and summarize their view of point."
}
team = [Confucius, Socrates, Aristotle]
manager_agent = ManagerAgent(name=manager_agent_info["name"], role=manager_agent_info["role"], llm=llm, TeamAgents=team)

exp_task = "What is the meaning of life?"
exp_task_pack = TaskPackage(instruction=exp_task)

act_1 = AgentAct(
    name=ThinkAct.action_name,
    params={INNER_ACT_KEY: f"""I can ask Confucius, Socrates and Aristotle one by one on their thoughts, and then summary the opinion myself."""
    },
)
obs_1 = "OK."

act_2 = AgentAct(
    name=Confucius.name,
    params={AGENT_CALL_ARG_KEY: "What is your opinion on the meaning of life?",
        },
)
obs_2 = """Based on my thought, I think the meaning of life is to pursue value for the whold world."""

act_3 = AgentAct(
    name=ThinkAct.action_name,
    params={INNER_ACT_KEY: f"""I have obtained information from Confucius, I need to collect more information from Socrates."""
    },
)
obs_3 = "OK."

act_4 = AgentAct(
    name=Socrates.name,
    params={AGENT_CALL_ARG_KEY: "What is your opinion on the meaning of life?",
        },
)
obs_4 = """I think the meaning of life is finding happiness."""

act_5 = AgentAct(
    name=ThinkAct.action_name,
    params={INNER_ACT_KEY: f"""I have obtained information from Confucius and Socrates, I can collect more information from Aristotle."""
    },
)
obs_5 = "OK."

act_6 = AgentAct(
    name=Aristotle.name,
    params={AGENT_CALL_ARG_KEY: "What is your opinion on the meaning of life?",
        },
)
obs_6 = """I believe the freedom of spirit is the meaning."""

act_7 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "Their thought on the meaning of life is to pursue value, happiniss and freedom of spirit."})
obs_7 = "Task Completed. The meaning of life is to pursue value, happiness and freedom of spirit."

exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4), (act_5, obs_5), (act_6, obs_6), (act_7, obs_7)]

manager_agent.prompt_gen.add_example(
    task = exp_task_pack, action_chain = exp_act_obs
)

from zigent.commons import AgentAct, TaskPackage

exp_task = "Which came first, the chicken or the egg?"
exp_task_pack = TaskPackage(instruction=exp_task)
manager_agent(exp_task_pack)
