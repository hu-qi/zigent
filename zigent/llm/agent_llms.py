from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


from zigent.llm.LLMConfig import LLMConfig
import requests

OPENAI_CHAT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-instruct-0914",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1-preview",
    "o1-preview-2024-09-12"
]
OPENAI_LLM_MODELS = [
    "babbage-002",
    "dall-e-2",
    "dall-e-3",
    "davinci-002",
    "omni-moderation-2024-09-26",
    "omni-moderation-latest",
    "text-ada-001",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
    "text-davinci-003",
    "tts-1",
    "tts-1-1106",
    "tts-1-hd",
    "tts-1-hd-1106",
    "whisper-1"
]

class BaseLLM:
    def __init__(self, llm_config: LLMConfig) -> None:
        self.llm_name = llm_config.llm_name
        self.context_len: int = llm_config.context_len
        self.stop: list = llm_config.stop
        self.max_tokens: int = llm_config.max_tokens
        self.temperature: float = llm_config.temperature
        self.end_of_prompt: str = llm_config.end_of_prompt

    def __call__(self, prompt: str) -> str:
        return self.run(prompt)

    def run(self, prompt: str):
        # return str
        raise NotImplementedError


class OpenAIChatLLM(BaseLLM):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config=llm_config)
        self.client = OpenAI(api_key=llm_config.api_key)

    def run(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


class LangchainLLM(BaseLLM):
    def __init__(self, llm_config: LLMConfig):
        from langchain_openai import OpenAI

        super().__init__(llm_config)
        llm = OpenAI(
            model_name=self.llm_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
        )
        human_template = "{prompt}"
        prompt = PromptTemplate(template=human_template, input_variables=["prompt"])
        self.llm_chain = prompt | llm | StrOutputParser()

    def run(self, prompt: str):
        return self.llm_chain.invoke({"prompt": prompt})


class LangchainChatModel(BaseLLM):
    def __init__(self, llm_config: LLMConfig):
        from langchain_openai import ChatOpenAI

        super().__init__(llm_config)
        llm = ChatOpenAI(
            model_name=self.llm_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
        )
        human_template = "{prompt}"
        prompt = PromptTemplate(template=human_template, input_variables=["prompt"])
        self.llm_chain = prompt | llm | StrOutputParser()


    def run(self, prompt: str):
        return self.llm_chain.invoke({"prompt": prompt})

class APIRequestLLM():
    def __init__(self, base_url: str):
        self.base_url = base_url
    def run(self, prompt: str):
        url = f'{self.base_url}?param={prompt}' 
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            return response.text
        except requests.exceptions.RequestException as e:
            # Handle request exceptions
            print(f"Request failed: {e}")
            return "Request failed, please try again later."
    
def get_llm_backend(llm_config: LLMConfig):
    llm_name = llm_config.llm_name
    llm_provider = llm_config.provider
    if llm_name in OPENAI_CHAT_MODELS:
        return LangchainChatModel(llm_config)
    elif llm_name in OPENAI_LLM_MODELS:
        return LangchainLLM(llm_config)
    elif llm_provider == "chat_model": 
        return LangchainChatModel(base_url=llm_config.base_url)
    elif llm_provider == "api_request_llm": 
        return APIRequestLLM(base_url=llm_config.base_url)
    else:
        return LangchainLLM(llm_config)
    # TODO: add more llm providers and inference APIs but for now we are using langchainLLM as the default
    # Using other LLM providers will require additional setup and configuration
    # We suggest subclass BaseLLM and implement the run method for the specific provider in your own best practices

