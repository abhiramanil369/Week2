import autogen
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY=os.getenv('GEMINI_FLASH_API_KEY')
config_list=[
    {
        'model':'gemini-flash-model',
        'base_url': 'http://localhost:8000/v1',
        'api_key':API_KEY
        
    }
]

llm_config={
    "seed":42,
    "config_list":config_list,
    "temperature":0
}
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "web",
        "use_docker": False
    },
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

task="""
    Give me a summary of article:https://aws.amazon.com/what-is/retrieval-augmented-generation/
"""

user_proxy.initiate_chat(
    assistant,
    message=task
)
