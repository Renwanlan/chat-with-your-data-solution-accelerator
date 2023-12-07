from langchain.agents import AgentExecutor, XMLAgent, tool, ZeroShotAgent
from langchain.chat_models import ChatAnthropic

from backend.utilities.tools.KustoQueryTool import KustoQueryTool
from .OrchestratorBase import OrchestratorBase
from typing import List
from ..helpers.LLMHelper import LLMHelper
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from ..parser.OutputParserTool import OutputParserTool
from ..common.Answer import Answer
from ..tools.QuestionAnswerTool import QuestionAnswerTool
from langchain.tools.render import format_tool_to_openai_function
from pydantic import BaseModel, Field
from langchain.schema.output_parser import StrOutputParser
import wikipedia

# Define the input schema
class UserManagerUAR(BaseModel):
    query: str = Field(..., description="A user to fetch UAR data for")

class ContentUAR(BaseModel):
    query: str = Field(..., description="A query to fetch UAR data for")


class FunctionLangChainAgent(OrchestratorBase):
    def __init__(self) -> None:
        super().__init__()   
        llm_helper = LLMHelper()
        self.model = llm_helper.get_llm()
        self.memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

    @tool(args_schema=ContentUAR)
    def search(query: str) -> dict:
        """Search things about current events."""
        print("searching==========",query)
        return "32 degrees"
    
    @tool
    def searchUARContent(query: str) -> str:
        """If the question is about UAR, call this function. Search things about UAR(User Access Resource) events."""
        print("searchingUAR==========",query)
        question_answer_tool = QuestionAnswerTool()
        answer = question_answer_tool.answer_question(query, chat_history=[])
        return str(answer)

    @tool(args_schema=UserManagerUAR)
    def SearchUARKusto(query: str) -> dict:
        """Only if you have the user name, you call this function. Search UAR from a manager's user. And give the UAR data. Including ResourceType, ResourceID. You just need to call once this function. Use the output as UAR data."""
        print("SearchUARKusto==========",query)
        kusto_query = KustoQueryTool()
        answer = kusto_query.query('shanliu', query)
        print('answer=========',answer)
        return answer
    
    @tool
    def search_wikipedia(query: str) -> str:
        """Run Wikipedia search and get page summaries."""
        page_titles = wikipedia.search(query)
        summaries = []
        for page_title in page_titles[: 3]:
            try:
                wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
                summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
            except (
                wikipedia.exceptions.DisambiguationError,
            ):
                pass
        if not summaries:
            return "No good Wikipedia Search Result was found"
        return "\n\n".join(summaries)
        
    tool_list = [searchUARContent, SearchUARKusto, search_wikipedia]

    def convert_intermediate_steps(self,intermediate_steps):
        log = ""
        for action, observation in intermediate_steps:
            log += (
                f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                f"</tool_input><observation>{observation}</observation>"
            )
        print("log=========",log)
        return log

    def convert_tools(self,tools):
        return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    
    def orchestrate(self, user_message: str, chat_history: List[dict], **kwargs: dict) -> dict:
        functions = [format_tool_to_openai_function(f) for f in self.tool_list]
        model = self.model.bind(functions=functions)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are helpful but sassy assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        chain = RunnablePassthrough.assign(
            agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | prompt | model | OpenAIFunctionsAgentOutputParser()

        agent_executor = AgentExecutor(agent=chain, tools=self.tool_list, verbose=True, memory=self.memory)
        answer =agent_executor.invoke({"question": user_message})
        messages = answer['output']
        print("messages=========",messages)
        output_formatter = OutputParserTool()
         # Format the output for the UI        
        messages = output_formatter.parse(question=user_message, answer=answer["output"], source_documents='')
        return messages
        