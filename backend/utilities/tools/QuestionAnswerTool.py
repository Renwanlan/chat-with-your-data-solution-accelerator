from typing import List
from .AnsweringToolBase import AnsweringToolBase

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

from ..helpers.AzureSearchHelper import AzureSearchHelper
from ..helpers.ConfigHelper import ConfigHelper
from ..helpers.LLMHelper import LLMHelper
from ..common.Answer import Answer
from ..common.SourceDocument import SourceDocument
import pdb
from langchain.text_splitter import TokenTextSplitter

class QuestionAnswerTool(AnsweringToolBase):
    def __init__(self) -> None:
        self.name = "QuestionAnswer"
        self.vector_store = AzureSearchHelper().get_vector_store()
        self.verbose = True
    
    def answer_question(self, question: str, chat_history: List[dict], **kwargs: dict):
        config = ConfigHelper.get_active_config_or_default()    
        answering_prompt = PromptTemplate(template=config.prompts.answering_prompt, input_variables=["question", "sources"])
        # print(f"Answering Prompt: {answering_prompt}/n/n")
        
        llm_helper = LLMHelper()
      
        # Retrieve documents as sources
        sources = self.vector_store.similarity_search(query=question, k=1, search_type="hybrid")
        
        # Generate answer from sources
        answer_generator = LLMChain(llm=llm_helper.get_llm(), prompt=answering_prompt, verbose=self.verbose)
        sources_text = "\n\n".join([f"[doc{i+1}]: {source.page_content}" for i, source in enumerate(sources)])
        text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=0)
        texts = text_splitter.split_text(sources_text)
                
        with get_openai_callback() as cb:
            result = answer_generator({'question': question, 'sources': sources_text})
        answer = result["text"]
                    
        # Generate Answer Object
        source_documents = []
        for source in sources:
            source_document = SourceDocument(
                id=source.metadata["id"],
                content=source.page_content,
                title=source.metadata["title"],
                source=source.metadata["filepath"],
                chunk=source.metadata["chunk_id"],
            )
            source_documents.append(source_document)
        
        clean_answer = Answer(question=question,
                              answer=answer,
                              source_documents=source_documents,
                              prompt_tokens=cb.prompt_tokens,
                              completion_tokens=cb.completion_tokens)
        return clean_answer
    