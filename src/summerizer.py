from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

llm = ChatOpenAI(temperature=0)


def summerize(papers_contents):
    """
        Args:
            papers_contents: a list of dict
                eg, [
                    {"content": "content-of-research-paper1"},
                    {"content": "content-of-research-paper2"},
                    {"content": "content-of-research-paper3"},
                    .
                    .
                    .
                ]
    """
    # Map
    map_template = """Given the content of your research paper, please provide a concise summary highlighting the most insightful points and key findings.
    Ensure that the summary captures the essence of the research, covering critical aspects such as methodology, results, and conclusions. 
    Be thorough in presenting the main contributions and discoveries without omitting any crucial insights or key elements.

    {content}
    summary:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    return map_chain.apply(papers_contents)