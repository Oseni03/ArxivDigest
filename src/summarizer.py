from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

llm = ChatOpenAI(temperature=0)


def summarize(papers_contents):
    """
        Args:
            papers_contents: a list of dict
                eg, [
                    {"title": "title of research paper1", "content": "content-of-research-paper1"},
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
    for paper in papers_contents:
        paper["summary"] = map_chain.run(paper["content"])
    return papers_contents