import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
import textwrap
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())

# --------------------------------------------------------------
# Load the HuggingFaceHub API token from the .env file
# --------------------------------------------------------------

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# --------------------------------------------------------------
# Load the LLM model from the HuggingFaceHub
# --------------------------------------------------------------

repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
)


# --------------------------------------------------------------
# Load a video transcript from YouTube
# --------------------------------------------------------------

def youtube_summarizer(video_url="https://www.youtube.com/watch?v=riXpu1tHzl0"):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
    docs = text_splitter.split_documents(transcript)

    # --------------------------------------------------------------
    # Summarization with LangChain
    # --------------------------------------------------------------

    # Add map_prompt and combine_prompt to the chain for custom summarization
    chain = load_summarize_chain(falcon_llm, chain_type="map_reduce", verbose=True)
    print(chain.llm_chain.prompt.template)
    print(chain.combine_document_chain.llm_chain.prompt.template)

    # --------------------------------------------------------------
    # Test the Falcon model with text summarization
    # --------------------------------------------------------------

    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(
        output_summary, width=100, break_long_words=False, replace_whitespace=False
    )
    return wrapped_text


def paper_summerizer(user_input):

    template = """
    
    Given a research paper abstract.

    Your goal is to Generate a concise summary for a research paper with the title 'Title Goes Here.' The summary should encapsulate the main objectives, methods employed, key findings, and implications presented in the paper. Ensure the generated summary is informative and reflects the essence of the research, providing a quick and accurate overview for readers.
    
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the research paper title and abstract: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=falcon_llm, prompt=chat_prompt)
    response = chain.run(user_input=user_input)

    return response


def newsletter_generator(user_input):

    template = """
    
    You are professional content marketer.

    Your goal is to Craft an interesting and informative newsletter content based on the research paper provided. 
    The content should capture the essence of the paper, highlighting key objectives, methods, findings, and implications. 
    Tailor the newsletter to engage a diverse audience, making the research accessible and intriguing. 
    Consider incorporating visuals, relevant statistics, and compelling language to captivate readers' attention. 
    Ensure the generated newsletter effectively communicates the significance of the research in a reader-friendly format.
    
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the research paper content: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=falcon_llm, prompt=chat_prompt)
    response = chain.run(user_input=user_input)

    return response



if "__name__"=="__main__":
    research_paper = """
    Truck Parking Usage Prediction with Decomposed Graph Neural Networks

    Truck parking on freight corridors faces various challenges, such as insufficient parking spaces and compliance with Hour-of-Service (HOS) regulations. 
    These constraints often result in unauthorized parking practices, causing safety concerns. To enhance the safety of freight operations, providing accurate parking usage prediction proves to be a cost-effective solution. 
    Despite the existing research demonstrating satisfactory accuracy for predicting individual truck parking site usage, few approaches have been proposed for predicting usage with spatial dependencies of multiple truck parking sites. 
    We present the Regional Temporal Graph Neural Network (RegT-GCN) as a predictive framework for assessing parking usage across the entire state to provide better truck parking information and mitigate unauthorized parking. 
    The framework leverages the topological structures of truck parking site distributions and historical parking data to predict occupancy rates across a state. To achieve this, we introduce a Regional Decomposition approach, which effectively captures the geographical characteristics. We also introduce the spatial module working efficiently with the temporal module. 
    Evaluation results demonstrate that the proposed model surpasses other baseline models, improving the performance by more than $20\\%$ compared with the original model. The proposed model allows truck parking sites' percipience of the topological structures and provides higher performance.
    """

    # print(paper_summerizer(research_paper))

    paper = input("Enter the paper content: ")
    print(newsletter_generator(paper))