from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableMap
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq




def load_stories(data_path):
    """
    Load all stories from the data directory.
    """
    stories = []
    for file_name in os.listdir(data_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(data_path, file_name), "r", encoding="utf-8") as file:
                story_text = file.read()
                story_title = os.path.splitext(file_name)[0]
                stories.append((story_title, story_text))
    return stories

def setup_vector_database(embeddings_folder="embeddings"):
    """
    Set up a local vector database.
    """
    load_dotenv()
    os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=embeddings_folder, embedding_function=embeddings)

def split_story_text(story_text):
    """
    Split the story text for embedding generation.
    """
    splitter = CharacterTextSplitter(separator=". ", chunk_size=200, chunk_overlap=20)
    return splitter.split_text(story_text)

    
def run_llm_chain(inputs, task):
    """
    Run the LLM chain for a specific task with the given inputs.
    :param inputs: A dictionary containing the 'story_chunk' and 'character_name'.
    :param task: The specific task to execute ('summary', 'relationships', 'character_type').
    """
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

    # Define prompt templates
    prompt_templates = {
        "summary": (
            "Provide a concise and focused summary of the character {character_name} mentioned in the following text. "
            "Highlight only key actions or roles performed by {character_name} in the story:\n\n"
            "{story_chunk}"
        ),
        "relationships": (
            "Extract all direct relationships for the character {character_name} mentioned in the following text as a JSON array. "
            "Each relationship should include the name of the related character and their relationship to {character_name}. "
            "Use the following format:\n\n"
            "[\n"
            "    {{ \"name\": \"<Related Character>\", \"relation\": \"<Relation to {character_name}>\" }},\n"
            "    ...\n"
            "]\n\n"
            "Text:\n{story_chunk}"
        ),
        "character_type": (
            "Based on the text provided, determine the character type of {character_name} in one word. "
            "Choose from the following options: Protagonist, Antagonist, Side character:\n\n"
            "{story_chunk}"
        ),
    }

    if task not in prompt_templates:
        raise ValueError(f"Invalid task: {task}")

    # Define the prompt template
    prompt = PromptTemplate(input_variables=["story_chunk", "character_name"], template=prompt_templates[task])

    # Create the chain
    chain = RunnableMap({
        "output": prompt | llm  # Use pipe (`|`) to chain the prompt to the LLM
    })

    try:
        # Pass both 'story_chunk' and 'character_name' to the chain
        result = chain.invoke(inputs)

        # Access the 'content' attribute of the result
        raw_output = result["output"]
        if hasattr(raw_output, "content"):  # Ensure 'content' exists
            return raw_output.content.strip()
        else:
            print(f"Unexpected output format: {raw_output}")
            return None
    except Exception as e:
        print(f"Error running LLM chain: {e}")
        return None
