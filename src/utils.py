from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableMap
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
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
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=200)
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

    prompt_templates = {
        "summary": (
            "Provide a concise and focused summary of the character {character_name} mentioned in the following text. "
            "Highlight only key actions or roles performed by {character_name} in the story:\n\n"
            "{story_chunk}"
        ),
        "relationships": (
            "Extract all direct relationships for the character {character_name} mentioned in the following text as a JSON array. "
            "Each relationship should include the name of the related character and their relationship to {character_name} as a single word. "
            "Provide only one-word descriptions for the relationship, such as 'friend,' 'colleague,' 'sibling,' or 'observer.' "
            "Do not include explanations or descriptions beyond the one-word relation type.\n\n"
            "Respond with a JSON array in the following format and nothing else:\n\n"
            "[\n"
            "    {{ \"name\": \"<Related Character>\", \"relation\": \"<One-word relation to {character_name}>\" }},\n"
            "    ...\n"
            "]\n\n"
            "Text:\n{story_chunk}"

        ),
        "character_type": (
            "Analyze the role of {character_name} in the following story excerpt. "
            "Classify the character as one of the following:\n"
            "1. Protagonist: The central character in the story. This is the person around whom the narrative revolves, and the reader experiences the story primarily through their perspective. The protagonist drives the plot forward with their goals, decisions, and struggles. They are often the focus of the story’s emotional, moral, or thematic journey. Examples include heroes, main characters, or focal points of the narrative.\n"
            "2. Antagonist: A character or force that actively opposes the protagonist. The antagonist creates challenges, conflicts, or obstacles that the protagonist must face or overcome. They are often responsible for driving the story’s tension and conflict. The antagonist is not necessarily evil or villainous but serves as the primary opposing force in the story.\n"
            "3. Side character: A supporting character who is not the focus of the story. Side characters contribute to the plot indirectly by assisting or interacting with the protagonist or antagonist. They are not central to the narrative and typically do not undergo significant development or drive the main events of the story.\n\n"
            "Using the provided definitions, evaluate {character_name}'s role in the story excerpt. "
            "Consider their importance, actions, focus, and how they interact with other characters and the plot.\n"
            "Respond with only one word: Protagonist, Antagonist, or Side.\n\n"
            "Story excerpt:\n{story_chunk}"

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
