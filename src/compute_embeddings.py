import sys
from src.utils import load_stories, setup_vector_database, split_story_text

def compute_embeddings(data_path):
    """
    Compute embeddings for stories using MistralAI and store them in a vector database.
    """
    stories = load_stories(data_path)
    vector_db = setup_vector_database()

    for story_title, story_text in stories:
        chunks = split_story_text(story_text)
        metadata = [{"story_title": story_title} for _ in chunks]
        vector_db.add_texts(chunks, metadatas=metadata)

    print("Embeddings have been computed and saved.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_embeddings.py <data_path>")
    else:
        compute_embeddings(sys.argv[1])
