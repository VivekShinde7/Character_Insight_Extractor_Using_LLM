import sys
import json
import os
from src.utils import setup_vector_database, run_llm_chain

def get_character_info(character_name):
    """
    Retrieve structured details about a character using embeddings and LLMs.
    """
    # Set up the vector database
    vector_db = setup_vector_database()
    search_results = vector_db.similarity_search(character_name, k=3)

    if not search_results:
        print(f"Character '{character_name}' not found.")
        return None

    # Combine the retrieved chunks for more context
    story_chunk = " ".join(result.page_content for result in search_results if character_name.lower() in result.page_content.lower())
    if not story_chunk:
        print(f"Character '{character_name}' not found in the retrieved chunks.")
        return None

    # Extract the story title from the first search result
    story_title = search_results[0].metadata.get("story_title", "Unknown")

    # Extract details using LLMs
    print(f"Processing summary for character: {character_name}")
    summary = run_llm_chain({"story_chunk": story_chunk, "character_name": character_name}, "summary")

    print(f"Processing relationships for character: {character_name}")
    relationships = run_llm_chain({"story_chunk": story_chunk, "character_name": character_name}, "relationships")

    print(f"Processing character type for character: {character_name}")
    character_type = run_llm_chain({"story_chunk": story_chunk, "character_name": character_name}, "character_type")

    # Handle invalid or empty summary
    if not summary:
        summary = "No summary available."
    else:
        summary = summary.strip()

    # Handle invalid JSON output for relationships
    if relationships:
        try:
            relationships_parsed = json.loads(relationships)
            # Filter relationships to exclude self-referential entries
            relationships_parsed = [
                rel for rel in relationships_parsed
                if character_name.lower() not in rel["name"].lower()
            ]
        except json.JSONDecodeError:
            print("Failed to parse relationships JSON. Defaulting to an empty list.")
            relationships_parsed = []
    else:
        relationships_parsed = []

    # Ensure character type is one word and valid
    valid_character_types = {"Protagonist", "Antagonist", "Side"}
    if character_type:
        character_type = character_type.split()[0].capitalize()
        if character_type not in valid_character_types:
            character_type = "Unknown"
    else:
        character_type = "Unknown"

    # Construct the response
    response = {
        "name": character_name,
        "storyTitle": story_title,
        "summary": summary,
        "relations": relationships_parsed,
        "characterType": character_type,
    }

    # Print and return the response
    print(json.dumps(response, indent=4))

    # Save the response as a JSON file in the output folder
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
    output_file = os.path.join(output_folder, f"{character_name.replace(' ', '_')}.json")
    
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(response, json_file, indent=4)

    print(f"\nOutput saved to {output_file}")
    return response

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_character_info.py <character_name>")
    else:
        get_character_info(sys.argv[1])
