import chromadb
import uuid

def upsert_message_pair(user_query: str, assistant_response: str, collection):
    """
    Upserts a user query and assistant response pair into a ChromaDB collection.

    Args:
        user_query (str): The user's query.
        assistant_response (str): The assistant's response.
        collection: The ChromaDB collection object.
    """
    # Concatenate the user query and assistant response into a single document.
    document = f"user: {user_query}\nassistant: {assistant_response}"
    
    # Generate a unique ID for the message pair.
    pair_id = str(uuid.uuid4())
    
    # Upsert the document into the collection.
    collection.add(
        ids=[pair_id],
        documents=[document]
    )
    print(f"Upserted message pair with ID: {pair_id}")

def retrieve_all_message_pairs(collection):
    """
    Retrieves all message pairs from a ChromaDB collection.

    Args:
        collection: The ChromaDB collection object.

    Returns:
        list: A list of tuples, where each tuple contains a user query and an assistant response.
    """
    # Retrieve all documents from the collection.
    results = collection.get(include=["documents"])
    
    message_pairs = []
    if results['documents']:
        for doc in results['documents']:
            try:
                # Split the document back into user query and assistant response.
                user_part, assistant_part = doc.split("\nassistant: ", 1)
                user_query = user_part.replace("user: ", "", 1)
                message_pairs.append((user_query, assistant_part))
            except ValueError:
                # Handle cases where the document is not in the expected format.
                print(f"Warning: Could not parse document: {doc}")
                
    return message_pairs

if __name__ == '__main__':
    # --- Example Usage ---
    
    # 1. Initialize ChromaDB client and create/get a collection.
    #    Using an in-memory ephemeral client for this example.
    #    For persistent storage, use: chromadb.PersistentClient(path="/path/to/db")
    client = chromadb.Client()
    collection_name = "chat_history"
    collection = client.get_or_create_collection(name=collection_name)
    
    # 2. Upsert some example message pairs.
    print("--- Upserting message pairs ---")
    upsert_message_pair("What is the capital of France?", "The capital of France is Paris.", collection)
    upsert_message_pair("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world.", collection)
    upsert_message_pair("What is the formula for water?", "The chemical formula for water is H2O.", collection)
    
    # 3. Retrieve all message pairs from the collection.
    print("\n--- Retrieving all message pairs ---")
    chat_history = retrieve_all_message_pairs(collection)
    
    # 4. Print the retrieved chat history.
    if chat_history:
        for i, (user, assistant) in enumerate(chat_history):
            print(f"\n--- Message Pair {i+1} ---")
            print(f"User: {user}")
            print(f"Assistant: {assistant}")
    else:
        print("No chat history found.")

    # 5. Clean up: delete the collection
    client.delete_collection(name=collection_name)
    print(f"\nCollection '{collection_name}' deleted.")
