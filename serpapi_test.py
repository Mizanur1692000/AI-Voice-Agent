from serpapi import GoogleSearch
import json

# Your SerpApi key
api_key = "1fde7aa91405173eb0d31b7e372d8e2bcf5b3e0cc4c674c5a3a43cfd33843ef0"

# Search for a query that often triggers an answer box
params = {
    "engine": "google",
    "q": "Give the series update of Bangladesh vs Netherlands 2025.",
    "api_key": api_key
}

try:
    # Perform the search
    search = GoogleSearch(params)
    results = search.get_dict()

    # Check for the presence of an answer box
    if "answer_box" in results:
        answer_box = results["answer_box"]
        
        # Check for various common keys where the answer might be
        if "answer" in answer_box:
            print(f"Answer found: {answer_box['answer']}")
        elif "result" in answer_box:
            print(f"Result found: {answer_box['result']}")
        elif "snippet" in answer_box:
            print(f"Snippet found: {answer_box['snippet']}")
        else:
            # If a direct answer key isn't found, print the entire answer box JSON for inspection
            print("Found an answer box, but could not parse a common key. Here is the full JSON:")
            print(json.dumps(answer_box, indent=2))
            
    # Fallback if no answer box is found
    elif "organic_results" in results and len(results["organic_results"]) > 0:
        first_result = results["organic_results"][0]
        print("No direct answer box found. Here's the first organic search result:")
        print(f"Title: {first_result['title']}"),
        print(f"Snippet: {first_result['snippet']}")
    else:
        print("No results found for your query.")

except Exception as e:
    print(f"An error occurred: {e}")