# RAG Pipeline Demo with LangChain, FAISS & OpenAI

This project demonstrates a Retrieval-Augmented Generation (RAG) workflow:
- Embeds organizational knowledge with LangChain & FAISS
- Answers user queries by retrieving relevant text and augmenting LLM output
- Uses OpenAI's GPT model for answer generation

## Setup

1. Clone the repo and cd into it
2. Set your `OPENAI_API_KEY` environment variable
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run the script:
   `python rag_pipeline.py`

## Files

- rag_pipeline.py -- main script
- data/knowledge_base.txt -- organizational knowledge
- output/sample_rag_results.txt -- sample answers with source context

## Example Output

Q: Who founded Arista Networks?
A: Arista Networks was founded by Andy Bechtolsheim, David Cheriton, and Kenneth Duda.
Context Snippet: ['Arista Networks was founded in 2004 by Andy Bechtolsheim, David Cheriton, and Kenneth Duda.']
