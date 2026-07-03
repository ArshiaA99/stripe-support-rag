from vectorstore import collection

def retrieve_context(question):
  results = collection.query(
    query_texts=[question],
    n_results=3,
    include=["documents","distances","metadatas"]
  )

  context = "\n".join(
    results["documents"][0]
  )

  return context