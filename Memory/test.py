from memory_manager import MemoryGemini, FAISS, SQLite, Neo4j

# sq = Neo4j()
# print(sq.get_all_labels())
memory_manager = MemoryGemini()
# print(mg.save("User prefers Thai and Japanese food. Tell me the parameters you used for each database and the results from each database"))
# print(memory_manager.get("What food does the user like?"))
print(memory_manager.save("Remind me to take a walk every evening"))
# print(mg.save("User is building an AI operating system called Bansal using Gemini."))
# print(mg.get("What is Bansal? Tell me the parameters you used to search each database and the results from each database. Then answer the question I asked"))
