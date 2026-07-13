from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
import sqlite3, os, faiss, torch
import numpy as np
from typing import List, Dict, Optional
import duckdb, traceback, json
from datetime import datetime
from neo4j import GraphDatabase, basic_auth

class Neo4j:
    def __init__(self, uri="neo4j://localhost:7687", user="neo4j", password="dishubansal", neo4j_bin_path="D:\SPECIAL\Jarvis\Memory\\neo4j-community-2025.05.0\\bin\\neo4j.bat"):
        self.uri = uri
        self.user = user
        self.password = password
        # if neo4j_bin_path:
        #     subprocess.Popen([neo4j_bin_path, "console"], shell=True)  # starts Neo4j desktop/server instance
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))


    def close(self):
        self.driver.close()

    def _now(self):
        return datetime.now().isoformat()

    # Save node with metadata
    def save_node(self, label: str, properties: dict) -> bool:
        """Saves the node with with specified properties in Neo4j db

        Args:
            label: The node name
            properties: Proerpties of the node

        Returns:
            A bool saying True is saved successfully or False if an error occurs
        """
        try:
            properties.update({
                "created_at": self._now(),
                "updated_at": self._now()
            })
            props = ', '.join([f'{k}: ${k}' for k in properties])
            query = f"MERGE (n:{label} {{ name: $name }}) SET n += {{ {props} }}"
            self.run_query(query, properties)
            return True
        except Exception:
            return False

    # Save relationship with optional metadata
    def save_relationship(self, from_node: str, rel: str, to_node: str,
                          from_label: str="Entity", to_label: str="Entity", metadata: dict[str,str] = {}) -> bool:
        """Saves the relationship into Neo4j db

        Args:
            from-node: The From node name
            rel: The relationship name
            to_node: The To node name
            from_label (Optional, Default="Entity"): The Label Name of From node
            to_label (Optional, Default="Entity"): The label Name of To node
            metadata (Optional, Default=None): The metadata to be added on this relationship

        Returns:
            A bool saying True is saved successfully or False if an error occurs
        """
        try:
            metadata = metadata or {}
            metadata.update({
                "created_at": self._now(),
                "updated_at": self._now()
            })
            meta_string = ', '.join([f'{k}: ${k}' for k in metadata])

            query = f"""
            MERGE (a:{from_label} {{name: $from}})
            MERGE (b:{to_label} {{name: $to}})
            MERGE (a)-[r:{rel.upper()}]->(b)
            SET r += {{ {meta_string} }}
            """
            params = {"from": from_node, "to": to_node, **metadata}
            self.run_query(query, params)
            return True
        except Exception:
            return False

    # Update relationship metadata
    def update_relationship_metadata(self, from_node: str, to_node: str, rel: str, updates: dict) -> bool:
        """Updates the relationship in Neo4j db

        Args:
            from-node: The From node name
            rel: The relationship name
            to_node: The To node name
            updates: The metadata values to be updated on this relationship

        Returns:
            A bool saying True is saved successfully or False if an error occurs
        """
        try:
            updates["updated_at"] = self._now()
            update_str = ', '.join([f'r.{k} = ${k}' for k in updates])
            query = f"""
            MATCH (a {{name: $from}})-[r:{rel.upper()}]->(b {{name: $to}})
            SET {update_str}
            """
            self.run_query(query, {"from": from_node, "to": to_node, **updates})
            return True
        except Exception:
            return False

    # Get all relationships from or to a node, optionally filtered
    def get_relationships(self, node_name: str, direction: str="both", rel_type: list[str]=[], label: str="") -> str:
        """Gets the matching relationships using the specified parameters from Neo4j db

        Args:
            node_name: The node name
            direction (Default="both"): The direction to search for. Enum=["in", "out", "both"]
            rel_type (Default=[]): A list of relationship types
            label (Default=""): The label name to search for

        Returns:
            A str list showing all the matches
        """
        assert direction in ("out", "in", "both"), "Invalid direction"

        # Relationship type pattern
        if len(rel_type) > 0:
            if len(rel_type) == 1:
                rel_filter = f":{rel_type}"
            elif len(rel_type) > 1:
                rel_filter = f":{ '|'.join(rel_type) }"
            else:
                raise ValueError("rel_type must be string or list of strings")
        else:
            rel_filter = ""

        # Label clause
        node_clause = f":{label}" if label != "" else ""

        queries = []

        if direction in ("out", "both"):
            queries.append(f"""
            MATCH (a{node_clause} {{name: $name}})-[r{rel_filter}]->(b)
            RETURN a.name AS source, type(r) AS relation, b.name AS target, r AS metadata
            """)

        if direction in ("in", "both"):
            queries.append(f"""
            MATCH (a)-[r{rel_filter}]->(b{node_clause} {{name: $name}})
            RETURN a.name AS source, type(r) AS relation, b.name AS target, r AS metadata
            """)

        query = "\nUNION\n".join(queries)
        result = self.run_query(query, {"name": node_name})
        return str([
            (r["source"], r["relation"], r["target"], dict(r["metadata"]))
            for r in result
        ])

    # Search nodes by label + metadata
    def search_nodes_by_property(self, label: str, property_key: str, value: str):
        """Gets the matching nodes using the specified properties from Neo4j db

        Args:
            label: The label of the node
            property_key: The property to match
            value: the value of the property to match

        Returns:
            A str list showing all the matches
        """
        query = f"MATCH (n:{label}) WHERE n.{property_key} = $value RETURN n.name, n"
        result = self.run_query(query, {"value": value})
        return str([(r["n.name"], dict(r["n"])) for r in result])

    # Delete a node
    def delete_node(self, name, label="Entity"):
        query = f"MATCH (n:{label} {{name: $name}}) DETACH DELETE n"
        self.run_query(query, {"name": name})

    def search_node(self, name: str):
        """Search for a node by name without requiring a label.

        Args:
            name: The name of the node

        Returns:
            A str list showing all the matches
        """
        query = f"""
        MATCH (n) 
        WHERE n.name = $name 
        RETURN labels(n) AS labels, properties(n) AS props
        """
        result = self.run_query(query, {"name":name})
        return [dict(r) for r in result]

    def get_all_labels(self) -> List[str]:
        """Return all node labels in the database.

        Args:
            None

        Returns:
            A str list showing all the matches
        """
        result = self.run_query("CALL db.labels()")
        return [record["label"] for record in result]
    
    # Full custom Cypher access
    def custom_query(self, query: str, params: dict[str, str]={}):
        """Run a custom Cypher query (for Neo4j DB) if none of the other available functions work

        Args:
            query: the cypher query to run
            parameters (Default=None): The parameters

        Returns:
            A str list showing all the matches
        """
        return str(list(self.run_query(query, params or {})))

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]

class TSDB:
    def __init__(self, file: str = "D:/SPECIAL/Jarvis/Memory/tsdb/life.db"):
        self.file = file
        con = duckdb.connect(file)

        # 1. Create table
        con.execute("""
        CREATE TABLE IF NOT EXISTS jarvis_life (
            timestamp TIMESTAMP,
            role TEXT,
            text TEXT
        )
        """)

        con.close()

    def saveData(self, role: str, text: str):
        con = duckdb.connect(self.file)

        con.execute("INSERT INTO jarvis_life VALUES (?, ?, ?)", (
            datetime.now(), role, text
        ))

        con.close()

    def getData(self, start: str, end: str, role: str = "") -> str:
        """Gets the data from a start date to end date (both inclusive) from Time Series Database

        Args:
            start: The start date. ISO format
            end: the end date. ISO Format
            role (Default=""): the specific role to search for. Enum=["user", "jarvis"]

        Returns:
            A str list showing all the, divided by lines
        """
        con = duckdb.connect(self.file)

        if role != "":
            query = """
                SELECT * FROM jarvis_life 
                WHERE timestamp BETWEEN ? AND ? AND role = ? 
                ORDER BY timestamp
            """
            rows = con.execute(query, (datetime.fromisoformat(start), datetime.fromisoformat(end), role)).fetchall()
        else:
            query = """
                SELECT * FROM jarvis_life 
                WHERE timestamp BETWEEN ? AND ? 
                ORDER BY timestamp
            """
            rows = con.execute(query, (datetime.fromisoformat(start), datetime.fromisoformat(end))).fetchall()

        final = ""
        for r in rows:
            final += f"[{r[0]}] [{r[1]}] {r[2]}\n"
        return final

class MemoryGemini:
    def __init__(self):
        self.client = genai.Client(api_key="<YOUR_GEMINI_API_KEY>")
        # faiss = FAISS()
        self.sqlite = SQLite()
        # neo = Neo4j()
        # tsdb = TSDB()
#         self.config = types.GenerateContentConfig(
#             tools=[faiss.save, faiss.get, sqlite.getAllTables, sqlite.getData, sqlite.createTable, sqlite.saveData, neo.update_relationship_metadata, neo.get_relationships, neo.save_node, neo.custom_query, neo.save_relationship, neo.search_nodes_by_property, neo.search_node, neo.get_all_labels],
#             system_instruction="""You are Jarvis, my personal AI assistant with persistent memory.
# Your goal is to understand my requests, manage my information across multiple databases, and provide helpful, contextual responses.
# You have access to the following Databases to interact with my knowledge base:

# ## Time Series Database Tools
# ## Faiss (Vector Similarity Search) Tools
# ## SQLite (Structured Facts, Rules, Reminders, To-Dos) Tools
# ## Neo4j (Relationships and Knowledge Graph)


# # Tool Definitions:
# # Each tool description explains its purpose, parameters, and expected output are already given to you
# # You MUST use these tools to perform actions related to memory and information retrieval.

# ---

# **Your operational guidelines:**

# **Saving Memory**
# 1) Analyze which databases query should be saved in ased on if it is a relationship, semantic memory or a fact/lietral memory. A single query can belong to multiple types. So you can save in all databases it should belong to.
# 2) For all the types that memory qualifies into, Call their 'save' tools to save the memory. Keep the names like table, column, nodex, labels, relationships very generic so they can be reused for other memories too.
# *Note*: If useing Neo4j, Make sure to connect all the nodes with relationships. dont leave just a hanging node.
# 3) Keep calling tools in a loop till the memory is saved appropriately
# *Notes*
# For SQLite - Always check if a table already exists for the query. Save in it. If all the current tables are totally unrelated, then only create new table.
# For Neo4j - Always check if similar memories are available and connect if there are. Dont randomly add memories without checking what already exists. Also Check what labels are available and reuse if there is an exact match. If not, create a new one.


# **Retrieving Memory**
# 1) Analyze what User is looking for. If the Query is very vague like 'We were talking about that thing at that time'. Feel free to clarify with user what they mean.
# 2) Call the 'get' tools of the databases in a loop. Retrieve all the related memory from all the databases.
# 3) Based on the retrieved memory in step 2. Decide if you need to retrieve more memory that will support as the indirect context.
# 4) keep retrieving, till you think the memory context is complete or if tools start returning empty results or unrelated memories. In the latter case, You can tell the user you dont remember it.
# *Notes*
# For FAISS - If the query doesnt match anything, Always try more keyword and variations. Try your best to retrieve memory
# For SQLite - Always start with getAllTables, Decide which table the data caan belong to and then use getData. If no match found, check other similar tables.
# For Neo4j - Try different variations of names/labels/relationships. Always use get_all_labels tool and use only available labels to search

# **MOST IMPORTANT INTRUCTIONS**
# 1) Dont stick to only 1 database when searching/retrieving. Always query all databases to get as much related information as possible.
# 2) When saving a memory, Save it only in the relevant databases. You are allowed to store in all databases only if the query seems complex enough. But try to avoid this data duplication.
# 3) For all databases - Keep looping with variations of names/keywords, Only after trying all the variations or When the tools start returning unrelated data, you can be sure that there is no memory of it.
# 4) For Neo4j, Follow 3 step process - Search for a node, Get all relationships for the node, Traverse to related nodes and repeat this process till you decide its enough context.
# """
#         )  # Pass the function itself

    def save(self, query: str):
        database_decision_prompt = f"""You are an AI data architect responsible for a persistent memory system that preserves history. You must never overwrite old information. Your system uses three databases:

SQLite (Relational Database): For structured facts (like Reminders, Todos, Rules etc.), designed with versioning. Tables have status (current, archived) and start_date/end_date columns to track history.
FAISS (Vector Database): For the semantic meaning of events and conversations. Memories here are an immutable stream of experiences, each with a timestamp.
Neo4j (Graph Database): For modeling relationships between entities as they change over time, using time-stamped relationships.
User's Memory to Save:
{query}

Your Task:
Analyze the memory. Is it new information, or is it an update to existing information? Based on this, decide which database(s) are required to store the memory while preserving the full history. Your output must be a JSON object containing database names as keys and true/false as decision.

Example Scenarios:

Memory: "My new shipping address is 456 Oak Avenue, Toronto." -> Decision: {{"SQLite":true, "Neo4j":true, "FAISS":true}} (This is a change event. The structured fact needs versioning in SQLite, the relationship to a location changes over time in Neo4j, and the event of the change itself is a memory for FAISS.)
Memory: "I just read a fascinating article about quantum entanglement." -> Decision: {{"SQLite":false, "FAISS":true, "Neo4j":true}} (The article's concept is for FAISS; the relationship between 'user' and 'quantum entanglement' is for Neo4j).
Memory: "My new contact lens prescription is -2.75." -> Decision: {{"SQLite":true, "FAISS":false, "Neo4j":false}} (A structured fact that needs versioning).
Now, analyze the provided memory and return your JSON decision.
"""
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=database_decision_prompt,
        )
        print("Decision: " + response.text)
        selection = json.loads(response.text)
        if selection["SQLite"]:
            prompt = f"""You are an SQLite data architect responsible for a persistent memory system that preserves history. You must never overwrite old information. Use th tools to follow these steps.
            Follow these steps when saving a memory:
            Step 1: Get All current Tables of the database.
            Step 2: if There is a relevant table which can contain similar memories, Query the table for matching memories.
            Step 3: If there are matching memories AND the matching memories are actually related to the current memory:
                    Step 3A: Decide if the new memory is an update to an existing memory.
                        If it is an update:
                        Step 3B: store it in a new row
                        Step 3C: Update 'start' Timestamp of new row to current time and set is_current column to true
                        Step 3D: Update 'end' Timestamp of old memory to current time and set is_current column to false
                        Step 3E: You are done here. You have successfully stored the memory.
                        If it is not an update but a completely new memory:
                        Step 3B: store it in a new row
                        Step 3C: Update 'start' Timestamp of new row to current time and set is_current column to true
                        Step 3E: You are done here. You have successfully stored the memory.
            Step 4: If there are no matches OR the matching memories are not related to the current memory:
            step 5: Get All tables again.
            Step 6: Check if there are any other relevant tables (apart from the ones already checked) which can contain similar memories. if yes, go to step 3, if not, Step 7.
            Step 7: Create a new table. Keep the table names and Column names very generic like reminders, todos, rules etc. Focus on reusability. Every table should have 'start' timestamp, 'end' timestamp and is_current column depicting to maintin versioning.
            Step 8: Save the memory in the new table with 'start' timestamp as current time, 'end' timestamp as Null, 'is_current' timestamp as true.

            Now save this memory:
            {query}
"""
            response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[self.sqlite.createTable, self.sqlite.getAllTables, self.sqlite.getData, self.sqlite.saveData, self.sqlite.update_row, self.sqlite.update_table_schema]
            )
        )
            print(response.text)
        # if selection["FAISS"]:
        # if selection["Neo4j"]:

        return response.text

    def get(self, query: str):
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"User Query: {query}",
            config=self.config,
        )

        return response.text