from neo4j import GraphDatabase

uri = "neo4j://localhost:7687"
username = "neo4j"
password = "Yuuki2Asuna"

driver = GraphDatabase.driver(uri, auth=(username, password))

def create_node(tx):
    tx.run("CREATE (n:Person {name: 'Alice'})")

with driver.session() as session:
    session.write_transaction(create_node)
