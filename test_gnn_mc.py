import os
import sys
import zipfile
import sqlite3
import random
import networkx as nx
from co_with_gnns_example.utils_mc import MaxCut

def read_graph_from_file(file_path, complement, comment_char='#'):
    """
    Read a graph from a text file into a NetworkX graph.
    Lines starting with the specified comment character are ignored.

    Parameters:
        file_path (str): Path to the text file containing the graph.
        comment_char (str): Character indicating comment lines.

    Returns:
        nx.Graph: NetworkX graph.
    """
    G = nx.Graph()
    neg = False

    with open(file_path, 'r') as file:
        # Skip comment lines
        for line in file:
            if not line.startswith(comment_char) and line != "":
                break

        # Read the number of nodes and edges
        num_nodes, num_edges = map(int, line.strip().split())

        # Read edges and add them to the graph
        for _ in range(num_edges):
            line = next(file)
            nodes_and_weight = line.strip().split()
            if len(nodes_and_weight) == 3:
                node1, node2, weight = nodes_and_weight
                weight=float(weight)
                if (weight < 0):
                    neg = True
                    break
                G.add_edge(node1, node2, weight=weight)
    
    if (complement):
        G = nx.complement(G)
        print("complement")
    G = nx.relabel.convert_node_labels_to_integers(G)
    nx_graph = nx.OrderedGraph()
    nx_graph.add_nodes_from(sorted(G.nodes()))
    nx_graph.add_edges_from(G.edges(data=True))

    return nx_graph, neg

    # Example usage
    # file_path = 'graph.txt'
    # graph = read_graph_from_file(file_path)
    # print("Nodes:", graph.nodes())
    # print("Edges:", graph.edges(data=True))

# Function to unzip a single file and perform some action
def unzip_and_use(folder_path,file_name,cursor,complement,db_conn):

    # Check if the file name is already in the database
    cursor.execute("SELECT id,filename,seed1,seed2,seed3,seed4,seed5 FROM " + os.path.basename(folder_path) + " WHERE filename=?", (file_name,))
    row = cursor.fetchone()
    if not row:
        cursor.execute("INSERT INTO " + os.path.basename(folder_path) + " (filename) VALUES (?)", (file_name,))
        db_conn.commit()
        cursor.execute("SELECT id,filename,seed1,seed2,seed3,seed4,seed5 FROM " + os.path.basename(folder_path) + " WHERE filename=?", (file_name,))
        row = cursor.fetchone()
    
    graph = None
    neg = False

    for i in range(1,6):
        if not row[i+1]:
            # Process
            if not graph:
                print("reading graph")
                extract_path = "extract/"
                zip_file_path = os.path.join(folder_path,file_name)
                # Create a ZipFile object for the specified ZIP file
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    # Extract all the contents to the same directory
                    zip_ref.extractall(path=extract_path)
                for filename in os.listdir(extract_path):
                    file_path = os.path.join(extract_path, filename)
                    graph, neg = read_graph_from_file(file_path,complement)
                    os.remove(file_path)

            if (neg):
                seed_value = gnn_time = mc_size = -1
            else:
                seed_value = random.randint(1,1000000)
                gnn_time, mc_size = MaxCut(graph,seed_value)
            cursor.execute("UPDATE " + os.path.basename(folder_path) + " SET seed" + str(i) +"=?, time" + str(i) +"=?, mc" + str(i) + "=? WHERE filename=?", (seed_value,gnn_time,mc_size,file_name,))
            db_conn.commit()

    extract_path = folder_path + "/extract"

# Specify the folder containing the ZIP files
# folder_path = 'MQLib/scripts/SomeGraphs'
folder_path = sys.argv[1]
complement = int(sys.argv[2])
db_path = "MC.db"

db_conn = sqlite3.connect(db_path)
try:
    cursor = db_conn.cursor()

    # Create the 'files' table if it doesn't exist
    cursor.execute("CREATE TABLE IF NOT EXISTS " + os.path.basename(folder_path) + " (id INTEGER PRIMARY KEY, filename TEXT,seed1 INTEGER, time1 FLOAT, mc1 INTEGER,seed2 INTEGER, time2 FLOAT, mc2 INTEGER,seed3 INTEGER, time3 FLOAT, mc3 INTEGER, seed4 INTEGER, time4 FLOAT, mc4 INTEGER,seed5 INTEGER, time5 FLOAT, mc5 INTEGER)")

    # Iterate through files in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is a ZIP file
        if filename.endswith('.zip'):
            print(file_path)
            # Unzip, use, and delete the file
            unzip_and_use(folder_path,filename,cursor,complement,db_conn)
finally:
    db_conn.close()

print(f'Unzipped, used, and deleted files in {folder_path}')
