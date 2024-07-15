from graphviz import Digraph



def trace(root):
    """
    This function traces all nodes and edges of a network starting from a given root node.
    All nodes must be Value-objects as defined in engine.py.

    Args:
        - root: The root node of the graph (which is the loss function is this case)
    
    Returns:
        - nodes: All nodes of the graph as a set.
        - edges: All edges of the graph as a set.
    """

    # create empty set to store the nodes and edges
    nodes, edges = set(), set()

    def build(v):

        # add current node if the node is not seen
        if v not in nodes:
            nodes.add(v)
            
            # iterate over all child nodes
            for child in v._prev:

                # add the edges of the node (line from current node to child node)
                edges.add((child, v))

                # run the function recursivly to traverse through all nodes in the graph
                build(child)
    
    # add all nodes and edges from the graph to the sets
    build(root)

    return nodes, edges



def draw_graph(root):
    """
    This function creates a visualizable graph from a given root node.

    Args:
        - root: The root node of the graph (which is the loss function is this case)
    
    Returns:
        - graph: The graph as graphviz.Digraph object. This object can be visualized easily.
    """

    # create graph object which stores all nodes and edges
    graph = Digraph(name="Loss-Graph", format="svg", graph_attr={"rankdir" : "LR"})

    # trace all nodes and edges starting from the input node
    nodes, edges = trace(root)

    # iterate over all nodes which were found
    for n in nodes:

        # name the node with ints in ascending order
        uid = str(id(n))

        # add current node to the graph
        graph.node(name=uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape="record")

        # 
        if n._op:
            graph.node(name=uid + n._op, label = n._op)
            graph.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        graph.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return graph