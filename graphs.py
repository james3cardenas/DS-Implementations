"""Python implementation of a Graph data structure."""


import collections


class Graph:
	"""An adjacency list application of a graph."""

    ROOT = 0 # Default root node.
	
    def __init__(self):
        """Initializer for an adjacency list graph instance."""
        self.edges    = {}  # Adjacency info.
        self.degree   = collections.defaultdict(int) # Out degree of each vertex
        self.numVertices = 0  # Number of vertices
        self.numEdges    = 0  # Number of edges
        self.directed    = False  # Graph is directed or undirected

    def insertEdge(self, x, y):
        """Insert an EdgeNode (x, y) into the graph.
        Arguments:
            x: int, vertex X connected to vertex Y by edge (x, y).
            y: int, vertex Y connected to vertex X by edge (x, y).
        """
        self._insertEdge(x, y, self.directed)

    def _insertEdge(self, x, y, directed):
        """Inserts an EdgeNode Y into X's adjacency list. If the graph is
           implemented as a directed graph then two copies of the EdgeNode
           will be inserted into the graph, i.e. an EdgeNode X will be inserted
           into Y's adjacency list as well.
        Arguments:
            x: int, vertex X whose adjacency list to append EdgeNode Y to.
            y: int, vertex Y of EdgeNode is to append to X's adjacency list.
        """
        edgeNode        = EdgeNode()
        edgeNode.weight = None
        edgeNode.y      = y
        edgeNode.next   = self.edges.get(x)

        self.edges[x]   = edgeNode
        self.degree[x] += 1

        if not directed:
            # An undirected graph requires two edges, one EdgeNode in 
            # X's adjacency list and one EdgeNode in Y's adjacency list. 
            self._insertEdge(y, x, True)
        else:
            self.numEdges += 1


class EdgeNode:
    """An adjacency list graph node representative of a directed edge."""

    def __init__(self):
        """Initializer for an EdgeNode instance."""
        self.next = None
        self.weight = None
        self.y = None  
