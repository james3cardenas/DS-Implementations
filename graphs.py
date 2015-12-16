"""Python implementation of a Graph data structure."""

import collections


class Graph:
  """An adjacency list application of a graph."""

  ROOT = 1 # Default root node to begin traversal at.

  def __init__(self, directed=False):
    """Initializer for an adjacency list graph instance."""
    self.directed    = directed  # Graph is directed or undirected
    self.edges       = collections.defaultdict(lambda: []) # Adjacency info
    self.degree      = collections.defaultdict(int) # Out-degree of vertices
    self.numVertices = 0  # Number of vertices
    self.numEdges    = 0  # Number of edges

  def breadthFirstSearch(self, start=ROOT):
    """Traverses the n-vertex, m-edge Graph using the breadth-first search
       graph traversal algorithm. [Run time: O(n+m)]
    Arguments:
      start: int, Vertex X defining the traversal starting point.
    """
    self._initializeSearch(start)
    self._BFS(start)

  def depthFirstSearch(self, start=ROOT):
    """Traverses the n-vertex, m-edge Graph using the depth-first search
       graph traversal algorithm. [Run time: O(n+m)]
    Arguments:
      start: int, Vertex X defining the traversal starting point.
    """
    self._initializeSearch(start)
    self._DFS(start)

  def findShortestPathBFS(self, start, end):
    """Finds the shortest path from vertex X to vertex Y. Using the fact
       that verticies are discovered in order of increasing distance from
       the ROOT in breadth first search, the discovered parent relation
       defines a tree that uses the smallest number of edges on any
       root-to-X path in the graph.
    Arguments:
      start: int, Vertex X defining the shortest path starting point.
      end: int, Vertex Y defining the shortest path end point.
    """
    # *NOTE*: The BFS shortest path tree is only useful if BFS is performed
    # with start (Vertex X) as the root. Also the shortest path is only
    # given if the graph is unweighted.
    self.breadthFirstSearch(start)

    shortestPath = [start]
    while start != end and end != -1:
      shortestPath.insert(1, end)
      end = self._parents[end]
    return shortestPath

  def insertEdge(self, x, y):
    """Insert an EdgeNode (x, y) into the graph.
    Arguments:
      x: int, vertex X connected to vertex Y by edge (x, y).
      y: int, vertex Y connected to vertex X by edge (x, y).
    """
    self._insertEdge(x, y, self.directed)

  ##############
  # PRIVATE

  def _BFS(self, start):
    """A basic breadth-first search implementation. During the course of the
       traversal every node and edge in the graph is explored. This search
       defines a tree on the verticies of graph, and the tree defines a
       shortest path from the root to every other node in the tree.
    Arguments:
      start: int, Vertex X defining the root of the BFS search.
    """
    queue = collections.deque()
    self._discoveredVertex[start] = True
    queue.append(start)

    while queue:
      # Each vertex in the graph should be processed.
      # Process the vertex in the front of the queue.
      vertexV = queue.popleft()
      self._processVertex(vertexV)
      self._processedVertex[vertexV] = True

      # Iterate through each Vertex's edges and add undiscovered
      # verticies to the queue so they can be explored.
      vertexEdges = self.edges[vertexV]
      for edgeNode in vertexEdges:
        vertexU = edgeNode.y

        # Each edge in the graph should be processed.
        if not self._processedVertex[vertexU] or self.directed:
          # Undirected Graph: Each edge will be considered exactly
          # twice. Ignore any edge that leads to an already processed
          # vertex since this yields no new info related to the graph.

          # Directed Graph: Each edge will be considered exactly once.
          # Process each edge that is encountered despite it leading
          # to an already processed vertex.
          self._processEdge(vertexV, vertexU)

        # Ignore an edge leading to an already discovered vertex since
        # it's already been added to the queue at this point.
        if not self._discoveredVertex[vertexU]:
          self._discoveredVertex[vertexU] = True
          self._parents[vertexU] = vertexV
          queue.append(vertexU)

  def _DFS(self, start):
    """A basic depth-first search implementation. During the course of the
       traversal every node and edge in the graph is explored. For
       undirected graphs this search defines two edges classes, namely
       'tree edges' and 'back edges'.
    Arguments:
      start: int, Vertex X defining the root of the DFS search.
    """
    vertexV = start
    vertexEdges = self.edges[vertexV]
    self._discoveredVertex[vertexV] = True

    # Iterate through each Vertex's edges and add undiscovered
    # verticies to the recursive stack so they can be explored.
    for edgeNode in vertexEdges:
      vertexU = edgeNode.y

      # Each edge in the graph should be processed.
      if not self._discoveredVertex[vertexU]:
        self._parents[vertexU] = vertexV
        self._processEdge(vertexV, vertexU)
        self._DFS(vertexU)
      elif not self._processedVertex[vertexU] or self.directed:
        # Undirected Graph: Each edge will be considered exactly
        # twice. Ignore any edge that leads to an already processed
        # vertex since this yields no new info related to the graph.

        # Directed Graph: Each edge will be considered exactly once.
        # Process each edge that is encountered despite it leading
        # to an already processed vertex.
        self._processEdge(vertexV, vertexU)

    # Each vertex in the graph should be processed.
    if not self._processedVertex[vertexV]:
      self._processedVertex[vertexV] = True
      self._processVertex(vertexV)

  def _initializeSearch(self, start):
    """Initializer for graph traversals. Reset processing information
       so new info can be recorded for a graph traversal.
    """
    self._processedVertex  = collections.defaultdict(bool)
    self._discoveredVertex = collections.defaultdict(bool)
    self._parents          = {start: -1}

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

    self.edges[x].append(edgeNode)
    self.degree[x] += 1

    if not directed:
      # An undirected graph requires two edges, one EdgeNode in
      # X's adjacency list and one EdgeNode in Y's adjacency list.
      self._insertEdge(y, x, True)
    else:
      self.numEdges += 1

  def _processEdge(self, x, y):
    """Processes the given Edge encountered during a traversal of
       the graph.
    Arguments:
        x: int, vertex X of the edge (x, y).
        y: int, vertex Y of the edge (x, y).
    """
    print('Process edge: (%s, %s)\n' % (x, y))

  def _processVertex(self, x):
    """Processes the given Vertex encountered during a traversal of
       the graph.
    Arguments:
        x: int, vertex X to process.
    """
    print('Process vertex: %s\n' % x)


class EdgeNode:
  """An adjacency list graph node representative of a directed edge."""

  def __init__(self):
    """Initializer for an EdgeNode instance."""
    self.weight = None
    self.y = None


def createGraph():
  # DELETE ME - Convienience function for testing at the moment.
  graph = Graph()
  graph.insertEdge(1, 2)
  graph.insertEdge(1, 5)
  graph.insertEdge(1, 6)
  graph.insertEdge(2, 3)
  graph.insertEdge(2, 5)
  graph.insertEdge(5, 4)
  graph.insertEdge(4, 3)
  return graph
