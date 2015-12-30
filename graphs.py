"""Python implementation of a Graph data structure."""

import collections
import types


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

  def findCycle(self):
    """Determines if an undirected graph contains any cycles. Specifically, if
       there is no back edge in the undirected graph, then all edges in the
       graph must be tree edges, and no cycle exists. Otherwise, any back edge
       going from a Vertex X to an ancestor Vertex Y creates a cycle in the
       graph from Vertex Y to Vertex X.
    Returns:
      cyclePath: a list, containing the path of the first cycle encountered in
                 a depth first search graph traversal.
    """
    cyclePath = []
    if self.directed:
      return cyclePath

    def _processEdge(self, x, y):
      """Temporary _processEdge() override method to process each undirected
         edge exactly once and to determine if any back edges exists and if
         so, what cycle path that the back edge creates in the graph. *Note*:
         processing an undirected edge exactly once is necessary to avoid False
         positive vertex cycles, otherwise two traversals of any single
         undirected edge (ex. (x, y) & (y, x)) would mimic a cycle. This
         functionality can also be simulated by not processing edges for all
         verticies that are not yet discovered in the _DFS implementation.
      Arguments:
        x: int, vertex X of the edge (x, y).
        y: int, vertex Y of the edge (x, y).
      """
      if not self._discoveredVertex[y]:
        print('Process tree edge: (%s, %s)\n' % (x, y))
      elif self._discoveredVertex[y] and not self._processedVertex[y]:
        if self._parents[x] != y:
          print('Process back edge: (%s, %s)\n' % (x, y))
          cyclePath.extend(self._findPath(y, x))
          self._finished = True

    # Temporarily bind nested _processEdge function to override the graph's
    # default implementation in order to process edges and locate cycles in the
    # appropriate manner. Execute a depth first search traversal of the
    # graph utilizing the above defined _processEdge() function to search for
    # cycles. Then rebind the default method after the search has completed.
    boundProcessEdge  = self._processEdge
    self._processEdge = types.MethodType(_processEdge, self)
    self.depthFirstSearch()
    self._processEdge = boundProcessEdge

    return cyclePath

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
    return self._findPath(start, end)

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
      for edgeNode in self.edges[vertexV]:
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
    if self._finished: # Support early termination of the traversal.
      return

    vertexV = start
    self._discoveredVertex[vertexV] = True

    # Iterate through each Vertex's edges and add undiscovered
    # verticies to the recursive stack so they can be explored.
    for edgeNode in self.edges[vertexV]:
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

      if self._finished: # Support early termination of the traversal.
        return

    # Each vertex in the graph should be processed.
    if not self._processedVertex[vertexV]:
      self._processedVertex[vertexV] = True
      self._processVertex(vertexV)

  def _findPath(self, start, end, path=None):
    """Constructs a path from the given end Vertex to the given start Vertex.
       Recursively reconstruct the path by following the chain of ancestors
       from the end Vertex to the start Vertex. *Note*: The parent relation
       dict (_parents) must be populated in a complete graph traversal (or
       populated enough to follow the complete parent relation for the given
       vertices) prior to calling this method, otherwise this method may not
       return a valid path.
    Arguments:
      start: int, Vertex X defining the path starting point.
      end: int, Vertex Y defining the path end point.
      path: list, a collections of verticies in the path from start to end.
    """
    if path is None:
      path = []

    if start == end or end == -1:
      path.append(start) # TODO -> data structure that supports quick prepend.
      return path
    else:
      path.append(end)   # TODO -> data structure that supports quick prepend.
      return self._findPath(start, self._parents.get(end, -1), path)

  def _initializeSearch(self, start):
    """Initializer for graph traversals. Reset processing information
       so new info can be recorded for a graph traversal.
    """
    self._processedVertex  = collections.defaultdict(bool)
    self._discoveredVertex = collections.defaultdict(bool)
    self._parents          = {}
    self._finished         = False

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
  # graph = Graph()
  # graph.insertEdge(1, 2)
  # graph.insertEdge(1, 5)
  # graph.insertEdge(1, 6)
  # graph.insertEdge(2, 3)
  # graph.insertEdge(2, 5)
  # graph.insertEdge(5, 4)
  # graph.insertEdge(4, 3)

  graph = Graph(directed=True)
  graph.insertEdge(1, 2)
  graph.insertEdge(1, 6)
  graph.insertEdge(2, 3)
  graph.insertEdge(3, 4)
  graph.insertEdge(4, 5)
  # graph.insertEdge(5, 1) # Back edge
  graph.insertEdge(5, 2) # Back edge

  return graph
