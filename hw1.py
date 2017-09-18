
# coding: utf-8

# # Part 3 Coding: Shortest Path

# # Q8

# In[1]:

import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx


# ## Q8a
# A procedure that produces a graph (represented by a matrix) of N nodes, each pair connected with probability p <br>
# input: number of nodes N, probability p <br>
# output: a graph represented by a matrix, start node, end node

# In[2]:

def generate_graph(N, p):
    graph = (np.random.rand(N, N) < p).astype(int)
    # make the graph undirected/ matrix symmetric
    i_lower = np.tril_indices(N, -1)
    graph[i_lower] = graph.T[i_lower]
    np.fill_diagonal(graph, 0)
    return graph


# In[3]:

graph = generate_graph(10, 0.5)
print (graph)

# Print AdjacencyList representation of graph
def converGraph(graph, N):
    result = [[] for i in range(N)]
    for i in range(N):
        for j in range(i, N):
            if graph[i,j] == 1:
                result[i].append(j)
                result[j].append(i)
    return result

print (converGraph(graph, 10))


# ## Q8b
# Generalized Shortest Path Algorithm: BFS (Breadth First Search) <br>
# Input: a graph represented by a matrix, start node, end node <br>
# Output: list of nodes leading from start node to end node

# In[4]:

def bfs(graph, start, end):
    # maintain a queue of paths
    queue = []
    explored = []
    # push the first path into the queue
    queue.append([start])
    while queue:
        # get the first path from the queue
        path = queue.pop(0)
        node = path[-1]
        if node not in explored:
            explored.append(node)
            if node == end:
                return path
            for adjacent in graph[node]:
                new_path = list(path)
                new_path.append(adjacent)
                queue.append(new_path)


# ## Q8c
# Generate a graph w/ p = 0.1; <br>
# Pick any two nodes and compute their shortest dist 10000 times;<br> 
# 100 sample output is printed;<br> 
# Average distance is 1.9

# In[154]:

N = 1000
p = 0.1
maxIter = 10000
total = 0;
pCount = 100;
graph = generate_graph(N, p)
myGraph = converGraph(graph, N)
numDC = 0;
for i in range(maxIter):
    start,end = random.sample(range(N), 2)
    path = bfs(myGraph, start, end)
    if (path == "infinity"):
        dist = sys.maxint
        numDC = numDC + 1
    else:
        dist = len(path)-1
        total = total + dist
    if pCount > 0:
        print ("(node A " + repr(start) + ", node B "+ repr(end) + "): path length " + repr(dist))
        pCount = pCount - 1

print ("Average distance is " + repr(float(total/(maxIter-numDC))))


# ## Q8d
# Run the shortest-path algorithm on data sets constructed with many values of p<br>
# Numerical data: [3.244, 2.637, 2.388, 2.14, 2.031, 1.901, 1.852, 1.801, 1.752, 1.689, 1.634, 1.585, 1.544, 1.502] <br>
# for p values: [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4, 0.45, 0.5] <br>
# Relation plotted below

# In[5]:

def avgDist(N, p, maxIter):
    total = 0;
    graph = generate_graph(N, p)
    myGraph = converGraph(graph, N)
    numDC = 0;
    for i in range(maxIter):
        start,end = random.sample(range(N), 2)
        path = bfs(myGraph, start, end)
        if (path == "infinity"):
            numDC = numDC + 1
        else:
            dist = len(path) - 1
        total = total + dist
    return float(total/(maxIter-numDC))


# In[168]:

p = [0.01, 0.02, 0.03, 0.04]
for i in range(10):
    p.append(0.05+0.05*i)
distList = []
for pvalue in p:
    distList.append(avgDist(1000, pvalue, 1000))
print (distList)


# In[169]:

print (p)


# In[170]:

plt.plot(p, distList)
plt.xlabel("p values")
plt.ylabel("average Distance")
plt.show()


# ## Q8e
# 
# As p increases, the average distance decreases. Intuitively, the more likely two nodes are connected, the shorter their shortest distance becomes. As p gets closer to 1, the shortest distance should converge to 1 as well because two nodes are highly like to be connected directly. The asymptotic line is "avg Dist = 1". 

# # Q9
# Load data and convert graph into adjacency list representation 

# In[136]:

# Q9
# Load fb data
fb =  np.loadtxt(fname = 'facebook_combined.txt', delimiter = ' ', dtype = 'int')


# In[141]:

# Convert fb data to graph
N = 4039
fb_graph = np.zeros((N, N))
for i in range(fb.shape[0]):
    fb_graph[fb[i,0], fb[i,1]] = 1
    fb_graph[fb[i,1], fb[i,0]] = 1
print (fb_graph)


# In[147]:

myfb = converGraph(fb_graph, N)
print (bfs(myfb, 0, 1))


# ## Q9a
# Below is the code for simulating Q8c on facebook graph; <br>
# 100 sample results are printed; <br>
# Average distance is 3.64

# In[157]:

# Q9a
N = 4039
maxIter = 100
total = 0;
pCount = 100;
numDC = 0;
myfb = converGraph(fb_graph, N)
for i in range(maxIter):
    start,end = random.sample(range(N), 2)
    path = bfs(myfb, start, end)
    if (path == "infinity"):
        dist = sys.maxint
        numDC = numDC + 1
    else:
        dist = len(path)-1
        total = total + dist
    if pCount > 0:
        print ("(node A " + repr(start) + ", node B "+ repr(end) + "): path length " + repr(dist))
        pCount = pCount - 1

print ("Average distance is " + repr(float(total/(maxIter-numDC))))


# ## Q9b
# p = number of edges/ total possible edges = 0.0108

# In[172]:

# Q9b
p = fb.shape[0]/ ((N*(N-1))/2)
print ("Facebook data has p value: " + repr(p))


# ## Q9C
# Average shortest path from Facebook data is about 3.6; <br>
# Average shortest path from constructed data is about 2.6, which is smaller than the actual data. <br>
# The reason may be: in the constructed data, the graph is uniformly generated with probability p. Each person has the same expected number of friends. All are equal. However, in the real world, the distribution of number of friends is not uniform. Some people tend to have more friends, some less. The distribution should be more like gaussian. In the graph view, some nodes in the actual Facebook data is more dense than others. That's why the average shortest path is larger in the real data set.

# In[7]:

# Q9c
avgDist(4039, 0.01081996, 1000)

