import networkx as nx
import matplotlib.pyplot as plt


#Creamos una instancia de un grafo
G = nx.MultiGraph()

#Agregamos las aristas en el nodo (junto con los nodos)
G.add_edge(1,2)
G.add_edge(1,2)
G.add_edge(1,1)
G.add_edge(1,1)
G.add_edge(1,4)
G.add_edge(3,4)
G.add_edge(3,5)

#Dibujamos el grafo
nx.draw(G, pos=nx.shell_layout(G,[[1,2],[3,4,5]]), node_color=['b','r','r','r','r'], edge_color=['r','b','b','b','b','b','b'], with_labels=True)
#Mostramos el grafo en pantalla
plt.show()