import networkx as nx
import matplotlib.pyplot as plt


#Creamos una instancia de un multigrafo
G = nx.MultiGraph()

#Agregamos las aristas en el nodo (junto con los nodos)
G.add_edge(1,2)
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(2,3)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(3,5)
G.add_edge(4,1)

#Dibujamos el grafo
nx.draw(G, pos=nx.random_layout(G), nodecolor='r', edge_color=['r','b','b','r','b','g','b','b'], with_labels=True)
#Mostramos el grafo en pantalla
plt.show()