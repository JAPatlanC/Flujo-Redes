import networkx as nx
import matplotlib.pyplot as plt


#Creamos una instancia de un grafo
G = nx.Graph()

#Agregamos las aristas en el nodo (junto con los nodos)
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(4,1)
G.add_edge(4,5)

#Dibujamos el grafo
nx.draw(G, pos=nx.circular_layout(G), node_color='r', edge_color='b', with_labels=True)
#Mostramos el grafo en pantalla
plt.show()