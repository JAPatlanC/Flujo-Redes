import networkx as nx
import matplotlib.pyplot as plt


#Creamos una instancia de un multigrafo
G = nx.MultiDiGraph()

#Agregamos las aristas en el nodo (junto con los nodos)
G.add_edge(1,2)
G.add_edge(1,1)
G.add_edge(1,4)
G.add_edge(3,4)
G.add_edge(3,4)

#Dibujamos el grafo
nx.draw(G, pos=nx.circular_layout(G), node_color=['b','r','r','r'], edge_color=['b','b','b','b','r'], with_labels=True)
#Mostramos el grafo en pantalla
plt.show()