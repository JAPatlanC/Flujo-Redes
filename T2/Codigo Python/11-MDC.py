import networkx as nx
import matplotlib.pyplot as plt


#Creamos una instancia de un multigrafo
G = nx.MultiDiGraph()

#Agregamos las aristas en el nodo (junto con los nodos)
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(2,5)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(4,1)

#Dibujamos el grafo
nx.draw(G, pos=nx.spectral_layout(G), nodecolor='r', edge_color=['b','b','r','b','b','b'], with_labels=True)
#Mostramos el grafo en pantalla
plt.show()