import networkx as nx
import matplotlib.pyplot as plt


#Creamos una instancia de un multigrafo dirigido
G = nx.MultiDiGraph()

#Agregamos las aristas en el nodo (junto con los nodos)
G.add_edge(1,2)
G.add_edge(1,4)
G.add_edge(1,4)
G.add_edge(3,4)
G.add_edge(2,5)

#Dibujamos el grafo
nx.draw(G, pos=nx.spectral_layout(G), nodecolor='r', edge_color=['b','r','b','b','b'], with_labels=True)
#Mostramos el grafo en pantalla
plt.show()