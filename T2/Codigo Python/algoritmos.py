22#Distintos algoritmos para trazar los grafos
nx.draw(G, pos=nx.spectral_layout(G), node_color='r', edge_color='b', with_labels=True)
nx.draw(G, pos=nx.circular_layout(G), node_color='r', edge_color='b', with_labels=True)
nx.draw(G, pos=nx.random_layout(G), node_color='r', edge_color='b', with_labels=True)
nx.draw(G, pos=nx.shell_layout(G), node_color='r', edge_color='b', with_labels=True)
nx.draw(G, pos=nx.spring_layout(G), node_color='r', edge_color='b', with_labels=True)