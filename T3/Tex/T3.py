import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats
import networkx.algorithms.approximation.treewidth as tree

#Declaracion de variables
G1 = nx.Graph()
G2 = nx.Graph()
G3 = nx.Graph()
G4 = nx.Graph()
G5 = nx.Graph()
G1.add_weighted_edges_from([(0,2,1.3),(0,1,4.2),(1,1,2.2),(2,1,1.5),(2,3,3.7),(1,0,2.6),(4,0,2.3),(5,2,6.1),(3,5,0.8),(2,4,1.2),(1,3,1.9)])
G2.add_weighted_edges_from([(1,3,1.3),(5,0,4.2),(2,2,2.2),(1,0,1.5),(3,4,3.7),(0,5,2.6),(5,1,2.3),(4,1,6.1),(4,0,0.8),(1,3,1.2),(2,4,1.9)])
G3.add_weighted_edges_from([(2,4,1.3),(4,1,4.2),(3,1,2.2),(0,5,1.5),(4,5,3.7),(5,4,2.6),(0,2,2.3),(3,0,6.1),(5,1,0.8),(0,2,1.2),(3,5,1.9)])
G4.add_weighted_edges_from([(3,5,1.3),(3,0,4.2),(4,2,2.2),(5,4,1.5),(5,0,3.7),(4,3,2.6),(1,3,2.3),(2,5,6.1),(0,2,0.8),(5,1,1.2),(4,0,1.9)])
G5.add_weighted_edges_from([(2,5,4.2),(5,3,2.2),(4,3,1.5),(0,1,3.7),(3,2,2.6),(2,4,2.3),(1,4,6.1),(5,3,0.8),(0,0,1.2),(3,1,1.9)])

G6 = nx.balanced_tree(3,3)
G7 = nx.balanced_tree(3,4)
G8 = nx.balanced_tree(3,5)
G9 = nx.balanced_tree(3,6)
G10 = nx.balanced_tree(3,7)

G11 = nx.balanced_tree(3,3)
G12 = nx.balanced_tree(3,4)
G13 = nx.balanced_tree(3,5)
G14 = nx.balanced_tree(3,6)
G15 = nx.balanced_tree(3,7)

#Nodos de los grafos
a1_nodes = [len(G1.nodes),len(G2.nodes),len(G3.nodes),len(G4.nodes),len(G5.nodes)]
a2_nodes = [len(G6.nodes),len(G7.nodes),len(G8.nodes),len(G9.nodes),len(G10.nodes)]
a3_nodes = [len(G11.nodes),len(G12.nodes),len(G13.nodes),len(G14.nodes),len(G15.nodes)]
a4_nodes = a3_nodes
a5_nodes = a3_nodes

#Aristas de los grafos
a1_edges = [len(G1.edges),len(G2.edges),len(G3.edges),len(G4.edges),len(G5.edges)]
a2_edges = [len(G6.edges),len(G7.edges),len(G8.edges),len(G9.edges),len(G10.edges)]
a3_edges = [len(G11.edges),len(G12.edges),len(G13.edges),len(G14.edges),len(G15.edges)]
a4_edges = a3_edges
a5_edges = a3_edges

a1_mean=0
a1_standard=0
a1_means = []
a1_std = []
a2_mean=0
a2_standard=0
a2_means = []
a2_std = []
a3_mean=0
a3_standard=0
a3_means = []
a3_std = []
a4_mean=0
a4_standard=0
a4_means = []
a4_std = []
a5_mean=0
a5_standard=0
a5_means = []
a5_std = []
a1_times=[]
a2_times=[]
a3_times=[]
a4_times=[]
a5_times=[]
a1_t1_times=[]
a1_t2_times=[]
a1_t3_times=[]
a1_t4_times=[]
a1_t5_times=[]
a2_t1_times=[]
a2_t2_times=[]
a2_t3_times=[]
a2_t4_times=[]
a2_t5_times=[]
a3_t1_times=[]
a3_t2_times=[]
a3_t3_times=[]
a3_t4_times=[]
a3_t5_times=[]
a4_t1_times=[]
a4_t2_times=[]
a4_t3_times=[]
a4_t4_times=[]
a4_t5_times=[]
a5_t1_times=[]
a5_t2_times=[]
a5_t3_times=[]
a5_t4_times=[]
a5_t5_times=[]

# *********************all shortest pats*********************
for j in range(1,31):
    algorithm1_start_time = time.time()
    g1_start_time = time.time()
    for i in range(3500000):
        nx.all_shortest_paths(G1,1,5)
    a1_t1_times.append(time.time() - g1_start_time)

    g2_start_time = time.time()
    for i in range(3500000):
        nx.all_shortest_paths(G2,1,5)
    a1_t2_times.append(time.time() - g2_start_time)

    g3_start_time = time.time()
    for i in range(3500000):
        nx.all_shortest_paths(G3,1,5)
    a1_t3_times.append(time.time() - g3_start_time)

    g4_start_time = time.time()
    for i in range(3500000):
        nx.all_shortest_paths(G4,1,5)
    a1_t4_times.append(time.time() - g4_start_time)

    g5_start_time = time.time()
    for i in range(3500000):
        nx.all_shortest_paths(G5,1,5)
    a1_t5_times.append(time.time() - g5_start_time)
    a1_times.append(time.time() - algorithm1_start_time)
    print("Iteracion: ",j)

a1_mean = np.array(a1_times).mean()
a1_standard = np.array(a1_times).std()
a1_means.append(np.array(a1_t1_times).mean())
a1_means.append(np.array(a1_t2_times).mean())
a1_means.append(np.array(a1_t3_times).mean())
a1_means.append(np.array(a1_t4_times).mean())
a1_means.append(np.array(a1_t5_times).mean())

a1_std.append(np.array(a1_t1_times).std())
a1_std.append(np.array(a1_t2_times).std())
a1_std.append(np.array(a1_t3_times).std())
a1_std.append(np.array(a1_t4_times).std())
a1_std.append(np.array(a1_t5_times).std())
print(a1_means)
print(a1_std)

n, bins, patches = plt.hist(a1_times, 'auto', density=True, facecolor='blue', alpha=0.75)
y = scipy.stats.norm.pdf(bins, a1_mean,a1_standard)
plt.plot(bins, y, 'r--')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Frecuencia')
plt.title('Ruta más corta (Media='+str(round(a1_mean,2))+' ,STD='+str(round(a1_standard,2))+' )',size=18, color='green')
plt.savefig('H1.eps', format='eps', dpi=1000)
plt.show()

#  Boxplots
to_plot=[a1_t1_times,a1_t2_times,a1_t3_times,a1_t4_times,a1_t5_times]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot, showfliers=False)
plt.xlabel('Grafo')
plt.ylabel('Tiempo (segundos)')
plt.title('Ruta más corta')
plt.savefig('BP1.eps', format='eps', dpi=1000)
plt.show()


# *********************dfs tree*********************
for j in range(1,31):
    algorithm2_start_time = time.time()
    g1_start_time = time.time()
    for i in range(300):
        nx.dfs_tree(G6)
    a2_t1_times.append(time.time() - g1_start_time)

    g2_start_time = time.time()
    for i in range(300):
        nx.dfs_tree(G7)
    a2_t2_times.append(time.time() - g2_start_time)

    g3_start_time = time.time()
    for i in range(300):
        nx.dfs_tree(G8)
    a2_t3_times.append(time.time() - g3_start_time)

    g4_start_time = time.time()
    for i in range(300):
        nx.dfs_tree(G9)
    a2_t4_times.append(time.time() - g4_start_time)

    g5_start_time = time.time()
    for i in range(300):
        nx.dfs_tree(G10)
    a2_t5_times.append(time.time() - g5_start_time)
    a2_times.append(time.time() - algorithm2_start_time)
    print("Iteracion: ",j)

a2_mean = np.array(a2_times).mean()
a2_standard = np.array(a2_times).std()
a2_means.append(np.array(a2_t1_times).mean())
a2_means.append(np.array(a2_t2_times).mean())
a2_means.append(np.array(a2_t3_times).mean())
a2_means.append(np.array(a2_t4_times).mean())
a2_means.append(np.array(a2_t5_times).mean())

a2_std.append(np.array(a2_t1_times).std())
a2_std.append(np.array(a2_t2_times).std())
a2_std.append(np.array(a2_t3_times).std())
a2_std.append(np.array(a2_t4_times).std())
a2_std.append(np.array(a2_t5_times).std())
print(a2_means)
print(a2_std)

n, bins, patches = plt.hist(a2_times, 'auto', density=True, facecolor='blue', alpha=0.75)
y = scipy.stats.norm.pdf(bins, a2_mean,a2_standard)
plt.plot(bins, y, 'r--')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Frecuencia')
plt.title('Árbol Búsqueda Profunda (Media='+str(round(a2_mean,2))+' ,STD='+str(round(a2_standard,2))+' )',size=18, color='green')
plt.savefig('H2.eps', format='eps', dpi=1000)
plt.show()

#  Boxplots
to_plot=[a2_t1_times,a2_t2_times,a2_t3_times,a2_t4_times,a2_t5_times]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot, showfliers=False)
plt.xlabel('Grafo')
plt.ylabel('Tiempo (segundos)')
plt.title('Árbol Búsqueda Profunda')
plt.savefig('BP2.eps', format='eps', dpi=1000)
plt.show()

# *********************max_clique*********************
for j in range(1,31):
    algorithm2_start_time = time.time()
    g1_start_time = time.time()
    for i in range(6):
        nx.make_max_clique_graph(G11)
    a3_t1_times.append(time.time() - g1_start_time)

    g2_start_time = time.time()
    for i in range(6):
        nx.make_max_clique_graph(G12)
    a3_t2_times.append(time.time() - g2_start_time)

    g3_start_time = time.time()
    for i in range(6):
        nx.make_max_clique_graph(G13)
    a3_t3_times.append(time.time() - g3_start_time)

    g4_start_time = time.time()
    for i in range(6):
        nx.make_max_clique_graph(G14)
    a3_t4_times.append(time.time() - g4_start_time)

    g5_start_time = time.time()
    for i in range(6):
        nx.make_max_clique_graph(G15)
    a3_t5_times.append(time.time() - g5_start_time)
    a3_times.append(time.time() - algorithm2_start_time)
    print("Iteracion: ",j)

a3_mean = np.array(a3_times).mean()
a3_standard = np.array(a3_times).std()
a3_means.append(np.array(a3_t1_times).mean())
a3_means.append(np.array(a3_t2_times).mean())
a3_means.append(np.array(a3_t3_times).mean())
a3_means.append(np.array(a3_t4_times).mean())
a3_means.append(np.array(a3_t5_times).mean())

a3_std.append(np.array(a3_t1_times).std())
a3_std.append(np.array(a3_t2_times).std())
a3_std.append(np.array(a3_t3_times).std())
a3_std.append(np.array(a3_t4_times).std())
a3_std.append(np.array(a3_t5_times).std())
print(a3_means)
print(a3_std)

n, bins, patches = plt.hist(a3_times, 'auto', density=True, facecolor='blue', alpha=0.75)
y = scipy.stats.norm.pdf(bins, a3_mean,a3_standard)
plt.plot(bins, y, 'r--')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Frecuencia')
plt.title('Problema liga de amigos (Media='+str(round(a3_mean,2))+' ,STD='+str(round(a3_standard,2))+' )',size=18, color='green')
plt.savefig('H3.eps', format='eps', dpi=1000)
plt.show()

#  Boxplots
to_plot=[a3_t1_times,a3_t2_times,a3_t3_times,a3_t4_times,a3_t5_times]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot, showfliers=False)
plt.xlabel('Grafo')
plt.ylabel('Tiempo (segundos)')
plt.title('Problema liga de amigos')
plt.savefig('BP3.eps', format='eps', dpi=1000)
plt.show()

# *********************treewidth_min_degree*********************

for j in range(1,31):
    algorithm2_start_time = time.time()
    g1_start_time = time.time()
    for i in range(20):
        tree.treewidth_min_degree(G11)
    a4_t1_times.append(time.time() - g1_start_time)

    g2_start_time = time.time()
    for i in range(20):
        tree.treewidth_min_degree(G12)
    a4_t2_times.append(time.time() - g2_start_time)

    g3_start_time = time.time()
    for i in range(20):
        tree.treewidth_min_degree(G13)
    a4_t3_times.append(time.time() - g3_start_time)

    g4_start_time = time.time()
    for i in range(20):
        tree.treewidth_min_degree(G14)
    a4_t4_times.append(time.time() - g4_start_time)

    g5_start_time = time.time()
    for i in range(20):
        tree.treewidth_min_degree(G15)
    a4_t5_times.append(time.time() - g5_start_time)
    a4_times.append(time.time() - algorithm2_start_time)
    print("Iteracion: ",j)

a4_mean = np.array(a4_times).mean()
a4_standard = np.array(a4_times).std()
a4_means.append(np.array(a4_t1_times).mean())
a4_means.append(np.array(a4_t2_times).mean())
a4_means.append(np.array(a4_t3_times).mean())
a4_means.append(np.array(a4_t4_times).mean())
a4_means.append(np.array(a4_t5_times).mean())

a4_std.append(np.array(a4_t1_times).std())
a4_std.append(np.array(a4_t2_times).std())
a4_std.append(np.array(a4_t3_times).std())
a4_std.append(np.array(a4_t4_times).std())
a4_std.append(np.array(a4_t5_times).std())
print(a4_means)
print(a4_std)

n, bins, patches = plt.hist(a4_times, 'auto', density=True, facecolor='blue', alpha=0.75)
y = scipy.stats.norm.pdf(bins, a4_mean,a4_standard)
plt.plot(bins, y, 'r--')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Frecuencia')
plt.title('Árbol de minimo grado (Media='+str(round(a4_mean,2))+' ,STD='+str(round(a4_standard,2))+' )',size=18, color='green')
plt.savefig('H4.eps', format='eps', dpi=1000)
plt.show()

#  Boxplots
to_plot=[a4_t1_times,a4_t2_times,a4_t3_times,a4_t4_times,a4_t5_times]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot, showfliers=False)
plt.xlabel('Grafo')
plt.ylabel('Tiempo (segundos)')
plt.title('Árbol de minimo grado')
plt.savefig('BP4.eps', format='eps', dpi=1000)
plt.show()


# *********************minimum_spanning_tree*********************
for j in range(1,31):
    algorithm2_start_time = time.time()
    g1_start_time = time.time()
    for i in range(201):
        nx.minimum_spanning_tree(G11)
    a5_t1_times.append(time.time() - g1_start_time)

    g2_start_time = time.time()
    for i in range(201):
        nx.minimum_spanning_tree(G12)
    a5_t2_times.append(time.time() - g2_start_time)

    g3_start_time = time.time()
    for i in range(201):
        nx.minimum_spanning_tree(G13)
    a5_t3_times.append(time.time() - g3_start_time)

    g4_start_time = time.time()
    for i in range(201):
        nx.minimum_spanning_tree(G14)
    a5_t4_times.append(time.time() - g4_start_time)

    g5_start_time = time.time()
    for i in range(201):
        nx.minimum_spanning_tree(G15)
    a5_t5_times.append(time.time() - g5_start_time)
    a5_times.append(time.time() - algorithm2_start_time)
    print("Iteracion: ",j)

a5_mean = np.array(a5_times).mean()
a5_standard = np.array(a5_times).std()
a5_means.append(np.array(a5_t1_times).mean())
a5_means.append(np.array(a5_t2_times).mean())
a5_means.append(np.array(a5_t3_times).mean())
a5_means.append(np.array(a5_t4_times).mean())
a5_means.append(np.array(a5_t5_times).mean())

a5_std.append(np.array(a5_t1_times).std())
a5_std.append(np.array(a5_t2_times).std())
a5_std.append(np.array(a5_t3_times).std())
a5_std.append(np.array(a5_t4_times).std())
a5_std.append(np.array(a5_t5_times).std())
print(a5_means)
print(a5_std)

n, bins, patches = plt.hist(a5_times, 'auto', density=True, facecolor='blue', alpha=0.75)
y = scipy.stats.norm.pdf(bins, a5_mean,a5_standard)
plt.plot(bins, y, 'r--')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Frecuencia')
plt.title('Árbol de minima expansión (Media='+str(round(a5_mean,2))+' ,STD='+str(round(a5_standard,2))+' )',size=18, color='green')
plt.savefig('H5.eps', format='eps', dpi=1000)
plt.show()

#  Boxplots
to_plot=[a5_t1_times,a5_t2_times,a5_t3_times,a5_t4_times,a5_t5_times]
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
bp=ax.boxplot(to_plot, showfliers=False)
plt.xlabel('Grafo')
plt.ylabel('Tiempo (segundos)')
plt.title('Árbol de minima expansión')
plt.savefig('BP5.eps', format='eps', dpi=1000)
plt.show()



#  Scatterplots
x_1_1 = (a1_nodes)
x_1_2 = (a2_nodes)
x_1_3 = (a3_nodes)
x_1_4 = (a4_nodes)
x_1_5 = (a5_nodes)

x_2_1 = (a1_edges)
x_2_2 = (a2_edges)
x_2_3 = (a3_edges)
x_2_4 = (a4_edges)
x_2_5 = (a5_edges)

y_1 = (a1_means)
y_2 = (a2_means)
y_3 = (a3_means)
y_4 = (a4_means)
y_5 = (a5_means)

z_1 = (a1_std)
z_2 = (a2_std)
z_3 = (a3_std)
z_4 = (a4_std)
z_5 = (a5_std)

plt.errorbar(y_1,x_1_1, xerr=z_1, fmt='o',color='blue',alpha=0.5)
plt.errorbar(y_2,x_1_2, xerr=z_2, fmt='s',color='yellow',alpha=0.5)
plt.errorbar(y_3,x_1_3, xerr=z_3, fmt='*',color='green',alpha=0.5)
plt.errorbar(y_4,x_1_4, xerr=z_4, fmt='h',color='red',alpha=0.5)
plt.errorbar(y_5,x_1_5, xerr=z_5, fmt='D',color='orange',alpha=0.5)
plt.xlabel('Tiempo (segundos)', size=14)
plt.ylabel('Nodos', size=14)
plt.title('Nodos vs tiempo',size=18)
plt.savefig('S1.eps', format='eps', dpi=1000)
plt.show()


plt.errorbar(y_1,x_2_1, xerr=z_1, fmt='o',color='blue',alpha=0.5)
plt.errorbar(y_2,x_2_2, xerr=z_2, fmt='s',color='yellow',alpha=0.5)
plt.errorbar(y_3,x_2_3, xerr=z_3, fmt='*',color='green',alpha=0.5)
plt.errorbar(y_4,x_2_4, xerr=z_4, fmt='h',color='red',alpha=0.5)
plt.errorbar(y_5,x_2_5, xerr=z_5, fmt='D',color='orange',alpha=0.5)
plt.xlabel('Tiempo (segundos)', size=14)
plt.ylabel('Aristas', size=14)
plt.title('Aristas vs tiempo',size=18)
plt.savefig('S2.eps', format='eps', dpi=1000)
plt.show()