import pandas as pd
import networkx as nx
import numpy as np
from random import choice
import time
import matplotlib.pyplot as plt
total_data=[]
num_fig=1
def node_attribute(G,node,nodeT):
    global total_data
    for i in range(10):
        data = {}
        start_time = time.time()
        for j in range(100):
            nx.maximum_flow(G, node, nodeT)
        data["G"] = G.degree(node)
        data["A"]= nx.clustering(G, node)
        data["C"] = nx.closeness_centrality(G, node)
        data["Ce"] = nx.load_centrality(G, node)
        data["E"] = nx.eccentricity(G, node)
        pageRank = nx.pagerank(G, weight="capacity")
        data["P"] = pageRank[node]
        data["Tiempo"] = time.time() - start_time
        total_data.append(data)

def graph_attributes(G):
    global num_fig
    nodes=G.nodes()
    degrees=[G.degree(i) for i in nodes]
    clustering=[nx.clustering(G,i) for i in nodes]
    closeness=[nx.closeness_centrality(G,i) for i in nodes]
    load=[nx.load_centrality(G,i) for i in nodes]
    eccentricity=[nx.eccentricity(G,i) for i in nodes]
    pageRank=nx.pagerank(G,weight="capacity")
    pageRank_G=[pageRank[i] for i in nodes]
    plt.figure(0)
    plt.ylabel('Replicas')
    plt.xlabel('Grado del nodo')
    plt.hist(degrees)
    plt.figure(1)
    plt.savefig('hist-grado-'+str(num_fig)+'.eps', format='eps', dpi=1000)
    #plt.clf()
    plt.ylabel('Replicas')
    plt.xlabel('Coeficiente de agrupamiento del nodo')
    plt.hist(clustering)
    plt.figure(2)
    plt.savefig('hist-agrupamiento-' + str(num_fig) + '.eps', format='eps', dpi=1000)
    plt.clf()
    plt.ylabel('Replicas')
    plt.xlabel('Coeficiente de cercania del nodo')
    plt.hist(closeness)
    plt.figure(3)
    plt.savefig('hist-cercania-' + str(num_fig) + '.eps', format='eps', dpi=1000)
    plt.clf()
    plt.ylabel('Replicas')
    plt.xlabel('Centralidad del nodo')
    plt.hist(load)
    plt.figure(4)
    plt.savefig('hist-centralidad-' + str(num_fig) + '.eps', format='eps', dpi=1000)
    plt.clf()
    plt.ylabel('Replicas')
    plt.xlabel('Excentricidad del nodo')
    plt.hist(eccentricity)
    plt.figure(5)
    plt.savefig('hist-excentricidad-' + str(num_fig) + '.eps', format='eps', dpi=1000)
    plt.clf()
    plt.ylabel('Replicas')
    plt.xlabel('PageRank del nodo')
    plt.hist(pageRank_G)
    plt.figure(6)
    plt.savefig('hist-pagerank-' + str(num_fig) + '.eps', format='eps', dpi=1000)
    plt.clf()
    #plt.show()

def do_graph(G):
    global num_fig
    mu=10
    sigma=2.5
    pos = nx.spring_layout(G)
    G1S = choice(list(G.nodes()))
    G1T=G1S
    while G1T==G1S:
        G1T = choice(list(G.nodes()))
    for e in G.edges():
        G[e[0]][e[1]]['capacity'] = round(np.random.normal(mu, sigma),2)
    val_map_color = {G1S: 1.0,G1T: 0}
    val_map_size = {G1S: 200.0,G1T: 200}
    map_color = [val_map_color.get(node, 0.5) for node in G.nodes()]
    map_size = [val_map_size.get(node, 70) for node in G.nodes()]
    edges,weights = zip(*nx.get_edge_attributes(G,'capacity').items())
    labels = nx.get_edge_attributes(G,'capacity')
    plt.figure(7)
    plt.savefig('grafo-' + str(num_fig) + '.eps', format='eps', dpi=1000)
    plt.clf()
    flow_value, flow_dict = nx.maximum_flow(G,G1S,G1T)
    #node_attribute(G,G1S,G1T)
    nx.draw(G,pos,node_size=map_size,weight='capacity',edge_color=weights, width=2.0,cmap=plt.get_cmap('viridis'), edge_cmap=plt.cm.Blues,edge_labels=labels,node_color=map_color, font_color='white')
    graph_attributes(G)
    #plt.show()
    res_G = nx.Graph()
    for node in flow_dict:
        for sub_node in flow_dict[node]:
            res_G.add_edge(node,sub_node,capacity=flow_dict[node][sub_node],color='r')
    val_map_color = {G1S: 1.0, G1T: 0}
    val_map_size = {G1S: 200.0, G1T: 200}
    map_color = [val_map_color.get(node, 0.5) for node in res_G.nodes()]
    map_size = [val_map_size.get(node, 70) for node in res_G.nodes()]
    edges, weights = zip(*nx.get_edge_attributes(res_G, 'capacity').items())
    labels = nx.get_edge_attributes(res_G, 'capacity')
    plt.figure(8)
    res_pos = nx.spring_layout(res_G)
    nx.draw(res_G, res_pos, node_size=map_size, weight='capacity', edge_color=weights, width=2.0, cmap=plt.get_cmap('viridis'),
            edge_cmap=plt.cm.Blues, edge_labels=labels, node_color=map_color, font_color='white')
    plt.savefig('res-' + str(num_fig) + '.eps', format='eps', dpi=1000)
    plt.clf()
    num_fig+=1
    #plt.show()


G1 = nx.connected_caveman_graph(4, 5)
G2 = nx.connected_caveman_graph(5, 6)
G3 = nx.connected_caveman_graph(6, 7)
G4 = nx.connected_caveman_graph(7, 8)
G5 = nx.connected_caveman_graph(8, 9)
do_graph(G5)
do_graph(G2)
do_graph(G3)
do_graph(G4)
do_graph(G1)
#dataframe = pd.DataFrame(total_data)
#dataframe.to_csv('datos.xls',sep='\t')
#model = ols('Tiempo~G*A*C*Ce*E*P',data=dataframe).fit()
#anova = anova_lm(model,typ=2)
#anova.to_csv('anova1.xls',sep='\t')