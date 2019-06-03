import pandas as pd
import networkx as nx
import numpy as np
from random import choice
import time
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

class Test:
    def __init__(self, generation,algorithm,nodes,density,time):
        self.generation = generation
        self.algorithm = algorithm
        self.nodes  = nodes
        self.density = density
        self.time = time
    def __repr__(self):
        return str(self.__dict__)
    def to_dict(self):
        return {
            'generation': self.generation,
            'algorithm': self.algorithm,
            'nodes': self.nodes,
            'density': self.density,
            'time': self.time,
        }
totalTests = list()
G1S=0
G2S=0
G3S=0
G1T=0
G2T=0
G3T=0
mu=10
sigma=2.5
#Orden logaritmico
for logarithmOrder in range(7,11):
    print('Orden: ',logarithmOrder)
    log = 2**logarithmOrder
    ban=True
    # 3 Metodos de generación distintos
    G1 = nx.lollipop_graph(log, 2)
    for e in G1.edges():
        G1[e[0]][e[1]]['capacity'] = np.random.normal(mu, sigma)
    G2 = nx.turan_graph(log, 2)
    for e in G2.edges():
        G2[e[0]][e[1]]['capacity'] = np.random.normal(mu, sigma)
    G3 = nx.ladder_graph(log)
    for e in G3.edges():
        G3[e[0]][e[1]]['capacity'] = np.random.normal(mu, sigma)
    for newPair in range(5):
        print('Pareja: ',newPair)
        #10 Grafos
        for graphRepetition in range(10):
            if ban:
                ban=False
                while G1T==G1S:
                    G1S = choice(list(G1.nodes()))
                    G1T = choice(list(G1.nodes()))
                while G2T==G2S:
                    G2S = choice(list(G2.nodes()))
                    G2T = choice(list(G2.nodes()))
                while G3T==G3S:
                    G3S = choice(list(G3.nodes()))
                    G3T = choice(list(G3.nodes()))
            # GRAFO 1
            g1_start_time = time.time()
            nx.maximum_flow(G1,G1S,G1T)
            g1_end_time = time.time() - g1_start_time
            totalTests.append(Test(0, 0, logarithmOrder, nx.density(G1), g1_end_time))

            g1_start_time = time.time()
            nx.maximum_flow_value(G1, G1S, G1T)
            g1_end_time = time.time() - g1_start_time
            totalTests.append(Test(0, 1, logarithmOrder, nx.density(G1), g1_end_time))

            g1_start_time = time.time()
            nx.minimum_cut_value(G1, G1S, G1T)
            g1_end_time = time.time() - g1_start_time
            totalTests.append(Test(0, 2, logarithmOrder, nx.density(G1), g1_end_time))

            # GRAFO 2
            g1_start_time = time.time()
            nx.maximum_flow(G2, G2S, G2T)
            g1_end_time = time.time() - g1_start_time
            totalTests.append(Test(1, 0, logarithmOrder, nx.density(G2), g1_end_time))

            g1_start_time = time.time()
            nx.maximum_flow_value(G2, G2S, G2T)
            g1_end_time = time.time() - g1_start_time
            totalTests.append(Test(1, 1, logarithmOrder, nx.density(G2), g1_end_time))

            g1_start_time = time.time()
            nx.minimum_cut_value(G2, G2S, G2T)
            g1_end_time = time.time() - g1_start_time
            totalTests.append(Test(1, 2, logarithmOrder, nx.density(G2), g1_end_time))

            # GRAFO 3
            g1_start_time = time.time()
            nx.maximum_flow(G3, G3S, G3T)
            g1_end_time = time.time() - g1_start_time
            totalTests.append(Test(2, 0, logarithmOrder, nx.density(G3), g1_end_time))

            g1_start_time = time.time()
            nx.maximum_flow_value(G3, G3S, G3T)
            g1_end_time = time.time() - g1_start_time
            totalTests.append(Test(2, 1, logarithmOrder, nx.density(G3), g1_end_time))

            g1_start_time = time.time()
            nx.minimum_cut_value(G3, G3S, G3T)
            g1_end_time = time.time() - g1_start_time
            totalTests.append(Test(2, 2, logarithmOrder, nx.density(G3), g1_end_time))
print(totalTests.__sizeof__())

data = pd.DataFrame.from_records([s.to_dict() for s in totalTests])
#data.boxplot(column=['time'],by=['nodes','generation','algorithm'])
data.boxplot(column=['time'],by='density')
print(data)
#ANOVA
formula1 = 'time ~ C(generation)'
formula2 = 'time ~ C(algorithm)'
formula3 = 'time ~ C(nodes)'
formula4 = 'time ~ C(density)'
formula5 = 'time ~ C(generation)*C(algorithm)'
formula6 = 'time ~ C(generation)*C(nodes)'
formula7 = 'time ~ C(generation)*C(density)'
formula8 = 'time ~ C(algorithm)*C(nodes)'
formula9 = 'time ~ C(algorithm)*C(density)'
formula10 = 'time ~ C(nodes)*C(density)'
formula11 = 'time ~ C(generation)*C(algorithm)*C(nodes)'
formula12 = 'time ~ C(generation)*C(algorithm)*C(density)'
formula13 = 'time ~ C(algorithm)*C(nodes)*C(density)'
formula14 = 'time ~ C(generation)*C(algorithm)*C(nodes)*C(density)'
model1 = ols(formula1, data).fit()
model2 = ols(formula2, data).fit()
model3 = ols(formula3, data).fit()
model4 = ols(formula4, data).fit()
model5 = ols(formula5, data).fit()
model6 = ols(formula6, data).fit()
model7 = ols(formula7, data).fit()
model8 = ols(formula8, data).fit()
model9 = ols(formula9, data).fit()
model10 = ols(formula10, data).fit()
model11 = ols(formula11, data).fit()
model12 = ols(formula12, data).fit()
model13 = ols(formula13, data).fit()
model14 = ols(formula14, data).fit()
anova_table1 = anova_lm(model1, typ=2)
anova_table2 = anova_lm(model2, typ=2)
anova_table3 = anova_lm(model3, typ=2)
anova_table4 = anova_lm(model4, typ=2)
anova_table5 = anova_lm(model5, typ=2)
anova_table6 = anova_lm(model6, typ=2)
anova_table7 = anova_lm(model7, typ=2)
anova_table8 = anova_lm(model8, typ=2)
anova_table9 = anova_lm(model9, typ=2)
anova_table10 = anova_lm(model10, typ=2)
anova_table11 = anova_lm(model11, typ=2)
anova_table12 = anova_lm(model12, typ=2)
anova_table13 = anova_lm(model13, typ=2)
anova_table14 = anova_lm(model14, typ=2)
#anova_table1.to_csv('anova1.xls',sep='\t')
#anova_table2.to_csv('anova2.xls',sep='\t')
#anova_table3.to_csv('anova3.xls',sep='\t')
#anova_table4.to_csv('anova4.xls',sep='\t')
#anova_table5.to_csv('anova5.xls',sep='\t')
#anova_table6.to_csv('anova6.xls',sep='\t')
#anova_table7.to_csv('anova7.xls',sep='\t')
#anova_table8.to_csv('anova8.xls',sep='\t')
#anova_table9.to_csv('anova9.xls',sep='\t')
#anova_table10.to_csv('anova10.xls',sep='\t')
#anova_table11.to_csv('anova11.xls',sep='\t')
#anova_table12.to_csv('anova12.xls',sep='\t')
#anova_table13.to_csv('anova13.xls',sep='\t')
#anova_table14.to_csv('anova14.xls',sep='\t')
#data.to_csv('data.xls', sep='\t')

#MATRIZ CORRELACION
data = data.iloc[:, :-1]
data.columns = ['algoritmo','densidad','generacion',
                     'orden']
corr = data.corr()
print(corr.columns)
fig, ax = plt.subplots(figsize=(6, 6))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
cax = ax.matshow(corr,cmap='OrRd', vmin=-1, vmax=1, aspect='equal', origin='lower')
fig.colorbar(cax)
plt.show()
