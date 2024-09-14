import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import torch
from cdlib.algorithms import greedy_modularity
import torch.nn as nn
import matplotlib.pyplot as plt
from cdlib import viz

options = {
    'node_color': 'orange',
    'node_size': 50,
    'width': 0.5,
    'with_labels':False,
    'arrows':False
}



class GECo:
    def __init__(self, device, gnn):
        self.device = device
        self.gnn = gnn
    
    
    def find_communities(self,graph):
        g = to_networkx(graph, to_undirected=True)
        comms=greedy_modularity(g)
        return comms
    

    
    def create_subgraph(self, graph, nodes):
        n = graph.x.size()[0]
        mask = torch.zeros(n, dtype=torch.bool).to(self.device)
        mask[list(nodes)] = True
        subgraph = graph.subgraph(mask)
        return subgraph
    
    def predict(self,graph):
        self.gnn.to(self.device)
        self.gnn.eval()
        softmax=nn.Softmax(dim=-1)
        graph=graph.to(self.device)
        with torch.no_grad():
            out = self.gnn(graph.x, graph.edge_index, graph.batch)
            probs=softmax(out)
            probs=probs.detach().cpu().numpy().reshape(-1)
            pred=np.argmax(probs)
        return pred,probs
    
    def visualize_results(self,graph,communities,explenation):
        g = to_networkx(graph, to_undirected=True)
        position = nx.spring_layout(g)
        viz.plot_network_highlighted_clusters(g,communities, position,node_size=50)
        plt.title("Detected communities")
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(8,8))
        plt.title("Explenation")
        nx.draw_networkx(g,position,**options)
        nx.draw_networkx_nodes(g,position, nodelist=explenation, node_color='r',node_size=50)
        plt.show()
        
                

    
    def explain(self,graph,pred,visualize):
        comm_prob=[]
        comms = self.find_communities(graph)
        communities=comms.communities
        for c in communities:
            subgraph = self.create_subgraph(graph, c)
            _, proba = self.predict(subgraph)
            comm_prob.append((c, proba[pred]))
        
        tau = np.mean([p[1] for p in comm_prob], dtype=np.float64)
        
        comm_prob.sort(key=lambda x: x[1], reverse=True)
        exp = [node for c, prob in comm_prob if prob >= tau for node in c]
        

        if visualize:
            self.visualize_results(graph,comms,exp)
        
        return exp


                