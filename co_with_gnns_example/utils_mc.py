import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dgl.nn.pytorch import GraphConv

import dgl
import random
import os
import numpy as np
import networkx as nx

from collections import OrderedDict, defaultdict
from itertools import chain, islice, combinations
from time import time


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

# GNN class to be instantiated with specified param values
class GCN_dev(nn.Module):
    def __init__(self, in_feats, hidden_sizes, num_classes, dropout, device):
        """
        Initialize a new instance of the core GCN model of provided size.
        Dropout is added in forward step.

        Inputs:
            in_feats: Dimension of the input (embedding) layer
            hidden_size: Hidden layer size
            dropout: Fraction of dropout to add between intermediate layer. Value is cached for later use.
            device: Specifies device (CPU vs GPU) to load variables onto
        """
        super(GCN_dev, self).__init__()

        # Combine all layers sizes into a single list
        all_layers = [in_feats] + hidden_sizes + [num_classes]
        # slice list into sub-lists of length 2
        self.layer_sizes = list(window(all_layers))
        # reference to ID final layer
        self.out_layer_id = len(self.layer_sizes) - 1
        self.dropout_frac = dropout
        self.layers = OrderedDict()
        for idx, (layer_in, layer_out) in enumerate(self.layer_sizes):
            self.layers[idx] = GraphConv(layer_in, layer_out).to(device)

    def forward(self, g, inputs):
        """
        Run forward propagation step of instantiated model.

        Input:
            self: GCN_dev instance
            g: DGL graph object, i.e. problem definition
            inputs: Input (embedding) layer weights, to be propagated through network
        Output:
            h: Output layer weights
        """

        for k, layer in self.layers.items():
            if k == 0: # reference to ID final layer
                h = layer(g, inputs)
                h = torch.relu(h)
                h = F.dropout(h, p=self.dropout_frac)
            elif 0 < k < self.out_layer_id: # intermediate layers
                h = layer(g,h)
                h = torch.relu(h)
                h = F.dropout(h, p=self.dropout_frac)
            else: # output layer
                h = layer(g, h)
                h = torch.sigmoid(h)
        return h


# Generate random graph of specified size and type,
# with specified degree (d) or edge probability (p)
def generate_graph(n, d=None, p=None, graph_type='reg', random_seed=0):
    """
    Helper function to generate a NetworkX random graph of specified type,
    given specified parameters (e.g. d-regular, d=3). Must provide one of
    d or p, d with graph_type='reg', and p with graph_type in ['prob', 'erdos'].

    Input:
        n: Problem size
        d: [Optional] Degree of each node in graph
        p: [Optional] Probability of edge between two nodes
        graph_type: Specifies graph type to generate
        random_seed: Seed value for random generator
    Output:
        nx_graph: NetworkX OrderedGraph of specified type and parameters
    """
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'prob':
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
    elif graph_type == 'erdos':
        print(f'Generating erdos-renyi graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.erdos_renyi_graph(n, p, seed=random_seed)
    elif graph_type == 'weight':
        print(f'Generating weighted d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
        for (u,v,w) in nx_temp.edges(data=True):
            w['weight'] = random.randint(1,10)
    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')

    # Networkx does not enforce node order by default
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)
    # Need to pull nx graph into OrderedGraph so training will work properly
    nx_graph = nx.OrderedGraph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges(data=True))
    return nx_graph


# helper function to convert Q dictionary to torch tensor
def qubo_dict_to_torch(nx_G, Q, torch_dtype=None, torch_device=None):
    """
    Output Q matrix as torch tensor for given Q in dictionary format.

    Input:
        Q: QUBO matrix as defaultdict
        nx_G: graph as networkx object (needed for node lables can vary 0,1,... vs 1,2,... vs a,b,...)
    Output:
        Q: QUBO as torch tensor
    """

    # get number of nodes
    n_nodes = len(nx_G.nodes)

    # get QUBO Q as torch tensor
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val

    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)

    return Q_mat


# Chunk long list
def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])


# helper function for custom loss according to Q matrix
def loss_func(probs, Q_mat):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """

    probs_ = torch.unsqueeze(probs, 1)

    # minimize cost = x.T * Q * x
    cost = (probs_.T @ Q_mat @ probs_).squeeze()

    return cost


# Construct graph to learn on
def get_gnn(n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    """
    Generate GNN instance with specified structure. Creates GNN, retrieves embedding layer,
    and instantiates ADAM optimizer given those.

    Input:
        n_nodes: Problem size (number of nodes in graph)
        gnn_hypers: Hyperparameters relevant to GNN structure
        opt_params: Hyperparameters relevant to ADAM optimizer
        torch_device: Whether to load pytorch variables onto CPU or GPU
        torch_dtype: Datatype to use for pytorch variables
    Output:
        net: GNN instance
        embed: Embedding layer to use as input to GNN
        optimizer: ADAM optimizer instance
    """
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']

    # instantiate the GNN
    net = GCN_dev(dim_embedding, hidden_dim, number_classes, dropout, torch_device)
    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, **opt_params)
    return net, embed, optimizer


# Parent function to run GNN training given input config
def run_gnn_training(q_torch, dgl_graph, net, embed, optimizer, number_epochs, tol, patience, prob_threshold):
    """
    Wrapper function to run and monitor GNN training. Includes early stopping.
    """
    # Assign variable for user reference
    inputs = embed.weight

    prev_loss = 1.  # initial loss value (arbitrary)
    count = 0       # track number times early stopping is triggered

    # initialize optimal solution
    best_bitstring = torch.zeros((dgl_graph.number_of_nodes(),)).type(q_torch.dtype).to(q_torch.device)
    best_loss = loss_func(best_bitstring.float(), q_torch)

    t_gnn_start = time()

    # Training logic
    for epoch in range(number_epochs):

        # get logits/activations
        probs = net(dgl_graph, inputs)[:, 0]  # collapse extra dimension output from model

        # build cost value with QUBO cost function
        loss = loss_func(probs, q_torch)
        loss_ = loss.detach().item()

        # Apply projection
        bitstring = (probs.detach() >= prob_threshold) * 1
        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_}')

        # early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f'Stopping early on epoch {epoch} (patience: {patience})')
            break

        # update loss tracking
        prev_loss = loss_

        # run optimization with backpropagation
        optimizer.zero_grad()  # clear gradient for step
        loss.backward()        # calculate gradient through compute graph
        optimizer.step()       # take step, update weights

    t_gnn = time() - t_gnn_start
    print(f'GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    print(f'GNN final continuous loss: {loss_}')
    print(f'GNN best continuous loss: {best_loss}')

    final_bitstring = (probs.detach() >= prob_threshold) * 1

    return net, epoch, final_bitstring, best_bitstring


# helper function to generate Q matrix for Maxi Cut Problem (MC)
def gen_q_dict_mc(nx_G):
    """
    Helper function to generate QUBO matrix for MIS as minimization problem.
    
    Input:
        nx_G: graph as networkx graph object (assumed to be unweigthed)
    Output:
        Q_dic: QUBO as defaultdict
    """

    # Initialize our Q matrix
    Q_dic = defaultdict(int)

    # Update Q matrix for every edge in the graph
    # all off-diagonal terms get penalty
    for (u,v,w) in nx_G.edges(data=True):
        Q_dic[(u, v)] = 2*w['weight']
        Q_dic[(u, u)] -= w['weight']
        Q_dic[(v, v)] -= w['weight']

    return Q_dic

# Calculate results given bitstring and graph definition, includes check for violations
def postprocess_gnn_mc(best_bitstring, nx_graph):
    """
    helper function to postprocess MIS results

    Input:
        best_bitstring: bitstring as torch tensor
    Output:
        size_mc: size of maxcut
    """

    # get bitstring as list
    bitstring_list = list(best_bitstring)

    first_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])

    size_mc = 0

    for (u, v, w) in nx_graph.edges(data=True):
        if u in first_set:
            if not (v in first_set):
                size_mc += w['weight']
        elif v in first_set:
            size_mc += w['weight']

    return size_mc

def MaxCut(nx_graph, seed_value = 1):

    # MacOS can have issues with MKL. For more details, see
    # https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    random.seed(seed_value)        # seed python RNG
    np.random.seed(seed_value)     # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG

    # Set GPU/CPU
    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TORCH_DTYPE = torch.float32
    print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')


    # get DGL graph from networkx graph, load onto device
    graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
    graph_dgl = graph_dgl.to(TORCH_DEVICE)

    # Construct Q matrix for graph
    q_torch = qubo_dict_to_torch(nx_graph, gen_q_dict_mc(nx_graph), torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)

    # Visualize graph
    # pos = nx.kamada_kawai_layout(nx_graph)
    # nx.draw(nx_graph, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # labels = nx.get_edge_attributes(nx_graph,'weight')
    # nx.draw_networkx_edge_labels(next,pos,edge_labels=labels)
    # plt.show()

    # Graph hypers
    n = nx_graph.number_of_nodes()
    graph_type = 'weight'

    # NN learning hypers #
    number_epochs = int(1e5)
    learning_rate = 1e-4
    PROB_THRESHOLD = 0.5

    # Early stopping to allow NN to train to near-completion
    tol = 1          # loss must change by more than tol, or trigger
    patience = 100    # number early stopping triggers before breaking loop

    # Establish dim_embedding and hidden_dim values
    dim_embedding = max( int(np.sqrt(n)) , 1)    # e.g. 10
    # dim_embedding = 369
    hidden_dim = [ max( int(dim_embedding/2) , 1) ]  # e.g. 5
    # hidden_dim = [5]

    # Establish pytorch GNN + optimizer
    opt_params = {'lr': learning_rate}
    gnn_hypers = {
        'dim_embedding': dim_embedding,
        'hidden_dim': hidden_dim,
        'dropout': 0.0,
        'number_classes': 1,
        'prob_threshold': PROB_THRESHOLD,
        'number_epochs': number_epochs,
        'tolerance': tol,
        'patience': patience
    }

    net, embed, optimizer = get_gnn(n, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)

    # For tracking hyperparameters in results object
    gnn_hypers.update(opt_params)

    print('Running GNN...')
    gnn_start = time()

    _, epoch, final_bitstring, best_bitstring = run_gnn_training(
        q_torch, graph_dgl, net, embed, optimizer, gnn_hypers['number_epochs'],
        gnn_hypers['tolerance'], gnn_hypers['patience'], gnn_hypers['prob_threshold'])

    gnn_time = time() - gnn_start

    final_loss = loss_func(final_bitstring.float(), q_torch)
    final_bitstring_str = ','.join([str(x) for x in final_bitstring])

    # Process bitstring reported by GNN
    size_mc = postprocess_gnn_mc(best_bitstring, nx_graph)
    gnn_tot_time = time() - gnn_start

    print(f'MacxCut found by GNN is {size_mc}')
    print(f'Took {round(gnn_tot_time, 3)}s, model training took {round(gnn_time, 3)}s')

    return gnn_tot_time, size_mc, epoch

    # Visualize result
    # pos = nx.drawing.layout.bipartite_layout(nx_graph,set([node for node, entry in enumerate(best_bitstring) if entry == 1]))
    # color_map = ['orange' if (best_bitstring[node]==0) else 'lightblue' for node in nx_graph.nodes]
    # nx.draw(nx_graph, pos, with_labels=True, node_color=color_map)
    # labels = nx.get_edge_attributes(nx_graph,'weight')
    # nx.draw_networkx_edge_labels(next,pos,edge_labels=labels)
    # plt.show()