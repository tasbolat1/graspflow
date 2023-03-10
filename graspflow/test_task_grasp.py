import numpy as np
import torch
from torch_geometric.nn import SAGEConv, BatchNorm, GCNConv
import torch.nn.functional as F
import torch.nn as nn
import pickle
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG

class GraphNet(torch.nn.Module):
    """Class for Graph Convolutional Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, conv_type):
        super(GraphNet, self).__init__()

        if conv_type == 'GCNConv':
            print('Using GCN Conv')
            ConvLayer = GCNConv
        elif conv_type == 'SAGEConv':
            print('Using SAGE Conv')
            ConvLayer = SAGEConv
        else:
            raise NotImplementedError('Undefine graph conv type {}'.format(conv_type))

        # [B, 156, 301]
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(ConvLayer(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(ConvLayer(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(ConvLayer(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        '''
        Inputs:
            x: [B, 156, 301]
            edge_index: [2, 1564]
        Outputs:
            x: [B, 156, 128]
        '''
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index) # [B, 156, 128]

            # make batch work! Only works when model.eval(True)
            x_size = x.size()
            x = x.reshape([x_size[0]*x_size[1], x_size[2]])
            x = batch_norm(x)
            x = x.reshape([x_size[0],x_size[1], x_size[2]])

            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x

class GCNGrasp(torch.nn.Module):
    def __init__(self, embedding_size=300, gcn_num_layers=6, gcn_conv_type='GCNConv'):
        super().__init__()

        self.embedding_size = embedding_size
        self.gcn_num_layers = gcn_num_layers
        self.gcn_conv_type = gcn_conv_type

        self._build_model()

        # Load graph here
        graph_data_path = '/home/tasbolat/some_python_examples/graspflow_models/graspflow_taskgrasp/data/knowledge_graph/kb2_task_wn_noi/graph_data.pkl'
    
        with open(graph_data_path, "rb") as fh:
            graph, seeds = pickle.load(fh)

        self.build_graph_embedding(graph)

    def _build_model(self):

        pc_dim = 1

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[pc_dim, 32, 32, 64], [pc_dim, 64, 64, 128], [pc_dim, 64, 96, 128]],
                use_xyz=True,
            )
        )

        input_channels = 64 + 128 + 128

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=True,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=True,
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, self.embedding_size)
        )

        self.fc_layer3 = nn.Sequential(
            # [1, 128]
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self.gcn = GraphNet(
            in_channels=self.embedding_size+1,
            hidden_channels=128,
            out_channels=128,
            num_layers=self.gcn_num_layers,
            conv_type=self.gcn_conv_type)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, node_x_idx, latent, edge_index):
        """ Forward pass of GCNGrasp

        Args:
            pointcloud: Variable(torch.cuda.FloatTensor) [B, N, 4] tensor, 
                B is batch size, N is the number of points. The last channel is (x,y,z,feature)
            
            node_x_idx: [V*B] graph index used to lookup embedding dictionary, must be type long

            latent: tensor of size [V*B + B, 1] where V is size of the graph, used to indicate goal task and classes
            
            edge_index: graph adjaceny matrix of size [2, E*B], where E is the number of edges in the graph

        returns:
            logits: binary classification logits
        """

        xyz, features = self._break_up_pc(pointcloud)

        for i, module in enumerate(self.SA_modules):
            xyz, features = module(xyz, features)
        shape_embedding = self.fc_layer(features.squeeze(-1)) # [1, 300]

        # node_x_idx = [155]
        node_embedding = self.graph_embedding(node_x_idx) # [155, 300]

        node_embedding = torch.cat([node_embedding, shape_embedding], dim=0) # [156, 300]
        # latent = [156, 1] [156, 300]
        # latent = [0...... 1, ...., 1,....., 1] <- 1 if task_gid, object_classs_gid or the last one

        node_embedding = torch.cat([node_embedding, latent], dim=1) # [156, 301]
        
        

        # Expected [B,156,128]
        node_embedding = node_embedding.unsqueeze(0)
        output = self.gcn(node_embedding, edge_index) # [156, 128]

        # Expected [B, 156,128]
        output = output.squeeze(0)

        batch_size = pointcloud.shape[0]
        output = output[-batch_size:, :] # [1, 128]
        logits = self.fc_layer3(output)

    
        return logits

    def build_graph_embedding(self, graph):
        """
        Creates and initializes embedding weights for tasks and class nodes in the graph.

        Args:
            graph: networkx DiGraph object
        """
        graph_size = len(list(graph.nodes))
        self.graph_embedding = nn.Embedding(graph_size, self.embedding_size)

if __name__ == "__main__":
    torch.manual_seed(40)

    #model = GraphNet(in_channels=301, hidden_channels=128, out_channels=128, num_layers=6, conv_type='GCNConv')

    device = torch.device('cuda:0')

    # define model
    model = GCNGrasp()

    # load weights
    weights = torch.load('/home/tasbolat/some_python_examples/graspflow_models/graspflow_taskgrasp/gcngrasp/gcn.pt')
    model.load_state_dict(weights)
    model = model.to(device)
    model = model.eval()  # make sure it's eval mode


    #### TEST 1: NO BATCH ####
    print('Test 1')


    B = 1
    B_future = 25
    x = torch.rand([B, 4096+7,4]).to(device)
    node_x_idx = torch.from_numpy(np.arange(155, dtype=int)).to(device)
    node_x_idx = node_x_idx.repeat(B)
    latent = torch.rand([155, 1]).to(device)

    # we will use it for future
    print(latent.shape)
    latent_for_future = latent.repeat([B_future,1])
    print(latent_for_future.shape)
    latent_for_future = torch.concat([latent_for_future, torch.ones([B_future,1]).to(device)], dim=0)

    print(latent_for_future.shape)

    latent = latent.repeat([B,1])
    latent = torch.concat([latent, torch.ones([B,1]).to(device)], dim=0)

    edge_index = torch.rand([2, 1564]).to(device).long()

    edge_index_future = edge_index.repeat([1,B_future]) # optional

    print('IN')
    print(x.shape, node_x_idx.shape, latent.shape, edge_index.shape)
    
    out = model(pointcloud=x, node_x_idx=node_x_idx, latent=latent, edge_index=edge_index)

    
    print(out.sum(1))


    #### TEST 2: BATCH CASE ####
    # Output shall be 4 times replicated of the previous one
    print('Test 2')

    x = x.repeat([B_future,1,1])
    node_x_idx = node_x_idx.repeat(B_future)


    print('IN')
    print(x.dtype, node_x_idx.dtype, latent_for_future.dtype, edge_index_future.dtype)

    x = x.requires_grad_(True) # here it says I need gradient respect to it
    np.savez("test_pc_base_random.npz", pc = x.cpu().detach().numpy() , node_x_idx = node_x_idx.cpu(), latent = latent_for_future.cpu(), edge_index = edge_index_future.cpu())

    out = model(pointcloud=x, node_x_idx=node_x_idx, latent=latent_for_future, edge_index=edge_index_future)

    loss = out.sum(1)

    print(loss)


    #### TEST 3: Backprop on input ####
    # You should see clear values in x.grad

    loss.backward(torch.ones_like(loss))

    print(x.grad) # here it calculates after backprop


    # #### TEST 4: COMPARE with original one:
    # # TODO
    # print('Test 4')
    # dddata = np.load("test_pc_base.npz")
    # pc = torch.tensor(dddata["pc"])
    # pc_mean = torch.tensor(dddata['pc_mean'])
    # t = torch.tensor(dddata['t'])
    # r = torch.tensor(dddata['r'])
    # query = dddata['query']

    # B = 4
    # x = torch.rand([B, 4096+7,4]).to(device)
    # x = x.repeat([B,1,1])
    # latent = torch.rand([155, 1]).to(device)
    # node_x_idx = torch.from_numpy(np.arange(155, dtype=int)).to(device)
    # node_x_idx = node_x_idx.repeat(B)


    # print('IN')
    # print(x.shape, node_x_idx.shape, latent.shape, edge_index.shape)

    # x = x.requires_grad_(True)
    
    # out = model(pointcloud=x, node_x_idx=node_x_idx, latent=latent_for_future, edge_index=edge_index_future)

    # loss = out.sum(1)

    # exit()
