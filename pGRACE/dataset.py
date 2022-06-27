import os.path as osp

from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon, TUDataset, PolBlogs
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy', 'PolBlogs',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code', 'Proteins']
    name = 'dblp' if name == 'DBLP' else name

    if name == 'Proteins':
        return TUDataset(root=path, name='PROTEINS', transform=T.NormalizeFeatures())

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    if name == 'PolBlogs':
        # 处理polblog数据集
        # from deeprobust.graph.data import Dataset
        # from deeprobust.graph.data import Dpr2Pyg
        # data_polblogs_deep = Dataset(root='A:/pycharm_project/CLGA-main/tmp/', name='polblogs')
        # data_polblogs_pyg = Dpr2Pyg(dpr_data=data_polblogs_deep)
        # return data_polblogs_pyg
        return PolBlogs(root=osp.join(path, 'Citation'), transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(path, 'Citation'), name, transform=T.NormalizeFeatures())
