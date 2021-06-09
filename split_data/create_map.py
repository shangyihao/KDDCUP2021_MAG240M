import time
import os.path as osp
from ogb.lsc import MAG240MDataset
from tqdm import tqdm
import pickle


def create_map():
    # _map = defaultdict(  # edge_type:'cites','writes','affiliated with'
    #     lambda: defaultdict(  # source_id
    #         lambda: []  # [source_id, target_id]
    #     ))

    # _map = {}
    # '''
    # _map={'edge_type:'cites','writes','affiliated with'':
    #             {source_id: [source_id, target_id]}
    #             }
    # '''

    edge_types = ['cites', 'writes', 'affiliated with']

    def add_map(edge_type):
        t = time.perf_counter()
        print('load edge_index...', end=' ', flush=True)
        if edge_type == 'cites':
            edge_index = dataset.edge_index('paper', 'paper')
            row, col = edge_index
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
        elif edge_type == 'writes':
            edge_index = dataset.edge_index('author', 'paper')
            row, col = edge_index
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
        else:
            edge_index = dataset.edge_index('author', 'institution')
            col, row = edge_index  # let institution as source_node
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        _len = len(row)
        edge = {}
        t = time.perf_counter()
        print('add specific map for the given edge_type...', end=' ', flush=True)
        for i in tqdm(range(_len)):
            e = [row[i], col[i]]
            edge.setdefault(row[i], [])
            edge[row[i]].append(e)
        # _map.setdefault(edge_type, {})
        # _map[edge_type] = edge
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
        return edge

    for _type in edge_types:
        file = f'{dataset.dir}/process_shang/map_{_type}.txt'
        if not osp.exists(file):
            edge_map = add_map(_type)
            t = time.perf_counter()
            print(f'save map_{_type}...', end=' ', flush=True)
            with open(file, 'wb') as outfile:
                pickle.dump(edge_map, outfile)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

    # path = f'{dataset.dir}/process_shang/map.txt'
    # if not osp.exists(path):
    #     t = time.perf_counter()
    #     print(f'save map.txt...', end=' ', flush=True)
    #     with open(path,  'wb')as outfile:
    #         pickle.dump(_map, outfile)
    #     print(f'Done! [{time.perf_counter() - t:.2f}s]')
    #
    #     t = time.perf_counter()
    #     print('Cleaning up...', end=' ', flush=True)
    #     for _type in edge_types:
    #         os.remove(f'{dataset.dir}/process_shang/map_{_type}.json')
    #     print(f'Done! [{time.perf_counter() - t:.2f}s]')


if __name__ == '__main__':
    ROOT = '/data/shangyihao/'
    dataset = MAG240MDataset(ROOT)
    create_map()



