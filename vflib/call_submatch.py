import argparse
import itertools
import json
import os
import pickle as pkl
from collections import defaultdict
from subprocess import call

import numpy as np


class Graph(object):

    def __init__(self, graph_path, node_info_path):
        self.init_nbs(graph_path, node_info_path)  # read graph.
        self.num_node = len(self.nbs)

    def init_nbs(self, graph_path, node_info_path):

        self.nbs = defaultdict(lambda: set())
        self.node_types = defaultdict(lambda: int)
        self.node_names = defaultdict(lambda: str)

        with open(graph_path, 'r') as f:
            next(f)
            for line in f:
                [n1, n2] = list(map(int, line.rstrip().split()))
                self.nbs[n1].add(n2)
                # self.nbs[n2].add(n1)
        with open(node_info_path, 'r') as f:
            for line in f:
                [id, type] = map(int, line.split('\t')[0:2])
                name = line.split('\t')[2]
                self.node_types[id] = type
                self.node_names[id] = name
    # get edge set

    def get_es(self):
        es = []
        for n in self.node_types:
            es += [(n, nb) for nb in self.nbs[n]]
        return es


class Preprocess(object):
    query_path = 'motif.txt'
    graph_submatch_path = 'graph.txt'

    def __init__(self, graph_path, node_info_path, motifs_path):
        self.graph_path = graph_path
        self.node_info_path = node_info_path
        self.motifs_path = motifs_path
        self.graph = Graph(self.graph_path, self.node_info_path)
    # Create graph input for the submatch tool.

    def create_graph(self):
        G = self.graph
        with open(self.graph_submatch_path, 'w') as f:
            f.write(str(len(G.node_types)) + '\n')
            for i in range(0, len(G.node_types)):
                f.write(str(i) + ' ' + str(G.node_types[i]) + '\n')
            adj = {}
            for i in range(0, len(G.node_types)):
                adj[i] = []
            for e in G.get_es():
                adj[e[0]].append(e[1])

            for i in range(0, len(G.node_types)):
                f.write(str(len(adj[i])) + '\n')
                for j in adj[i]:
                    f.write('%d %d\n' % (i, j))

    # Create motif - as query -  input for the submatch tool.

    def create_motifs(self):
        motifs = {}
        for k, v in json.load(open(self.motifs_path)).items():
            motifs[int(k)] = v
        self.num_ns = [len(v['v']) for k, v in sorted(motifs.items())]
        self.num_es = [len(v['e']) for k, v in sorted(motifs.items())]

        with open(self.query_path, 'w') as f:
            for key, val in sorted(motifs.items()):
                # f.write('t # ' + str(key) + '\n')
                adj = {}
                for i in range(0, len(val['v'])):
                    adj[i] = []

                for e in val['e']:
                    adj[e[0]].append(e[1])

                f.write(str(len(val['v'])) + '\n')
                for i in range(len(val['v'])):
                    f.write('%d %d\n' % (i, int(val['v'][i])))
                # f.write(str(len(val['v']))+'\t'+str(len(val['e']))+'\n')
                for i in range(0, len(val['v'])):
                    f.write(str(len(adj[i])) + '\n')
                    for j in adj[i]:
                        f.write('%d %d\n' % (i, j))
                # f.write(str(len(val['e']))+'\n')

                # for e in val['e']:
                #    f.write('%d %d %d\n' % (e[0], e[1],1))
                # f.write('\n')
        self.motifs = motifs

    def verifyInstance(self, instance, motif):
        motif_types = map(int, motif['v'])
        instance_types = map(lambda x: self.graph.node_types[x], instance)
        if instance_types != motif_types:
            return False
        # Check each edge now.
        for e in motif['e']:
            a = instance[int(e[0])]
            b = instance[int(e[1])]
            if b not in self.graph.nbs[a]:
                return False
        return True

    def getMappings(self, instance, motif):
        instances_ordered = []
        perms = itertools.permutations(instance)
        for perm in perms:
            skip = False
            for v in range(0, len(motif['v'])):
                # label of node in the instance.
                label = self.graph.node_types[perm[v]]
                if str(label) != str(motif['v'][v]):
                    skip = True
                    break

            for edge in motif['e']:
                a = perm[int(edge[0])]
                b = perm[int(edge[1])]
                if b not in self.graph.nbs[a]:
                    skip = True
                    break
            if skip is False:
                instances_ordered.append(np.array(perm))
        mappings = []
        for instance_ordered in instances_ordered:
            mapping = []
            ordered_perm = instance_ordered  # take each ordered instance.
            for x in ordered_perm:
                mapping.append(instance.index(x))

            mapped_instance = []
            for i in range(0, len(instance)):
                mapped_instance.append(instance[mapping[i]])
            mappings.append(mapping)
        return mappings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', '-G',
                        default='../motif-cnn/data/dblp/links.txt')
    parser.add_argument('--node_info_path', '-N',
                        default='../motif-cnn/data/dblp/node_info.txt')
    parser.add_argument('--motifs_path', '-M', default='./motifs.json')
    args = parser.parse_args()

    graph_path = os.path.abspath(args.graph_path)
    node_info_path = os.path.abspath(args.node_info_path)
    motifs_path = os.path.abspath(args.motifs_path)

    preproc = Preprocess(graph_path, node_info_path, motifs_path)
    preproc.create_motifs()
    preproc.create_graph()
    call('./vflib_3_0_1 motif.txt graph.txt > instances.txt',
        cwd='../vflib/', shell=True)
