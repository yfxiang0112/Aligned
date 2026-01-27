import json
import pygraphblas as pgb
from tqdm import tqdm
from rdflib import Graph, RDFS, OWL, URIRef

class Rule():
    def __init__(self, owl_path, input_path, annotation_path):
        self.owl_path = owl_path
        self.input_path = input_path
        self.annotation_path = annotation_path
        self.nodes = set()
        self.edges = set()
        
        ''' read input & annotation json file '''
        with open(input_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                self.input = data
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                self.annotation = data

        ''' get domain of terms, from input and annotation '''
        self.domain = set()
        for _, terms in self.input.items():
            for term in terms:
                self.domain.add(term)
        for _, terms in self.annotation.items():
            for term in terms:
                self.domain.add(term)

    def read_owl(self):
        ''' Load ontology '''
        self.ontology = Graph()
        self.ontology.parse(self.owl_path, format='xml')

    def vertical(self):

        ''' Find subclasses and superclasses for each term '''
        for term in self.domain:
                ''' add term to set, and skip if current term is added '''
                if term in self.nodes:
                    continue
                self.nodes.add(term)

                ''' Check all subclasses of the term '''
                for subclass in self.ontology.subjects(predicate=RDFS.subClassOf, object=URIRef(term)):
                    self.nodes.add(subclass)
                    self.edges.add((subclass, term))
        
                ''' Check all superclasses of the term '''
                for superclass in self.ontology.objects(subject=URIRef(term), predicate=RDFS.subClassOf):
                    self.nodes.add(superclass)
                    self.edges.add((term, superclass))

        print(len(self.nodes))
        return self.nodes

    def horizontal(self, restriction_path):
        with open(restriction_path, 'r') as f:
            line = f.read()
            while line:
                nodes = line.split(', ')
                self.nodes.add(nodes[0])
                self.nodes.add(nodes[1])
                self.edges.add((nodes[0], nodes[1]))
                line = f.read()



    def to_csv(self, node_path, edge_path):
        with open(node_path, 'w') as f:
            for node in self.nodes:
                f.write(f'{node}\n')
        with open(edge_path,'w') as f:
            for pred, succ in self.edges:
                f.write(f'{pred}, {succ}\n')

    def to_prolog(self, kb_path):
        with open(kb_path, 'w') as f:
            for node in self.nodes:
                f.write(f'node("{node}").\n')
            for pred, succ in self.edges:
                f.write(f'edge("{pred}","{succ}").\n')

    def to_pmatrix(self, kb_path, node_idx_path):
        ''' generate term - node index mapping '''
        node_idx = {}
        for idx, node in enumerate(self.nodes):
            node_idx.update({node: idx})

        ''' initialize a pgb matrix '''
        num_nodes = len(self.nodes)
        matrix = pgb.Matrix.sparse(pgb.types.BOOL, num_nodes, num_nodes)
        for src, dst in self.edges:
            matrix[node_idx[src], node_idx[dst]] = True

        ''' save node-index mapping & matrix '''
        matrix.to_binfile(kb_path)
        with open(node_idx_path, 'w') as f:
            json.dump(node_idx, f, indent=4)

if __name__ == '__main__':
    ''' File paths (replace with your actual file paths) '''
    owl_file = 'rules/go.owl'
    input_file = 'dataset/data2concepts.json'
    annotation_file = 'rules/annotations.json'

    ruleset = Rule(owl_file, input_file, annotation_file)
    ruleset.read_owl()
    print('----------  OWL loading completed  ----------')
    ruleset.vertical()

    ruleset.horizontal('rules/restrictions.csv')
    #ruleset.to_csv('rules/nodes.csv', 'rules/edges.csv')
    #ruleset.to_prolog('rules/matrix.pl')
    ruleset.to_pmatrix('rules/matrix_bin', 'rules/node_idx.json')

    #object_property_uri = 'http://purl.obolibrary.org/obo/RO_0002211'  # Replace with the object property URI
    #max_depth = 1  # Set the maximum recursion depth

    #results = ruleset.find_horizontal( object_property_uri, max_depth)

    #print(len(results))
