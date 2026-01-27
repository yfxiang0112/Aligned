'''
This script extracts specific object property restrictions from an OWL file
and finds their predecessors and successors based on a specified object property.
'''

from rdflib import Graph, URIRef, RDF, RDFS, OWL



def extract_restrictions_with_connections(owl_file, object_property_uri):
    '''
    Extracts object property restrictions and their predecessors and successors.

    Arguments:
        owl_file (str): Path to the OWL file.
        object_property_uri (str): URI of the object property for which restrictions are to be extracted.

    Returns:
        list: A list of dictionaries, each containing restriction details along with predecessors and successors.
    '''

    '''
    Load the OWL file into a graph
    '''
    g = Graph()
    g.parse(owl_file, format="xml")

    '''
    Convert the object property URI to URIRef
    '''
    object_property = URIRef(object_property_uri)

    '''
    Find all restrictions related to the specified object property
    '''
    restrictions = []
    for restriction_node, _, _ in g.triples((None, RDF.type, OWL.Restriction)):
        '''
        Check if the restriction is on the given object property
        '''
        for _, _, prop in g.triples((restriction_node, OWL.onProperty, None)):
            if prop == object_property:
                restriction_details = {'restriction': str(restriction_node), 'pred':set()}

                ''' Check for someValuesFrom restriction '''
                for _, _, value in g.triples((restriction_node, OWL.someValuesFrom, None)):
                    restriction_details['type'] = 'someValuesFrom'
                    restriction_details['value'] = str(value)

                ''' Check for allValuesFrom restriction '''
                for _, _, value in g.triples((restriction_node, OWL.allValuesFrom, None)):
                    restriction_details['type'] = 'allValuesFrom'
                    restriction_details['value'] = str(value)

                ''' Check for hasValue restriction '''
                for _, _, value in g.triples((restriction_node, OWL.hasValue, None)):
                    restriction_details['type'] = 'hasValue'
                    restriction_details['value'] = str(value)


                ''' Recursive helper function to find predecessors '''
                pattern = "http://purl.obolibrary.org/obo/"
                def traverse(term, depth, max_depth, visited):
                    '''
                    Recursively finds predecessors until the pattern is matched or max depth is reached.
                
                    Arguments:
                        term (URIRef): Current term in the traversal.
                        depth (int): Current depth of recursion.
                        visited (set): Set of visited terms to avoid cycles.
                
                    Returns:
                        list: A list of predecessors found.
                    '''
                    if depth > max_depth or term in visited:
                        return []
                
                    visited.add(term)
                
                    ''' Check if the current term matches the pattern '''
                    if pattern in str(term):
                        return [str(term)]
                
                    ''' Find direct predecessors '''
                    predecessors = []
                    for subject, _, _ in g.triples((None, None, term)):
                        predecessors += traverse(subject, depth + 1, max_depth, visited)
                
                    return predecessors

                for value in traverse(restriction_node, 0, 7, set()):
                    restriction_details['pred'].add(str(value))


                ''' Add the restriction details to the list '''
                restrictions.append(restriction_details)

    return restrictions

if __name__ == "__main__":
    owl_file_path = "rules/go-plus.owl"  # Replace with the path to your OWL file
    object_property_uri = "http://purl.obolibrary.org/obo/RO_0002211"  # Replace with the object property URI

    results = extract_restrictions_with_connections(owl_file_path, object_property_uri)

    ''' Print the extracted restrictions with predecessors and successors '''
    if results:
        print("Object Property Restrictions with Connections:")
        for result in results[:100]:
            print(f"Restriction: {result['restriction']}")
            print(f"  Type: {result['type']}")
            print(f"  Value: {result['value']}")
            print(f"  Predecessors: {result['pred']}")
            #print(f"  Successors: {result['successors']}")
        print(len(results))

        #edges = set()
        with open('rules/restrictions.csv', 'w') as f:
            for result in results:
                for pred in result['pred']:
                #edges.add((result['pred'], result['value']))
                    f.write(f'{pred}, {result["value"]}\n')

    else:
        print("No restrictions found for the specified object property.")
        
