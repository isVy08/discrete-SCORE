from synthetic import SyntheticDataset

def get_data_config(config_id, graph_type, seed, max_card, degree):
    num_nodes = [5, 10, 15, 20]
    
    config = {
        'n': 10000, 
        'max_card': max_card,
        'min_card': 3,
        'graph_type': graph_type,
        'degree': degree, 
    }

    i = int(str(config_id)[-1])
    config['d'] = num_nodes[i - 1]
    config['code'] = f'{graph_type}{config_id}-S{seed}'
    return config



def get_data(root, config_id, graph_type, seed, max_card, degree):

    config_id = int(config_id)

    config = get_data_config(config_id, graph_type, seed, max_card, degree)

    dataset = SyntheticDataset(root = root, config_code = config['code'],
                            num_samples = config['n'], 
                            num_nodes = config['d'], 
                            max_cardinality = config['max_card'],
                            min_cardinality = config['min_card'],
                            graph_type = config['graph_type'],
                            degree = config['degree']
                    )
    
    return dataset

if __name__ == "__main__":

    get_data('dataset', '1', "ER", seed=0, max_card=6, degree=2)
        