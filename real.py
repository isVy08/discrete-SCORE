import numpy as np
import os
import pandas as pd

class RealDataset:
    def __init__(self, config_code):
        
        if config_code == 'bridge':
            from ucimlrepo import fetch_ucirepo
            # https://archive.ics.uci.edu/dataset/18/pittsburgh+bridges
            data = fetch_ucirepo(id=18) 
  
            # data (as pandas dataframes) 
            df = data.data.features 
            df = df.sample(frac=1)

            colnames = ['ERECTED', 'MATERIAL', 'SPAN', 'LANES']

            df = df.loc[:, colnames]
            df = df.dropna(inplace=False)

            for name in colnames: 
                col = df.loc[:, name]
                values = {name:i for i, name in enumerate(col.unique())}
                df.loc[:, name] = col.map(values)
            
            self.X = df
            self.B_bin = np.zeros((len(colnames), len(colnames)))
            edges = [(0,2), (1,2), (1,3)]
            for i,j in edges: 
                self.B_bin[i,j] = 1.0
        
        else: 
            from pgmpy.utils import get_example_model
            # import pdb; pdb.set_trace()
            # model = get_example_model(model=config_code)
            code = config_code[:-1]
            file_path = f'dataset/storage/{config_code}.csv'
            model = get_example_model(model=code)
            if not os.path.isfile(file_path):
                df = model.simulate(n_samples=int(5e3))
                colnames = sorted([name for name in df.columns])
                df = df[colnames]
                df.to_csv(file_path, index=False)
    
            else:
                df = pd.read_csv(file_path)
                
                
            colnames = [name for name in df.columns]
            
            B_bin = np.zeros((df.shape[1], df.shape[1]))
            for vi, vj in model.edges():
                i,j = colnames.index(vi), colnames.index(vj)
                B_bin[i,j] = 1
            
            self.B_bin = B_bin
            from sklearn.preprocessing import LabelEncoder
            self.X = df.apply(LabelEncoder().fit_transform)

        self.X = self.X.rename(columns={name: i for i, name in enumerate(colnames)})
        self.X = self.X.astype('int')
        self.num_samples, self.num_nodes = self.X.shape
        self.max_cardinality = self.X.values.max()
        self.config_code = config_code
            
   
