import os, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pgmpy.utils import get_example_model

class RealDataset:
    def __init__(self, config_code):
        
        name = config_code
        file_path = f'dataset/storage/{config_code}.csv'
        model = None
        if not os.path.isfile(file_path):
            model = get_example_model(model=name)
            df = model.simulate(n_samples=int(20e3))
            colnames = sorted([name for name in df.columns])
            df = df[colnames]
            df.to_csv(file_path, index=False)

        else:
            df = pd.read_csv(file_path)
            
        colnames = [name for name in df.columns]
        
        dag_path = f'dataset/storage/dag-{name}.csv'
        if not os.path.isfile(dag_path):
            if model is None:
                model = get_example_model(model=name)
            B_bin = np.zeros((df.shape[1], df.shape[1]))
            for vi, vj in model.edges():
                i,j = colnames.index(vi), colnames.index(vj)
                B_bin[i,j] = 1
            df = pd.DataFrame(B_bin)
            df.to_csv(dag_path, header=True, index=False)
            
        else:
            B_bin = pd.read_csv(dag_path).values
        
        self.B_bin = B_bin
        
        self.X = df.apply(LabelEncoder().fit_transform)

        self.X = self.X.rename(columns={name: i for i, name in enumerate(colnames)})
        self.X = self.X.astype('int')
        self.num_samples, self.num_nodes = self.X.shape
        self.max_cardinality = self.X.values.max()
        self.config_code = config_code
            
   
