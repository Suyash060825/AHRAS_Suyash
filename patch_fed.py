import re

file_path = "federated/fed_learning.py"
with open(file_path, "r") as f:
    content = f.read()

noniid_func = """
def create_noniid_client_splits(
    records: List[Any],
    n_clients: int = 5,
    noniid_degree: float = 0.8,
) -> Dict[str, List[Any]]:
    \"\"\"
    Dirichlet distribution-based non-IID partitioning for Federated Learning evaluation.
    \"\"\"
    labels = np.array([getattr(r, 'label', 0) for r in records])
    classes = np.unique(labels)
    client_data = {f"client_{i}": [] for i in range(n_clients)}
    
    for cls in classes:
        cls_idxs = np.where(labels == cls)[0]
        # Dirichlet allocation
        proportions = np.random.dirichlet(
            alpha=np.ones(n_clients) * (1.0 - noniid_degree + 0.01),
        )
        splits = (proportions * len(cls_idxs)).astype(int)
        
        # Ensure we don't exceed bounds
        start = 0
        for i, count in enumerate(splits):
            if i == len(splits) - 1:
                # Give remaining to last client
                count = len(cls_idxs) - start
            for j in cls_idxs[start:start+count]:
                client_data[f"client_{i}"].append(records[j])
            start += count
    return client_data

class ModelUpdate:
"""

content = content.replace("class ModelUpdate:", noniid_func)

with open(file_path, "w") as f:
    f.write(content)
