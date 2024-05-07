import datasets


class Apps():

    def __init__(self, split: str = 'test', **kwargs) -> None:
        self.dataset = datasets.load_dataset('apps', split=split, **kwargs)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        dp = self.dataset[idx]
        question = dp['question'].split()