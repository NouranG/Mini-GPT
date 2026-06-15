from torch.utils.data import DataLoader, dataloader
 
        
def create_dataloader(
    dataset,
    batch_size,
    shuffle=True
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )