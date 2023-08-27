from util.dataloader import Dataloader
from util.store_tools import StoreTools
from Model import Darknet19
from tqdm import tqdm
import yaml
import os

def testing():
    for e in range(config['EPOCH']):
        pbar = tqdm(range(0, len(data_loader)), ncols=120)

        for b, batch in zip(pbar, data_loader):
            x_batch, b_batch, y_batch = batch
            y_pred = model(x_batch)
            store_tools.visualize(x_batch, y_pred)


if __name__ == '__main__':
    # Loading config
    with open('./Config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.CLoader)

    # Build Model
    model = Darknet19(k=config['K'],
                      num_classes=config['NUM_CLASSES'])

    model.build(input_shape=(None, 416, 416, 3))

    # Loading weight
    if os.path.isfile(config["WEIGHT"]):
        print('Loading weight...')
        model.load_weights(config['WEIGHT'])

    # Dataloader
    data_loader = Dataloader(config=config)

    # Store tools
    store_tools = StoreTools(config=config)

    # Training
    testing()
