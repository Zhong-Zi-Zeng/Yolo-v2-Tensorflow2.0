from util.loss_function import CustomLoss
from util.tools import *
from util.dataloader import Dataloader
from util.store_tools import StoreTools
from tensorflow.keras.optimizers import Adam
from Model import Darknet19
from tqdm import tqdm
import yaml
import os

tf.random.set_seed(10)

def training():
    for e in range(config['EPOCH']):
        pbar = tqdm(range(0, len(data_loader)), ncols=120)

        for b, batch in zip(pbar, data_loader):
            x_batch, b_batch, y_batch = batch
            aug_img = augmentation(x_batch)

            with tf.GradientTape() as tape:
                y_pred = model(aug_img)
                loss = custom_loss.get_loss(y_batch, b_batch, y_pred)

            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            pbar.set_description("epoch:{} - loss:{:.4f}".format(e, loss))

            if b % 100 == 0:
                model.save_weights(config["WEIGHT"])
                store_tools.store_result(x_batch, y_pred, image_name="Epoch_{}_B_{}".format(e, b))


if __name__ == '__main__':
    # Loading config
    with open('./Config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.CLoader)

    # Loss function
    custom_loss = CustomLoss(config=config)

    # Build Model
    model = Darknet19(k=config['K'],
                      num_classes=config['NUM_CLASSES'])

    model.build(input_shape=(None, 416, 416, 3))

    # Loading weight
    if not os.path.isdir('./weight'):
        os.mkdir('./weight')

    if os.path.isfile(config["WEIGHT"]):
        print('Loading weight...')
        model.load_weights(config['WEIGHT'])

    # Optimizer
    opt = Adam(learning_rate=config['LR'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)

    # Dataloader
    data_loader = Dataloader(config=config)

    # Store tools
    store_tools = StoreTools(config=config)

    # Training
    training()
