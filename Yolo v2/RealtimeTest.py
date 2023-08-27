from Model import Darknet19
from util.store_tools import StoreTools
import numpy as np
import cv2
import yaml
import os



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

    # Store tools
    store_tools = StoreTools(config=config)

    # Realtime test
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (416, 416))
        img = np.array(img[None, ...], dtype=np.float32) / 255.
        y_pred = model(img)

        img, pred_box_x1y1, pred_box_x2y2, pred_box_class = store_tools.post_process(img, y_pred, 0)

        for box_x1y1, box_x2y2, cls in zip(pred_box_x1y1, pred_box_x2y2, pred_box_class):
            x1 = int(box_x1y1[0])
            y1 = int(box_x1y1[1])
            x2 = int(box_x2y2[0])
            y2 = int(box_x2y2[1])

            color = list(np.random.random(size=(3,)))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, config["LABELS"][int(cls)], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
