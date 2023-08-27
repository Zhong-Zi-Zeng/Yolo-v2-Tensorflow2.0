from tools import *
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class StoreTools:
    def __init__(self, config):
        self.config = config
        self.cell_grid = generate_cell_grid(grid_w=config['GRID_SIZE'],
                                            grid_h=config['GRID_SIZE'],
                                            batch_size=config['BATCH_SIZE'],
                                            k=config['K'])
        self.anchor_box = np.array(config['ANCHOR_BOX'])

    def post_process(self, x_batch, pred, idx):
        image = np.clip(x_batch[idx] * 255, 0, 255).astype(np.uint8)

        # post process output (x、y、w、h介於0~13)
        pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = post_processing_output(pred,
                                                                                         self.cell_grid,
                                                                                         self.anchor_box)

        # 整體信心等於 Pr(object exists) * Pr(Class | object exists)
        class_conf = pred_box_conf[idx][..., None] * tf.nn.softmax(pred_box_class[idx],
                                                                   axis=-1)  # [grid_h, grid_w, k, num_classes]

        # 找出大於閥值的anchor
        indices = tf.where(class_conf > self.config['CONF_THRESHOLD'])

        # 找出信心值有大於閥值的BBOX
        pred_box_xy = tf.gather_nd(pred_box_xy[idx], indices[..., :3])  # [N, 2]
        pred_box_wh = tf.gather_nd(pred_box_wh[idx], indices[..., :3])  # [N, 2]
        pred_box_conf = tf.gather_nd(pred_box_conf[idx], indices[..., :3])  # [N, ]
        pred_box_class = indices[..., -1]  # [N, num_classes]

        # 將座標轉換到像素座標
        pred_box_xy *= 416 / 13.
        pred_box_wh *= 416 / 13.

        # (cx, cy, w, h) -> (x1, y1, x2, y2)
        pred_box_x1y1 = pred_box_xy - pred_box_wh / 2.
        pred_box_x2y2 = pred_box_xy + pred_box_wh / 2.
        pred_box_x1y1x2y2 = tf.concat([pred_box_x1y1, pred_box_x2y2], axis=-1)  # [N, 4]

        # NMS
        nms_indices = tf.image.non_max_suppression(boxes=pred_box_x1y1x2y2,
                                                   scores=pred_box_conf,
                                                   max_output_size=self.config['NMS_MAX_OUTPUT_SIZE'],
                                                   iou_threshold=self.config['NMS_IOU_THRESHOLD'])

        pred_box_x1y1 = tf.gather(pred_box_x1y1, nms_indices)
        pred_box_x2y2 = tf.gather(pred_box_x2y2, nms_indices)
        pred_box_class = tf.gather(pred_box_class, nms_indices)

        return image, pred_box_x1y1, pred_box_x2y2, pred_box_class
    def _render(self, image, pred_box_x1y1, pred_box_x2y2, pred_box_class):
        # 繪製
        plt.figure(figsize=(16, 10))
        plt.imshow(image[..., ::-1])
        ax = plt.gca()

        for box_x1y1, box_x2y2, cls in zip(pred_box_x1y1, pred_box_x2y2, pred_box_class):
            x1 = int(box_x1y1[0])
            y1 = int(box_x1y1[1])
            x2 = int(box_x2y2[0])
            y2 = int(box_x2y2[1])

            color = list(np.random.random(size=(3,)))

            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, color=color, linewidth=3))

            text = f'{self.config["LABELS"][int(cls)]}'
            ax.text(x1, y1, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')

    def visualize(self, x_batch, pred):
        """
        :param x_batch: [B, 416, 416, 3] (0 ~ 1)
        :param pred: [B, 13, 13, 5 + num_classes]
        """
        for b in range(self.config['BATCH_SIZE']):
            image, pred_box_x1y1, pred_box_x2y2, pred_box_class = self.post_process(x_batch, pred, idx=b)
            self._render(image, pred_box_x1y1, pred_box_x2y2, pred_box_class)
            plt.show()

    def store_result(self, x_batch, pred, image_name):
        """
        :param x_batch: [B, 416, 416, 3] (0 ~ 1)
        :param pred: [B, 13, 13, 5 + num_classes]
        :param image_name: The name of the image.
        """

        image, pred_box_x1y1, pred_box_x2y2, pred_box_class = self.post_process(x_batch, pred, idx=0)
        self._render(image, pred_box_x1y1, pred_box_x2y2, pred_box_class)
        plt.savefig(os.path.join(self.config["RESULT_DIR"], image_name + ".png"))
        plt.close()
        plt.clf()
