from pascal import PascalVOC
from tools import *
import cv2
import numpy as np
import os


class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, config):
        self.config = config
        self.anchors = np.array(self.config['ANCHOR_BOX'])
        self.xml_file_list = os.listdir(self.config['ANNOTATIONS_DIR'])

    def _find_best_anchor(self, target_bbox_wh) -> int:
        """
            將傳入的bbox找出與其iou最大的anchor索引
        :param target_bbox_wh: [w, h]
        :return:
            idx: 哪一個anchor與傳入的bbox有最大iou
        """
        idx = -1
        max_iou = -1

        for i in range(self.config['K']):
            anchor = self.anchors[i]
            iou = IOU(target_bbox_wh, anchor)

            if iou > max_iou:
                max_iou = iou
                idx = i

        return idx

    def __len__(self):
        return int(np.ceil(len(self.xml_file_list) / self.config['BATCH_SIZE']))

    def __getitem__(self, item):
        """
            返回bath_size張照片和label
        :return:
            x_batch: [B, H, W, 3]
            y_batch: [B, GRID_SIZE, GRID_SIZE, K, 5 + NUM_CLASSES]
                5 + NUM_CLASSES -> [cx, cy, w, h, 1 or 0, ...] cx, cy, w, h 都是在網格座標下，數值介於0~13
            b_batch: [B, 1, 1, 1, 50, 4]: 用來存放target box用，假設一張照片所有物件不超過50個
        """

        l_bound = item * self.config['BATCH_SIZE']
        r_bound = (item + 1) * self.config['BATCH_SIZE']

        if r_bound > len(self.xml_file_list):
            r_bound = len(self.xml_file_list)
            l_bound = r_bound - self.config['BATCH_SIZE']

        x_batch = np.zeros((r_bound - l_bound, 416, 416, 3))
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_SIZE'], self.config['GRID_SIZE'], self.config['K'],
                            5 + self.config['NUM_CLASSES']))
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, 50, 4))

        for b, xml_file in enumerate(self.xml_file_list[l_bound:r_bound]):
            true_box_index = 0

            # Loading annotation
            annotation = PascalVOC.from_xml(os.path.join(self.config['ANNOTATIONS_DIR'], xml_file))

            # Loading image
            img = cv2.imread(os.path.join(self.config['IMAGES_DIR'], annotation.filename))

            # Scale
            w_scale = 416. / img.shape[1]
            h_scale = 416. / img.shape[0]

            # Resize
            img = cv2.resize(img, (416, 416))

            # Normalize
            img = np.array(img, dtype=np.float32) / 255.

            # Loading bboxes and category
            for object in annotation.objects:
                # 先將框rescale
                x1 = object.bndbox.xmin * w_scale
                y1 = object.bndbox.ymin * h_scale
                x2 = object.bndbox.xmax * w_scale
                y2 = object.bndbox.ymax * h_scale

                # 再將框轉換成(Cx, Cy, w, h)後, 映射到cell grid座標
                # 每個值都介於 0 ~ 13
                cx = 0.5 * (x1 + x2)
                cx = cx * self.config['GRID_SIZE'] / 416.
                cy = 0.5 * (y1 + y2)
                cy = cy * self.config['GRID_SIZE'] / 416.

                w = (x2 - x1) * self.config['GRID_SIZE'] / 416.
                h = (y2 - y1) * self.config['GRID_SIZE'] / 416.
                box = [cx, cy, w, h]

                # 接下來要將GT的bbox看要分配給哪一個anchor box比較好
                best_anchor = self._find_best_anchor(np.array([w, h]))

                grid_x = int(cx)
                grid_y = int(cy)
                cls = self.config['LABELS'].index(object.name)
                y_batch[b, grid_y, grid_x, best_anchor, :4] = box
                y_batch[b, grid_y, grid_x, best_anchor, 4] = 1.
                y_batch[b, grid_y, grid_x, best_anchor, 5 + cls] = 1
                b_batch[b, 0, 0, 0, true_box_index] = box
                true_box_index += 1

            x_batch[b] = img

        return [x_batch, b_batch, y_batch]
