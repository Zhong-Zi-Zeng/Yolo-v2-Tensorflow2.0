from tools import *
import tensorflow as tf

"""
    All losses formula are referred from:
        https://farm8.staticflickr.com/7904/45592720625_821897e898_b.jpg
"""


class CustomLoss:
    def __init__(self, config):
        self.config = config
        self.cell_grid = generate_cell_grid(grid_w=config['GRID_SIZE'],
                                            grid_h=config['GRID_SIZE'],
                                            batch_size=config['BATCH_SIZE'],
                                            k=config['K'])

        self.anchor_box = np.array(config['ANCHOR_BOX'])

    def get_loss(self, y_batch, b_batch, y_pred):
        """
        :param y_batch: [B, GRID_SIZE, GRID_SIZE, K, 5 + NUM_CLASSES]
        :param b_batch : [B, 1, 1, 1, 50, 4]
        :param y_pred: [B, GRID_SIZE, GRID_SIZE, K, 5 + NUM_CLASSES]
        """

        pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = post_processing_output(y_pred, self.cell_grid,
                                                                                         self.anchor_box)
        true_box_xy, true_box_wh, true_box_conf, true_box_class = extract_ground_truth(y_batch)

        loss_cord = self._loss_xywh(true_box_conf, self.config['LAMBDA_COORD'], true_box_xy, pred_box_xy, true_box_wh,
                                    pred_box_wh)
        loss_cls = self._loss_class(true_box_conf, self.config['LAMBDA_CLASS'], true_box_class, pred_box_class)
        loss_conf = self._loss_confidence(self.config['LAMBDA_NO_OBJECT'], self.config['LAMBDA_OBJECT'], b_batch,
                                          pred_box_xy,
                                          pred_box_wh, pred_box_conf, true_box_xy, true_box_wh, true_box_conf)

        return loss_cord + loss_cls + loss_conf

    def _loss_xywh(self, true_box_conf, LAMBDA_COORD, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh):
        """
            Calculate coordinate loss.

        :param true_box_conf: [B, GRID_SIZE, GRID_SIZE, K]
        :param LAMBDA_COORD: loss weight
        :param true_box_xy: [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param pred_box_xy [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param true_box_wh:  [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param pred_box_wh:  [B, GRID_SIZE, GRID_SIZE, K, 2]
        """

        nb_coord_box = tf.reduce_sum(true_box_conf)

        xy_loss = tf.square(true_box_xy - pred_box_xy)  # [B, GRID_SIZE, GRID_SIZE, K, 2]
        wh_loss = tf.square(tf.sqrt(true_box_wh) - tf.sqrt(pred_box_wh))  # [B, GRID_SIZE, GRID_SIZE, K, 2]

        xywh_loss = tf.reduce_sum(xy_loss, axis=-1) + tf.reduce_sum(wh_loss, axis=-1)  # [B, GRID_SIZE, GRID_SIZE, K]
        xywh_loss = tf.reduce_sum(LAMBDA_COORD * xywh_loss * true_box_conf)
        xywh_loss /= (nb_coord_box + 1e-6)

        return xywh_loss

    def _loss_class(self, true_box_conf, LAMBDA_CLASS, true_box_class, pred_box_class):
        """
                Calculate class loss.

        :param true_box_conf: [B, GRID_SIZE, GRID_SIZE, K]
        :param LAMBDA_CLASS: loss weight
        :param true_box_class: [B, GRID_SIZE, GRID_SIZE, K] 最後包含0~num_class
        :param pred_box_class: [B, GRID_SIZE, GRID_SIZE, K, num_classes]
        """
        loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class,
                                                                  logits=pred_box_class)  # [B, GRID_SIZE, GRID_SIZE, K]

        nb_coord_box = tf.reduce_sum(true_box_conf)

        loss_cls = tf.reduce_sum(LAMBDA_CLASS * loss_cls * true_box_conf)
        loss_cls /= (nb_coord_box + 1e-6)

        return loss_cls

    def _assigned_confidence_loss(self, LAMBDA_OBJECT,
                                  pred_box_xy,
                                  pred_box_wh,
                                  pred_box_conf,
                                  true_box_xy,
                                  true_box_wh,
                                  true_box_conf,
                                  use_iou=False):
        """
            原本在標註label時，有決定存在物體的cell grid中，哪一個anchor要負責預測有物體
        :param LAMBDA_OBJECT: object loss weight
        :param true_box_conf: [B, GRID_SIZE, GRID_SIZE, K]
        :param pred_box_xy: [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param pred_box_wh: [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param true_box_xy: [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param true_box_wh: [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param use_iou: 是否要利用IOU來計算confidence損失
        """
        if use_iou:
            # 先將所有box的座標轉換成x1, y1, x2, y2
            pred_box_x1y1 = pred_box_xy - pred_box_wh / 2  # [B, GRID_SIZE, GRID_SIZE, K, 2]
            pred_box_x2y2 = pred_box_xy + pred_box_wh / 2  # [B, GRID_SIZE, GRID_SIZE, K, 2]

            true_box_x1y1 = true_box_xy - true_box_wh / 2  # [B, GRID_SIZE, GRID_SIZE, K, 2]
            true_box_x2y2 = true_box_xy + true_box_wh / 2  # [B, GRID_SIZE, GRID_SIZE, K, 2]

            # 計算iou
            lt = tf.maximum(pred_box_x1y1, true_box_x1y1)  # [B, GRID_SIZE, GRID_SIZE, K, 2]
            rb = tf.minimum(pred_box_x2y2, true_box_x2y2)  # [B, GRID_SIZE, GRID_SIZE, K, 2]
            wh = tf.maximum(0., rb - lt)  # [B, GRID_SIZE, GRID_SIZE, K, 2]
            inter = wh[..., 0] * wh[..., 1]  # [B, GRID_SIZE, GRID_SIZE, K]

            pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]  # [B, GRID_SIZE, GRID_SIZE, K]
            true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]  # [B, GRID_SIZE, GRID_SIZE, K]

            union = pred_box_area + true_box_area - inter  # [B, GRID_SIZE, GRID_SIZE, K]
            iou = inter / union  # [B, GRID_SIZE, GRID_SIZE, K]

            loss = tf.reduce_sum(tf.square(iou - pred_box_conf) * true_box_conf)
        else:
            loss = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * true_box_conf)

        loss *= LAMBDA_OBJECT

        return loss

    def _no_obj_confidence_loss(self, LAMBDA_NO_OBJECT,
                                pred_box_conf,
                                true_box_conf,
                                pred_box_xy,
                                pred_box_wh,
                                b_batch):
        """
        :param LAMBDA_NO_OBJECT: non object weight
        :param pred_box_conf: [B, GRID_SIZE, GRID_SIZE, K]
        :param true_box_conf: [B, GRID_SIZE, GRID_SIZE, K]
        :param pred_box_xy: [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param pred_box_wh: [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param b_batch: [B, 1, 1, 1, 50, 4]
        """

        true_xy = b_batch[..., :2]  # [B, 1, 1, 1, 50, 2]
        true_wh = b_batch[..., 2:]  # [B, 1, 1, 1, 50, 2]

        # 把每一個anchor都去和true box去算iou
        pred_box_x1y1 = pred_box_xy - pred_box_wh / 2  # [B, GRID_SIZE, GRID_SIZE, K, 2]
        pred_box_x2y2 = pred_box_xy + pred_box_wh / 2  # [B, GRID_SIZE, GRID_SIZE, K, 2]
        true_box_x1y1 = true_xy - true_wh / 2  # [B, 1, 1, 1, 50, 2]
        true_box_x2y2 = true_xy + true_wh / 2  # [B, 1, 1, 1, 50, 2]

        pred_box_x1y1 = tf.expand_dims(pred_box_x1y1, 4)  # [B, GRID_SIZE, GRID_SIZE, K, 1, 2]
        pred_box_x2y2 = tf.expand_dims(pred_box_x2y2, 4)  # [B, GRID_SIZE, GRID_SIZE, K, 1, 2]

        # 計算iou
        lt = tf.maximum(pred_box_x1y1, true_box_x1y1)  # [B, GRID_SIZE, GRID_SIZE, K, 50, 2]
        rb = tf.minimum(pred_box_x2y2, true_box_x2y2)  # [B, GRID_SIZE, GRID_SIZE, K, 50, 2]
        wh = tf.maximum(0., rb - lt)  # [B, GRID_SIZE, GRID_SIZE, K, 50, 2]
        inter = wh[..., 0] * wh[..., 1]  # [B, GRID_SIZE, GRID_SIZE, K, 50]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]  # [B, GRID_SIZE, GRID_SIZE, K]
        true_box_area = true_wh[..., 0] * true_wh[..., 1]  # [B, 1, 1, 1, 50]

        pred_box_area = tf.expand_dims(pred_box_area, axis=4)  # [B, GRID_SIZE, GRID_SIZE, K, 1]
        union = pred_box_area + true_box_area - inter  # [B, GRID_SIZE, GRID_SIZE, K, 50]
        iou = inter / union  # [B, GRID_SIZE, GRID_SIZE, K, 50]

        # 把每一個anchor最大的iou找出來，如果最大的那個iou > 0.6的話，就不用算non object loss
        max_iou = tf.reduce_max(iou, axis=-1)  # [B, GRID_SIZE, GRID_SIZE, K]
        no_obj_mask = (max_iou < 0.6)  # [B, GRID_SIZE, GRID_SIZE, K]

        # 找出要算non object的anchor後，要記的把原本指定要計算物件的那個anchor box也去掉
        # 不然有可能那個anchor與target box的iou < 0.6，同時計算了有物件的損失和沒有物件的損失
        no_obj_mask = tf.cast(no_obj_mask, tf.float32) * (1 - true_box_conf)
        loss = tf.reduce_sum(tf.square(pred_box_conf) * no_obj_mask)
        loss *= LAMBDA_NO_OBJECT

        return loss, no_obj_mask

    def _loss_confidence(self, LAMBDA_NO_OBJECT,
                         LAMBDA_OBJECT,
                         b_batch,
                         pred_box_xy,
                         pred_box_wh,
                         pred_box_conf,
                         true_box_xy,
                         true_box_wh,
                         true_box_conf):
        """
            loss_confidence會分為2個部分:
                1. 原本在標註label時，有決定存在物體的cell grid中，哪一個anchor要負責預測有物體
                2. 計算沒有包含物件的anchor的non object損失，但是如果這個anchor與某一個target box的iou > 0.6，這個anchor就不計算任何損失

        :param LAMBDA_NO_OBJECT: non object weight
        :param LAMBDA_OBJECT: object loss weight
        :param b_batch: [B, 1, 1, 1, 50, 4] 包含所有target box
        :param pred_box_xy: [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param pred_box_wh: [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param pred_box_conf: [B, GRID_SIZE, GRID_SIZE, K]
        :param true_box_xy: [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param true_box_wh: [B, GRID_SIZE, GRID_SIZE, K, 2]
        :param true_box_conf: [B, GRID_SIZE, GRID_SIZE, K]
        """

        assigned_conf_loss = self._assigned_confidence_loss(LAMBDA_OBJECT,
                                                            pred_box_xy,
                                                            pred_box_wh,
                                                            pred_box_conf,
                                                            true_box_xy,
                                                            true_box_wh,
                                                            true_box_conf)

        no_obj_loss, no_obj_mask = self._no_obj_confidence_loss(LAMBDA_NO_OBJECT,
                                                                pred_box_conf,
                                                                true_box_conf,
                                                                pred_box_xy,
                                                                pred_box_wh,
                                                                b_batch)

        N = tf.reduce_sum(true_box_conf + no_obj_mask)
        loss = (assigned_conf_loss + no_obj_loss) / (N + 1e-6)

        return loss
