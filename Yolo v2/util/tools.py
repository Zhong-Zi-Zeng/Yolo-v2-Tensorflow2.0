import copy

import numpy as np
import tensorflow as tf


def generate_cell_grid(grid_w, grid_h, batch_size, k):
    """
        生成每個cell grid所對應的座標
    :param grid_w: 特徵圖寬度
    :param grid_h: 特徵圖高度
    :param k: anchor boxes 數量
    :return: cell_grid [batch_size, grid_h, grid_w, k, 2]
    """
    cell_x = tf.range(grid_w)
    cell_y = tf.range(grid_h)
    cell_x, cell_y = tf.meshgrid(cell_x, cell_y)
    cell_y = tf.reshape(cell_y, (-1, 1))
    cell_x = tf.reshape(cell_x, (-1, 1))
    cell_grid = tf.reshape(tf.concat([cell_x, cell_y], axis=1), (grid_w, grid_h, 2))[None, :, :, None, :]
    cell_grid = tf.tile(cell_grid, [batch_size, 1, 1, k, 1])
    cell_grid = tf.cast(cell_grid, dtype=tf.float32)

    return cell_grid


def post_processing_output(pred, cell_grid, anchors):
    """
        將model輸出加到cell_grid上，再利用anchors所算出來的寬和高進行縮放
    :param pred: 模型輸出 [batch_size, grid_h, grid_w, k, 5 + num_classes], 5 -> (tx, ty, tw, th, c)
    :param cell_grid: 每個網格的x, y座標 [batch_size, grid_h, grid_w, k, 2]
    :param anchors: anchor的寬和高 [k, 2], 2 -> (anchor_w, anchor_h)
    :return
        pred_box_xy [batch_size, grid_h, grid_w, k, 2]
        pred_box_wh [batch_size, grid_h, grid_w, k, 2]
        pred_box_conf [batch_size, grid_h, grid_w, k]
        pred_box_class [batch_size, grid_h, grid_w, k, num_classes]
    """

    # x, y = sigmoid(tx, ty) + cell_grid
    pred_box_xy = tf.nn.sigmoid(pred[..., :2]) + cell_grid

    # w, h = exp(tw, th) * (anchors_w, anchors_h)
    pred_box_wh = tf.exp(pred[..., 2:4]) * anchors[None, None, None, ...]

    # predict confidence
    pred_box_conf = tf.nn.sigmoid(pred[..., 4])

    # class probabilities
    pred_box_class = pred[..., 5:]

    return (pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class)


def extract_ground_truth(y_true):
    """
        將label進行分割
    :param y_true: [B, GRID_SIZE, GRID_SIZE, K, 5 + NUM_CLASSES]
           5 + NUM_CLASSES -> [cx, cy, w, h, 1 or 0, ...] cx, cy, w, h 都是在網格座標下，數值介於0~13

    :return:
        true_box_xy: [B, GRID_SIZE, GRID_SIZE, K, 2]
        true_box_wh: [B, GRID_SIZE, GRID_SIZE, K, 2]
        true_box_conf: [B, GRID_SIZE, GRID_SIZE, K]
        true_box_class: [B, GRID_SIZE, GRID_SIZE, K] 最後包含0~num_class

    """
    true_box_xy = y_true[..., :2]
    true_box_wh = y_true[..., 2:4]
    true_box_conf = y_true[..., 4]
    true_box_class = tf.argmax(y_true[..., 5:], axis=-1)

    true_box_xy = tf.convert_to_tensor(true_box_xy, dtype=tf.float32)
    true_box_wh = tf.convert_to_tensor(true_box_wh, dtype=tf.float32)
    true_box_conf = tf.convert_to_tensor(true_box_conf, dtype=tf.float32)
    true_box_class = tf.convert_to_tensor(true_box_class, dtype=tf.int64)

    return (true_box_xy, true_box_wh, true_box_conf, true_box_class)


def IOU(boxes, anchor):
    """
        計算anchor和boxes之間的IOU，傳入的數值必須是已經將中心點移到原點處!
    :param boxes: [w, h]
    :param anchor: [w, h]
    :return:
    """

    w_min = np.minimum(boxes[0], anchor[0])
    h_min = np.minimum(boxes[1], anchor[1])
    inter = w_min * h_min

    boxes_area = boxes[0] * boxes[1]
    anchor_area = anchor[0] * anchor[1]

    union = boxes_area + anchor_area

    return inter / (union - inter)


def augmentation(image):
    """
        將圖像進行增強
    :param image: [B, H, W, 3] 數值介於0 ~ 1
    :return: aug_img: [B, H, W, 3] 數值介於0 ~ 1
    """

    aug_img = copy.deepcopy(image) * 255.

    aug_img = tf.image.random_brightness(aug_img, max_delta=0.1)
    aug_img = tf.image.random_contrast(aug_img, lower=0.8, upper=3)
    aug_img = tf.image.random_hue(aug_img, max_delta=0.2)
    aug_img = tf.image.random_saturation(aug_img, lower=0, upper=4)

    aug_img /= 255.

    return aug_img
