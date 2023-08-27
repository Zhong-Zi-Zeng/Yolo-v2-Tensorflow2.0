from matplotlib.patches import Rectangle
from pascal import PascalVOC
from pycocotools.coco import COCO
from tools import IOU
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import random
import copy

random.seed(10)
np.random.seed(10)


def get_voc_bbox_wh():
    """
        讀取voc2007資料集中的所有bbox，並只取出每個bbox的w和h後做正規化
    :return: bbox_list [N, 2]
             N: 總共有幾個bbox
             2: (w_norm, h_norm)
    """
    bbox_list = []

    for xml_file in os.listdir(config["ANNOTATIONS_DIR"]):
        annotation = PascalVOC.from_xml(os.path.join(config["ANNOTATIONS_DIR"], xml_file))
        img_width = annotation.size.width
        img_height = annotation.size.height

        for bbox in annotation.objects:
            x1 = bbox.bndbox.xmin
            y1 = bbox.bndbox.ymin
            x2 = bbox.bndbox.xmax
            y2 = bbox.bndbox.ymax

            w_norm = (x2 - x1) / img_width
            h_norm = (y2 - y1) / img_height
            bbox_list.append([w_norm, h_norm])

    return bbox_list


def get_coco_bbox_wh():
    """
            讀取coco資料集中的所有bbox，並只取出每個bbox的w和h後做正規化
        :return: bbox_list [N, 2]
                 N: 總共有幾個bbox
                 2: (w_norm, h_norm)
        """
    coco = COCO(config["ANNOTATIONS_DIR"])
    bbox_list = []

    for img_id in coco.getImgIds():
        annotation = coco.getAnnIds(img_id)
        img_info = coco.loadImgs(img_id)[0]

        img_width = img_info['width']
        img_height = img_info['height']

        for ann_id in annotation:
            bbox = coco.loadAnns(ann_id)[0]["bbox"]

            w_norm = bbox[2] / img_width
            h_norm = bbox[3] / img_height
            bbox_list.append([w_norm, h_norm])

    return bbox_list



def K_means(boxes, K, max_iter):
    """
        執行K-means演算法，找出最適當的anchor大小
    :param boxes: 資料集中全部的bbox的w, h [N, 2]
    :param K: 需要幾個簇類
    :param max_iter: 最大循環次數，超過則停止尋找
    :return: anchor [K, 2]
    """

    # 先隨機從boxes裡面取出K個當作anchor
    anchors = random.sample(boxes, K)

    # 把boxes、anchor轉成array
    anchors = np.array(anchors)  # [K, 2]
    boxes = np.array(boxes)  # [N, 2]

    # 紀錄迭代次數
    step = 0

    # 紀錄每次對應的boxes索引，如果新索引沒有變動代表找到最佳解了
    old_idx = None

    # 可視化用
    fig = plt.figure(figsize=(8, 6))

    while True:
        step += 1

        # 計算現在的anchor與所有boxes之間的IOU
        iou = IOU(boxes, anchors)

        # 一般聚類都是以距離來計算，所以是找最小的距離去更新
        # 但是如果是iou的話，則是找最大iou的去更新
        new_idx = np.argmax(iou, axis=1)

        # 列印目前平均iou
        print('Step:{} | Average Iou:{:.3f}'.format(step, np.mean(np.max(iou, axis=1))))

        # 可視化
        ax = fig.add_subplot()
        visualize_anchors(copy.deepcopy(anchors), ax)

        # 判斷有沒有更舊的索引一樣，如果一樣就停止
        if (new_idx == old_idx).all():
            return anchors

        # 更新anchor
        for i in range(K):
            anchors[i] = np.mean(boxes[i == new_idx], axis=0)

        # 更新舊索引
        old_idx = new_idx

        # 如果超過迭代次數就停止
        if step > max_iter:
            return anchors


def visualize_anchors(anchors, ax):
    w_img, h_img = 600, 600

    anchors[:, 0] *= w_img
    anchors[:, 1] *= h_img
    anchors = np.round(anchors).astype(np.int32)

    rects = np.empty((config['K'], 4), dtype=np.int32)
    for i in range(len(anchors)):
        w, h = anchors[i]
        x1, y1 = -(w // 2), -(h // 2)
        rects[i] = [x1, y1, w, h]

    for rect in rects:
        x1, y1, w, h = rect
        rect1 = Rectangle((x1, y1), w, h, color='royalblue', fill=False, linewidth=2)
        ax.add_patch(rect1)
    plt.xlim([-(w_img // 2), w_img // 2])
    plt.ylim([-(h_img // 2), h_img // 2])

    plt.pause(0.1)


if __name__ == "__main__":
    # Loading Config
    with open('../Config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.CLoader)

    assert config["DATASET"] in ["COCO", "VOC2007"], 'Can\'t find {} dataset, only allow COCO or VOC2007'.format(

        config["DATASET"])

    if config["DATASET"] == 'COCO':
        print('Loading COCO dataset...')
        boxes_ = get_coco_bbox_wh()

    elif config["DATASET"] == 'VOC2007':
        print('Loading VOC2007 dataset...')
        boxes_ = get_voc_bbox_wh()

    print('Start K-means...')
    anchors = K_means(boxes_, K=config['K'], max_iter=config['MAX_ITER'])

    print('Result anchors:')
    print(anchors)
    plt.show()
    plt.ioff()
