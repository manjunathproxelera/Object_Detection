import cv2, json, numpy as np


def preprocess(img, new_w, new_h, color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_shape=new_h

    scale = min(new_shape / h, new_shape / w)
    nw, nh = int(w * scale), int(h * scale)

    img_resized = cv2.resize(img, (nw, nh))

    # padding
    pad_w = (new_shape - nw) // 2
    pad_h = (new_shape - nh) // 2

    img_padded = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    img_padded[pad_h:pad_h + nh, pad_w:pad_w + nw] = img_resized

    return img_padded, scale, pad_w, pad_h


def postprocess_yolo(pred, orig_w, orig_h, scale, pad_w, pad_h,
                     conf_thres=0.20, iou_thresh=0.45):

    boxes = pred[:, :4]
    obj_conf = pred[:, 4]
    cls_scores = pred[:, 5:]

    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(len(cls_scores)), cls_ids]

    final_conf = obj_conf * cls_conf

    mask = final_conf >= conf_thres
    boxes = boxes[mask]
    final_conf = final_conf[mask]
    cls_ids = cls_ids[mask]

    if len(boxes) == 0:
        return []

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    x1 -= pad_w
    y1 -= pad_h
    x2 -= pad_w
    y2 -= pad_h

    x1 /= scale
    y1 /= scale
    x2 /= scale
    y2 /= scale

    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    nms_boxes = []
    for b in boxes_xyxy:
        nms_boxes.append([int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])])

    indices = cv2.dnn.NMSBoxes(nms_boxes, final_conf.tolist(),
                               conf_thres, iou_thresh)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                "box": boxes_xyxy[i],
                "score": float(final_conf[i]),
                "class_id": int(cls_ids[i])
            })

    return detections


def draw_boxes(img, detections, class_names):
    for det in detections:
        x1, y1, x2, y2 = det["box"].astype(int)
        score = det["score"]
        cls_id = det["class_id"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{class_names[cls_id]} {score:.2f}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)
    return img


def load_config(path):
    return json.load(open(path))

