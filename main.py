
import cv2
import sys
import datetime
sys.path.append("../dx-all-suite/dx-runtime/dx_rt/python_package/src")
from dx_engine import InferenceEngine
from config import preprocess, postprocess_yolo, draw_boxes, load_config

def run_image_detection(model_path, config_path, image_path, save_path):

    cfg = load_config(config_path)
    class_names = cfg["class_names"]

    ie = InferenceEngine(model_path)
    print("Loaded model!")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image missing!")

    print("Running inference on image...")
    
    orig_h, orig_w = img.shape[:2]

    # LETTERBOX to model input size
    input_info = ie.get_input_tensors_info()[0]
    input_h = input_info["shape"][1]
    input_w = input_info["shape"][2]

    img_lb, scale, pad_w, pad_h = preprocess(img, input_w, input_h)

    print("Running inference...")
    outputs = ie.run(img_lb)  # uint8 BGR NHWC

    pred = outputs[0][0]  # shape (25200, 85)

    detections = postprocess_yolo(
        pred, orig_w, orig_h, scale, pad_w, pad_h,
        conf_thres=0.20,  # lower threshold = detect all
        iou_thresh=0.45
    )
    
    out_img = draw_boxes(img, detections, class_names)
    
    cv2.imwrite(save_path, out_img)
    cv2.imshow("output", out_img)
    print("Press 'q' or 'Esc' to close the image window.")
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27: 
        cv2.destroyAllWindows()
        
        
    print("Saved:", save_path)
    
def run_video_detection(model_path, config_path, video_save_path, video_path=0):
    # Placeholder for video/camera detection
    cfg = load_config(config_path)
    class_names = cfg["class_names"]

    ie = InferenceEngine(model_path)
    print("Loaded model!")
    
    cap=cv2.VideoCapture(0)
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=None
    fps=100
    print("Running inference on video stream...")
    print("Press 'q' or 'Esc' to stop the video stream.")

    while fps > 0:
        
        ret, frame=cap.read()
        if not ret:
            break
        orig_h, orig_w = frame.shape[:2]

        input_info = ie.get_input_tensors_info()[0]
        input_h = input_info["shape"][1]
        input_w = input_info["shape"][2]

        img_lb, scale, pad_w, pad_h = preprocess(frame, input_w, input_h)

        outputs = ie.run(img_lb)  # uint8 BGR NHWC

        pred = outputs[0][0]  # shape (25200, 85)

        detections = postprocess_yolo(
            pred, orig_w, orig_h, scale, pad_w, pad_h,
            conf_thres=0.20,  # lower threshold = detect all
            iou_thresh=0.45
        )
        
        out_img = draw_boxes(frame, detections, class_names)
        
        if out is None:
           out=cv2.VideoWriter(video_save_path,fourcc,30,(orig_w,orig_h))
        
        out.write(out_img) 
        fps-=1
        
        cv2.imshow("output", out_img)
        key = cv2.waitKey(1) & 0xFF 
        if key == ord("q") or key == 27: 
            cv2.destroyAllWindows()
            break 
        
    out.release()
    cap.release()
    print("Saved video:", video_save_path)
    

if __name__ == "__main__":

    model_path="/home/pi/dx-all-suite/dx-runtime/dx_app/assets/models/YOLOV5S-1.dxnn"
    config_path="coco_classes.json"
    image_path="/home/pi/dx-all-suite/dx-runtime/dx_app/sample/img/1.jpg"
    save_path=f"src/output/5-out.jpg"
    video_save_path=f"src/output/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
    video_path="0"
    
    while True:
        print("===============================")
        print("1. Run Object Detection on image")
        print("2. Run Object Detection on video/camera")
        print("3. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            run_image_detection(model_path, config_path, image_path, save_path)
        elif choice == "2":
            # Placeholder for video/camera detection
            run_video_detection(model_path, config_path, video_save_path, video_path,)
        elif choice == "3":
            break

