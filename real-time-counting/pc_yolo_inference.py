import cv2
import torch
import time
import utils.utils
import model.detector

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit(1)
    
    # Load configuration and model
    cfg = utils.utils.load_datafile('./data/coco.data')
    model_path = 'modelzoo/yolofv2-nano-200-epoch-0.978090ap-model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    yolo_model.load_state_dict(torch.load(model_path, map_location=device))
    yolo_model.eval()
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        LABEL_NAMES = [line.strip() for line in f.readlines()]

    threshold = 0.6
    current_time = 0
    last_time = 0
    INTERVAL = 0

    try:
        while True:
            ret, frame = cap.read()
            current_time = time.time()

            if current_time - last_time >= INTERVAL:

                # Resize frame to model input size
                res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
                img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
                img = torch.from_numpy(img.transpose(0, 3, 1, 2)).to(device).float() / 255.0

                # Model inference
                start = time.perf_counter()
                preds = yolo_model(img)
                end = time.perf_counter()
                print(f"Inference time: {(end - start) * 1000:.2f} ms")
                # check_resources()

                # Procescsi_data_ready_events predictions
                output = utils.utils.handel_preds(preds, cfg, device)
                output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

                h, w, _ = frame.shape
                scale_h, scale_w = h / cfg["height"], w / cfg["width"]
                cv_count = 0
                # Draw bounding boxes
                for box in output_boxes[0]:
                    box = box.tolist()
                    obj_score = box[4]
                    category = LABEL_NAMES[int(box[5])]

                    if category == 'person' and obj_score >= threshold:
                        cv_count += 1
                        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, f'{category} {obj_score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"CV count: {cv_count}")
                last_time = current_time

            cv2.putText(frame, f"CV count: {cv_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display result
            cv2.imshow("CSI Camera Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # time.sleep(2)

        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("Stopping YOLO inference")