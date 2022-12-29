import os

import cv2
import torch

if __name__ == "__main__":

    video = 'P026_tissue1.wmv'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device", device)

    model = torch.hub.load('yolov5', 'custom', path='models/best.pt', source='local')
    model.cuda()

    # maximum number of detections per image
    model.max_det = 2

    # Automatic Mixed Precision (AMP) inference
    model.amp = True

    print()

        vid_name = vid.split(".")[0]
        print(f"Labeling video: {vid_name}")

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter("videos/" + vid_name + "_labeled.wmv", fourcc, 30.0, (640, 480))
        cap = cv2.VideoCapture("videos/" + vid)

        if not cap.isOpened():
            print("Error opening video stream or file")

        left_running_average = []
        right_running_average = []

        window_span = 10
        final_predictions = {"Left_pred_after": [], "Right_pred_after": [], "Left_gt": [], "Right_gt": [],
                             "Right_pred_before": [],
                         "Left_pred_before": []}

    i = 0
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_res = model(frame)
            bbox = frame_res.pandas().xyxy[0][['xmin', 'ymin', 'xmax', "ymax"]].astype(int).values.tolist()
            classes = frame_res.pandas().xyxy[0][['confidence', 'name']].values.tolist()
            results = frame_res.pandas().xyxy[0]
            before_labels = [bla[1] for bla in enumerate(classes)]

            try:
                final_predictions['Right_pred_before'].append(
                    [rlabel[1] for rlabel in before_labels if "Right" in rlabel[1]][0])
            except Exception:
                final_predictions['Right_pred_before'].append('0')
            try:
                final_predictions['Left_pred_before'].append(
                    [rlabel[1] for rlabel in before_labels if "Left" in rlabel[1]][0])
            except Exception:
                final_predictions['Left_pred_before'].append('0')

            if len(results) == 0:
                right_running_average.append('0')
                left_running_average.append('0')
                right_box = -1
                left_box = 0
            elif len(results) == 1:
                pred = results['name'].iloc[0]

                if 'Right' in pred:
                    right_running_average.append(pred)
                    left_running_average.append('0')
                    right_box = 0
                    left_box = -1
                else:
                    right_running_average.append('0')
                    left_running_average.append(pred)
                    right_box = -1
                    left_box = 0
            else:
                right_running_average.append(results[results['xmin'] == min(results['xmin'])]['name'].iloc[0])
                right_box = list(results[results['xmin'] == min(results['xmin'])].index)[0]
                left_running_average.append(results[results['xmin'] == max(results['xmin'])]['name'].iloc[0])
                left_box = list(results[results['xmin'] == max(results['xmin'])].index)[0]

            if i < window_span:
                left_label = left_running_average[-1] if left_running_average[-1] != '0' else 'Left_Empty'
                right_label = right_running_average[-1] if right_running_average[-1] != '0' else 'Right_Empty'
                final_predictions['Left_pred_after'].append(left_label)
                final_predictions['Right_pred_after'].append(right_label)

            if len(right_running_average) == window_span:
                left_label = max(set(left_running_average), key=left_running_average.count)
                right_label = max(set(right_running_average), key=right_running_average.count)
                final_predictions['Left_pred_after'].append(left_label)
                final_predictions['Right_pred_after'].append(right_label)
                right_running_average = right_running_average[1:]
                left_running_average = left_running_average[1:]

            for t in vid_right_labels:
                if int(t[0]) <= i <= int(t[1]):
                    final_predictions['Right_gt'].append(tool_usage_right[t[2].strip()])

            for t in vid_left_labels:
                if int(t[0]) <= i <= int(t[1]):
                    final_predictions['Left_gt'].append(tool_usage_left[t[2].strip()])

            classes = [f"{label}" for conf, label in classes]
            frame = bbv.draw_multiple_rectangles(frame, bbox, bbox_color=(0, 255, 0))
            frame = bbv.bbox_visualizer.add_multiple_T_labels(frame, classes, bbox, draw_bg=True,
                                                              text_bg_color=(255, 255, 255), text_color=(0, 0, 0))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            color = (255, 0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            fontScale = 0.5
            org = (10, 15)
            gt_text = f"Ground truth: {final_predictions['Right_gt'][-1]} and {final_predictions['Left_gt'][-1]}"
            pred_text = f"Prediction: {final_predictions['Right_pred_after'][-1]} and {final_predictions['Left_pred_after'][-1]}"

            draw_text(frame, gt_text, pos=(10, 15), font_scale=1, font_thickness=thickness)
            draw_text(frame, pred_text, pos=(10, 30), font_scale=1, font_thickness=thickness)
            out.write(frame)
            # if i == 500:
            #     break
            # if cv2.waitKey(33) & 0xFF == ord('q'):
            #     print("waiting")
            #     break
        else:
            # print("Im here")
            break

    cap.release()
    out.release()
    # Closes all the frames
    # cv2.destroyAllWindows()
    # left_gt = [label_to_idx[label] for label in final_predictions['Left_gt']]
    # right_gt = [label_to_idx[label] for label in final_predictions['Right_gt']]
    # left_before = [label_to_idx[label] for label in final_predictions['Left_pred_before']]
    # right_before = [label_to_idx[label] for label in final_predictions['Right_pred_before']]
    # left_after = [label_to_idx[label] for label in final_predictions['Left_pred_after']]
    # right_after = [label_to_idx[label] for label in final_predictions['Right_pred_after']]
    print(vid_name + " Predictions:")
    print("\tRight predictions")
    precision, recall, fscore, _ = precision_recall_fscore_support(final_predictions['Right_gt'],
                                                                   final_predictions['Right_pred_before'],
                                                                   labels=labels, average=None)
    precision = pd.DataFrame(precision.reshape(1, -1), columns=labels)
    precision.to_csv("precision_right_before.csv")
    recall = pd.DataFrame(recall.reshape(1, -1), columns=labels)
    recall.to_csv("recall_right_before.csv")
    fscore = pd.DataFrame(fscore.reshape(1, -1), columns=labels)
    fscore.to_csv("f1_right_before.csv")

    print(f"\t\tPrecision before smoothing: {precision}")
    print("############################################")
    print(f"\t\tRecall before smoothing: {recall}")
    print("############################################")
    print(f"\t\tF1 before smoothing: {fscore}")
    print("############################################")
    print()
    precision, recall, fscore, _ = precision_recall_fscore_support(final_predictions['Right_gt'],
                                                                   final_predictions['Right_pred_after'],
                                                                   labels=labels, average=None)
    precision = pd.DataFrame(precision.reshape(1, -1), columns=labels)
    precision.to_csv("precision_right_after.csv")
    recall = pd.DataFrame(recall.reshape(1, -1), columns=labels)
    recall.to_csv("recall_right_after.csv")
    fscore = pd.DataFrame(fscore.reshape(1, -1), columns=labels)
    fscore.to_csv("f1_right_after.csv")
    print(f"\t\tPrecision after smoothing: {precision}")
    print("############################################")
    print(f"\t\tRecall after smoothing: {recall}")
    print("############################################")
    print(f"\t\tF1 after smoothing: {fscore}")
    print("############################################")
    print()
    print("\tLeft predictions")
    precision, recall, fscore, _ = precision_recall_fscore_support(final_predictions['Left_gt'],
                                                                   final_predictions['Left_pred_before'],
                                                                   labels=labels, average=None)
    precision = pd.DataFrame(precision.reshape(1, -1), columns=labels)
    precision.to_csv("precision_left_before.csv")
    recall = pd.DataFrame(recall.reshape(1, -1), columns=labels)
    recall.to_csv("recall_left_before.csv")
    fscore = pd.DataFrame(fscore.reshape(1, -1), columns=labels)
    fscore.to_csv("f1_left_before.csv")
    print(f"\t\tPrecision before smoothing: {precision}")
    print("############################################")
    print(f"\t\tRecall before smoothing: {recall}")
    print("############################################")
    print(f"\t\tF1 before smoothing: {fscore}")
    print("############################################")
    print()
    precision, recall, fscore, _ = precision_recall_fscore_support(final_predictions['Left_gt'],
                                                                   final_predictions['Left_pred_after'],
                                                                   labels=labels, average=None)
    precision = pd.DataFrame(precision.reshape(1, -1), columns=labels)
    precision.to_csv("precision_left_after.csv")
    recall = pd.DataFrame(recall.reshape(1, -1), columns=labels)
    recall.to_csv("recall_left_after.csv")
    fscore = pd.DataFrame(fscore.reshape(1, -1), columns=labels)
    fscore.to_csv("f1_left_after.csv")
    print(f"\t\tPrecision after smoothing: {precision}")
    print("############################################")
    print(f"\t\tRecall after smoothing: {recall}")
    print("############################################")
    print(f"\t\tF1 after smoothing: {fscore}")
    print("############################################")
