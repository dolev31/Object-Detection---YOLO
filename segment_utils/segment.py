import cv2
import bbox_visualizer as bbv


def single_segmentation(frame_res):
    # sorting the predictions by horizontal - position of bounding boxes
    pred = sorted(frame_res.xyxy[0].int().tolist(), key=lambda x: x[0])

    right_label = 'None'
    left_label = 'None'

    if len(pred) == 2:
        right_pred = pred[0][5]
        left_pred = pred[1][5]

        if left_pred % 2 == right_pred % 2:
            # both predictions refer to the same hand
            if right_pred % 2 == 0:
                # both predictions refer to the Right hand
                pred[1][5] = pred[1][5] + 1
            else:
                # both predictions refer to the Left hand
                pred[0][5] = pred[0][5] - 1

        numeric_classes = [p[5] for p in pred]
        str_classes = [frame_res.names[p] for p in numeric_classes]

        right_label = [p for p in str_classes if "Right" in p][0]
        left_label = [p for p in str_classes if "Left" in p][0]

    elif len(pred) == 1:
        numeric_class = pred[0][5]
        str_class = frame_res.names[numeric_class]

        if numeric_class % 2 == 0:
            right_label = str_class
        else:
            right_label = str_class

    elif len(pred) == 0:
        pass

    else:
        print("Error! not eligible amount of predictions")

    return pred, right_label, left_label


def single_img_annotation(frame, bbox, right_pred, left_pred):
    curr_pred = []
    if right_pred != 'None':
        curr_pred.append(right_pred)

    if left_pred != 'None':
        curr_pred.append(left_pred)

    assert len(curr_pred) == len(bbox), \
        f"num hands != num boxes!! len(bboxes) = {len(bbox)} and len(hands) = {len(curr_pred)}"

    if len(bbox):

        new_frame = bbv.draw_multiple_rectangles(
            img=frame,
            bboxes=bbox,
            bbox_color=(255, 102, 102)
        )

        new_frame = bbv.bbox_visualizer.add_multiple_labels(
            img=new_frame,
            labels=curr_pred,
            bboxes=bbox,
            draw_bg=True,
            text_bg_color=(255, 102, 102),
            text_color=(0, 0, 0)
        )

    else:
        new_frame = frame

    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    return new_frame
