import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # Indeces for readability 
    xmin,ymin,xmax,ymax = 0,1,2,3
    # Compute intersection
    # This is not super intuitive, but rest assured, it works
    overlapping_x = max(0,min(prediction_box[xmax],gt_box[xmax])-max(prediction_box[xmin],gt_box[xmin]))
    overlapping_y = max(0,min(prediction_box[ymax],gt_box[ymax])-max(prediction_box[ymin],gt_box[ymin]))
    intersection = overlapping_x*overlapping_y
    # Compute union
    union = (gt_box[xmax]-gt_box[xmin])*(gt_box[ymax]-gt_box[ymin])
    union += (prediction_box[xmax]-prediction_box[xmin])*(prediction_box[ymax]-prediction_box[ymin])

    union -= intersection
    try:
        iou = intersection/union
        
    except ZeroDivisionError:
        print("Encountered zero division in Calculate IoU function...")

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if(num_tp+num_fp == 0):
        return 1.0
    return ((num_tp)/(num_tp+num_fp))


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if(num_tp+num_fn==0):
        return 0.0
    return ((num_tp)/(num_tp+num_fn))


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold

    matches_pred = np.empty((0,4),float)
    matches_gt =  np.empty((0,4),float)
    indexes_gt = []
    indexes_pred = []
    ious = np.array([])


    for gt in range(gt_boxes.shape[0]):
        for pred in range(prediction_boxes.shape[0]):
            iou = calculate_iou(prediction_boxes[pred],gt_boxes[gt])
            if iou>=iou_threshold:
                indexes_gt.append(gt)
                indexes_pred.append(pred)
                ious = np.append(ious,iou)

    # Sort all matches on IoU in descending order
    
    increasing = np.argsort(ious)
    decreasing = increasing[::-1]
    
    taken_gt = []
    taken_pred = []

    for index in decreasing:
        
        pred_index = indexes_pred[index]
        gt_index = indexes_gt[index]
        if ((pred_index not in taken_pred) and (gt_index not in taken_gt)):
            taken_gt.append(gt_index)
            taken_pred.append(pred_index)
            matches_pred = np.append(matches_pred,[prediction_boxes[pred_index]],axis = 0)
            matches_gt = np.append(matches_gt,[gt_boxes[gt_index]],axis = 0)
    # Sort all matches on IoU in descending order
    

    # Find all matches with the highest IoU threshold
    
    return matches_pred, matches_gt


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    CM = {"true_pos": 0,
            "false_pos": 0,
            "false_neg": 0}

    match_pred, match_gt = get_all_box_matches(prediction_boxes,gt_boxes,iou_threshold)
    CM["false_pos"] = prediction_boxes.shape[0]-match_pred.shape[0]
    CM["true_pos"] = match_pred.shape[0]
    CM["false_neg"] = gt_boxes.shape[0]-match_gt.shape[0]
    return CM

def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for pred_boxes, gt_boxes in zip(all_prediction_boxes,all_gt_boxes):
  
        CM = calculate_individual_image_result(pred_boxes, gt_boxes, iou_threshold)
        true_pos += CM["true_pos"]
        false_pos += CM["false_pos"]
        false_neg += CM["false_neg"]
    
    recall = calculate_recall(true_pos,false_pos,false_neg)
    precision = calculate_precision(true_pos,false_pos,false_neg)
    
    return precision,recall


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)


    precisions = []
    recalls = []
    for threshold in confidence_thresholds:
        all_preds = []
        for conf, preds in zip(confidence_scores, all_prediction_boxes):
            img= []
            for i, bound in enumerate(conf):
                if(bound >= threshold):
                    img.append(preds[i])

            all_preds.append(np.array(img))
        precision, recall = calculate_precision_recall_all_images(all_preds, all_gt_boxes, iou_threshold)

        recalls.append(recall)
        precisions.append(precision)

    return np.array(precisions), np.array(recalls)

def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE

    # Calculate the mean average precision given these recall levels.
    AP = 0
    for rec in recall_levels:
        myPre = 0
        for i in range(len(precisions)):
            if precisions[i] > myPre and recalls[i] >= rec:
                myPre = precisions[i]
        AP += myPre
    AP = AP/float(len(recall_levels))
    return AP


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
