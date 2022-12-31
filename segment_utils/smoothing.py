import collections


def majority_voting_categorical_segmentation(original_segmentation, window_size):
    """Smooths categorical video segmentation over time using majority voting.

    Args:
        original_segmentation: A sequence of segmentations, where each segmentation is a label.
        window_size: The size of the temporal window to use for smoothing.

    Returns:
        A sequence of smoothed segmentation labels, with the same length as the input video.
    """

    # Initialize a queue to store the segmentation results for the last N frames
    segmentation_queue = []

    smoothed_segmentation = []

    # Loop over the frames in the video
    for segmentation in original_segmentation:

        if segmentation != 'None':

            # Add the segmentation result to the queue
            segmentation_queue.append(segmentation)

            # If the queue is larger than the window size, remove the oldest element
            if len(segmentation_queue) > window_size:
                segmentation_queue.pop(0)

            # Use majority voting to determine the smoothed segmentation
            vote = collections.Counter(segmentation_queue).most_common(1)[0][0]
            if vote == 'None':
                vote = segmentation
            smoothed_segmentation.append(vote)

        else:
            smoothed_segmentation.append('None')

    return smoothed_segmentation
