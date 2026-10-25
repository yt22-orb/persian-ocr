# Deferred mirror image checks

Consider eight rotation checks of horizontally flipped images for the rotation and contrast-plus-rotation paths. This feature is deferred: it has not been implemented or benchmarked, and its latency and accuracy effects have not been measured. Evaluate added correct answers, wrong accepted answers, negative rejection, and runtime before changing the defaults. If preserving all current predictions is the goal, try mirrored views only after the existing cascade fails, while retaining accepted BASE results and BASE multiple-letter rejections.
