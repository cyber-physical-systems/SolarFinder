import Augmentor
p = Augmentor.Pipeline("/aul/homes/final/split/location17/panel/")
# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.

# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.

# Add operations to the pipeline as normal:
p.rotate90(probability=1)
p.rotate270(probability=1)
p.flip_left_right(probability=1)
p.flip_top_bottom(probability=1)
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=1)
p.zoom_random(probability=1, percentage_area=0.8)
p.flip_top_bottom(probability=1)
p.sample(27420)
