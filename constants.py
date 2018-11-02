n = 3
image_size = (225, 225)
piece_crop_percentage = 0.85
tile_size = tuple(round(piece_crop_percentage * int(x / n)) for x in image_size)
mean = [0.485, 0.456, 0.406]
stdDev = [0.229, 0.224, 0.225]
stl10_number_of_classes = 10
