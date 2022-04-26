# Graphic-seam-carving

Our implemention of seam carving algorithm.

Content-aware image resizing changes the image resolution while maintaining the aspect ratio
of important regions.

The program is a command-line application with the following options:
o --image_path (str)– An absolute/relative path to the image you want to process
o --output_dir (str)– The output directory where you will save your outputs.
o --height (int) – the output image height
o --width (int) – the output image width
o --resize_method (str) – a string representing the resize method. Could be one of
the following: [‘nearest_neighbor’,
‘seam_carving’]
o --use_forward_implementation – a boolean flag indicates if forward looking
energy function is used or not.
o --output_prefix (str) – an optional string which will be used as a prefix to the
output files. If set, the output files names will start with the given prefix. For
seam carving, we will output two images, the resized image, and visualization of
the chosen seems. So if --output_prefix is set to “my_prefix” then the output will
be my_prefix_resized.png and my_prefix_horizontal _seams.png,
my_prefix_vertical_seams.png. If the prefix is not set, then we will chose “img”
as a default prefix.

Exmaple 

--image_path <your path> --output_dir output --height 1100 --width 1500 --resize_method seam_carving --out_prefix our_forward_camels_w1500_h1100 --use_forward_implementation
  
  
Returns in addition to the resized image, you will need to return two additional images which visualize
the chosen seams in each direction (vertical and horizontal).

