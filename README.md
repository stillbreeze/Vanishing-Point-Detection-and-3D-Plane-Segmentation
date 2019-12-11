# Vanishing Point Detection and 3D Plane Segmentation

-----------------------
Two different problems are solved here. The first is automatically finding 3 orthogonal vanishing points in Manhattan scenes. Second is detecting and segmenting planar regions of images in 3D.

-----------------------

Automatic vanishing point detection

- Edge Detection using Canny
- Line segmenta formation using Probabilistic Hough Transforms
- Identification of 3 dominant directions of line segments using iterative RANSAC 
- Finding best fitting vanishing points for each direction

<img src="pdf_images/q1/q1_img4/P1030001_inliers_iter3000_thresh2_sigma5_hlen11_hgap7.png" width="400">

<img src="pdf_images/q1/q1_img4/P1030001_vanishing_point_iter3000_thresh2_sigma5_hlen11_hgap71.png" width="400">

<img src="pdf_images/q1/q1_img4/P1030001_vanishing_point_iter3000_thresh2_sigma5_hlen11_hgap72.png" width="400">

<img src="pdf_images/q1/q1_img4/P1030001_vanishing_point_iter3000_thresh2_sigma5_hlen11_hgap70.png" width="400">

<img src="pdf_images/q1/q1_img4/P1030001_vanishing_point_iter3000_thresh2_sigma5_hlen11_hgap7.png" width="400">

-----------------------
