Nighttime Vehicle Detection Dataset
School of Mobile Information Engineering, Sun Yat-sen University

1. Detection Images
This dataset have 5776 nighttime vehicle images named from 000001.jpg to 00005576.jpg.
All images have at least one vehicle.
Attention: 000001.jpg to 002663.jpg images have 640 height x 360 width, and 002664.jpg to 005576.jpg images have  720 height x 405 width.

2. Ground Truth of Detection Images
The Ground Truth file is named as GT5576.txt. The nth row in the file corresponds to the nth detection image; for example, if the nth detection image contains m vehicles with location pairs (x y width height), where x and y present X-coordinate and Y-coordinate of the top-left point. Width and height present the width and height of the vehicle in the detection image. Then the nth row looks like:
test<n> m x1 y1 width1 height1 x2 y2 width2 height2 ... xm ym widthm heightm.

The GT5576.txt is organized as follows:
--------------------------------------------------------------------
ImageName  |  VehicleNumbers  |  [x1   y1  width  height] for each vehicle
 000001             6            347   142  108     64   252 145 56 30 252 145 56 30 3 132 95 54 140 142 35 25 185 142 49 25
