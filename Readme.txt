Full project report:https://inst.eecs.berkeley.edu/~cs194-26/sp20/upload/files/proj2/cs194-26-act/

Results from this assignments can be reproduced by running the main.py

Part1.1:
Run main.py with path of the source image as the first argument, followed by 'x', 'y' or 'e' as the second, indicating that the resulting image extract gradient on x axis, y axis or gradient magnitude. The resulting image will be saved into the current directory as "D_x.jpg", "D_y.jpg" or "Edge_mag.jpg."

Part1.2:
Run main.py with path of the source image as the first argument, followed by 'gaux', 'gauy' or 'gaue' as the second, indicating that the resulting image extract gradient on x axis, y axis or gradient magnitude. The resulting image will be saved into the current directory as "D_x Gau.jpg", "D_y Gau.jpg" or "Edge_mag Gau.jpg."

Part1.3:
Run main.py with path of the source image as the first argument, followed by 'straight' as the second. The computed
angle will be printed in the terminal and the resulting image will be saved into the current directory, with the same name with 'straight_' added at the beginning.

Part2.1:
Run main.py with path of the source image as the first argument, followed by 'sharp' as the second. The resulting 
image will be saved into the current directory, with the same name with 'sharp_' added at the beginning.

Part2.2:
Run main.py with path of the high frequency image as the first argument, path of the low frequency image as the second argument and "hybrid" as the third argument. Click on the align points in the pop up windows. The resulting image will be saved in the currect directory with the first three letters from both images combined and followed by "_hybrid.jpg"

Part2.3:
Run main.py with path of the source image as the first argument, followed by "gaupyramid" or "lappyramid". The program
will then create and save 5 images into the current directory, each image corresponds to a layer in the pyramid. The 
names of the saved images are the layer number followed by "gau" or "lap" and then the original image name.

Part2.4:
Run main.py with path of the two images as the first and second arguments, "blend" as the third argument and either "h" or "v" as the final argument, indicating whether the two source images are blended vertically or horizontally. The program will then create and save the blended image. The name of the image file created will start with the first three letters from both images combined followed by "_blend.jpg". The two input images must have the same shape.

