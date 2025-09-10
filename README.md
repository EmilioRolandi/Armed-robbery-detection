# Armed-robbery-detection
The following project fine tunes a pretrained model to detect guns in pictures of security footage. 

In order use it, I searched "armed robbery footage" in YouTube. This is important to get actual security 
footage instead of documentaries about certain robberies. From there, I downloaded multiple frames (pictures)
where a gun was clearly visible.

This was downloaded into a folder called "youtube_photos". Then, the script marker.py was run. 
It loops through all the photos in "youtube_photos" and displays them. You can click on a given coordinate and a 100x100
square will be drawn around that point. Also, another other 100x100 square will be drawn in a random position of the picture,
ensuring both squares do not overlap.
If you press Enter, the 100x100 square
that you selected will be saved in a folder called "output_folder" and its name will start with y_ (denoting that the picture 
contains a gun). The other square will be saved in the same folder but its name will start with n_ (denoting that the picture 
does not contain a gun). 

Finally, you can run pretrained.py. It will use an already trained model (REsNET18) and train it on the data from "output_folder". 
It will start by "freezing" (not updating) every layer but the last one. Then it will unfreeze the previous layer after 5 epochs. 
After another 5 epochs it will release the previous layer. In total, the model is trained for 20 epochs and typically reaches 
a test accuracy of 88%. 

Possible improvements include focusing only on images of high quality, usig different models, a bigger train size, or only focusing on 
regions that are near people. 
