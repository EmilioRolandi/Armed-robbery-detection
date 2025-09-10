import cv2
import random
import os

def rectangles_overlap(r1, r2):
    x1, y1 = r1
    x2, y2 = r2
    return not (x1 + square_size <= x2 or x2 + square_size <= x1 or
                y1 + square_size <= y2 or y2 + square_size <= y1)

# Helper: get random non-overlapping squares
def get_random_squares(img_shape, tl, br, count):
    h, w, _ = img_shape
    squares = []
    tries = 0
    while len(squares) < count and tries < 1000:
        rand_x = random.randint(0, w - square_size)
        rand_y = random.randint(0, h - square_size)
        rand_pos = (rand_x, rand_y)
        if all(not rectangles_overlap(point, rand_pos)for point in tl):
                squares.append(rand_pos)
        tries += 1
    return squares


# Mouse callback
def mouse_callback(event, x, y, flags, param):
    global click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)

# Settings
for filename in os.listdir("youtube_photos/"):
    image_path = os.path.join("youtube_photos/", filename)
    if os.path.isfile(image_path):
        image_name, ext = os.path.splitext(filename)
        output_folder = 'output_folder'
        square_size = 100  # Size of the square to crop
        click_pos = None  # To store the position of the mouse click

        # Load the image
        image = cv2.imread(image_path)

        # Resize the image (Optional)
        frame_width = 640
        frame_height = 480
        image_resized = cv2.resize(image, (frame_width, frame_height))
        os.makedirs(output_folder, exist_ok=True)

        # Globals
        click_pos = None
        current_frame = None

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", mouse_callback) 

        # Helper: check overlap

        # Load video
        cap = cv2.VideoCapture(image_path)

        # Set the frame width and height
        frame_width = 640
        frame_height = 480
        top_left = []
        bottom_right = []


        display_image = image_resized.copy()
        while True:
            

            # Draw the square if clicked
            if click_pos:
                x, y = click_pos
                top_left.append((max(0, x-square_size//2), max(0, y-square_size//2)))
                bottom_right.append((top_left[-1][0] + square_size, top_left[-1][1] + square_size))
                cv2.rectangle(display_image, top_left[-1], bottom_right[-1], (0, 255, 0), 2)

            cv2.imshow("Image", display_image)

            key = cv2.waitKey(15) & 0xFF

            if key == ord(' '):  # Space to pause/resume (optional, you could ignore it for an image)
                break  # No need to pause in this case

            elif key == 13 and click_pos:  # Enter key (code 13)
                for top_l in top_left:
                    x1, y1 = top_l
                    crop_main = image_resized[y1:y1 + square_size, x1:x1 + square_size]

                # Save clicked square
                    cv2.imwrite(os.path.join(output_folder, f"y_image"+image_name+str(top_l)+".png"), crop_main)

                # Save three random squares (you can define get_random_squares or use any other logic)
                random_squares = get_random_squares((frame_height, frame_width, 3), top_left, bottom_right, 1)
                for idx, (rx, ry) in enumerate(random_squares):
                    crop_rand = image_resized[ry:ry + square_size, rx:rx + square_size]
                    cv2.imwrite(os.path.join(output_folder, f"n_image"+image_name+str(top_l)+".png"), crop_rand)

                print(f"Saved 1 'y' and 1 'n' images!")

                click_pos = None  # Reset click after saving

            elif key == ord('q'):  # 'q' to quit
                break

        cv2.destroyAllWindows()
