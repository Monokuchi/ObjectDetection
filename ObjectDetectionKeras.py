# Creates a YOLOv3 Keras model and saves it to a H5 file
import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import yolo3_one_file_to_detect_them_all as yolov3
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Defining the model
model = yolov3.make_yolov3_model()

# Load model weights
weight_reader = yolov3.WeightReader('yolov3.weights')

# Set the model weights into the model
weight_reader.load_weights(model)

# At this stage the model is created and loaded with the weighs and now ready to be used

# Saves the model to a H5 file
model.save('model.h5')

# Loading a yolov3 model
model = load_model('model.h5')


# Makes a method called load_image_pixels that takes in the picture filename and the desired target size and returns
# the scaled pixel data, ready to be inputted into the Keras model and also returns the original width and height of the image#
def load_image_pixels(filename, shape):
    # Loading the image to get it's shape
    image = load_img(filename)
    width, height = image.size

    # Now we need to load in our image with appropriate dimensions (416 pixels, 416 pixels)
    image = load_img(filename, target_size=shape)

    # Converting the image to a numpy array
    image = img_to_array(image)

    # Scaling the pixel values from [0,255] to [0,1]
    image = image.astype('float32')
    image /= 255.0

    # Adds a dimension to the image array in the 0'th position so that we can have one sample
    image = np.expand_dims(image, 0)

    # Returns the image as well as the original picture's width and height
    return image, width, height


# Defines the expected input dimensions for our model (These dimensions are fixed for the Keras model)
input_w = 416
input_h = 416

# Defines our photo
photo_filename = 'people.jpg'

# Loads and prepares the image
image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))


# AT THIS POINT THE IMAGE IS NOW READY TO BE FED INTO THE KERAS MODEL IN ORDER TO MAKE PREDICTIONS


# Make predictions
predictions = model.predict(image)

# Summarize the shape of the list of arrays
print([a.shape for a in predictions])

# AT THIS STAGE WE CAN TAKE THE RESULTS AND INTERPRET THEM


# Defining the anchors
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

# Defining the probability threshold for the detections of objects
probability_threshold = .8
boxes = list()
for i in range(len(predictions)):
    # Decoding the output of the network
    boxes += yolov3.decode_netout(predictions[i][0], anchors[i], probability_threshold, input_h, input_w)

# Now we need to stretch the bounding boxes back to the original picture size
yolov3.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

# Suppressing overlapping boxes (We will still have same amount of boxes but the amount of boxes that we actually
# care about is lowered)
overlap_threshold = .5
yolov3.do_nms(boxes, overlap_threshold)


# Now we want to create a function that checks and returns a list of boxes that passes the probability threshold that we set
def get_boxes(boxes, labels, threshold):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # Count up all the boxes
    for box in boxes:
        # Count up all the possible labels
        for i in range(len(labels)):
            # Checks to see if the threshold for each individual label is high enough
            if box.classes[i] > threshold:
                # If the threshold is high enough we go ahead and append the accepted values into the three lists
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
    return v_boxes, v_labels, v_scores


# Defining the label names that will appear on our boxes as identifiers
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# We can now call our get_boxes function to get back our final product boxes, labels, and scores
v_boxes, v_labels, v_scores = get_boxes(boxes, labels, probability_threshold)

# Summarizing the end result of everything
for i in range(len(v_boxes)):
    print(v_labels[i], v_scores[i])


# WE HAVE THE FINAL RESULTS IN NUMBER FORM, NOW WE WANT TO DISPLAY THEM ON OUR ORIGINAL IMAGE WITH THE BOXES


# Function to draw all the final results onto the original picture
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # Load in the original picture
    data = pyplot.imread(filename)
    # Plot the picture
    pyplot.imshow(data)
    # Get the data for drawing in the boxes
    ax = pyplot.gca()
    # Plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # Get the coordinates associated with each box
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # Calculate the width and height of each of the boxes
        width, height = x2-x1, y2-y1
        # Make the shape
        rect = Rectangle((x1, y1), width, height, fill = False, color = 'white')
        # Finally, draw in the box and the associated label and scores
        ax.add_patch(rect)
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color = 'white')
    # Show the finished plot
    pyplot.show()


# Call the method draw_boxes to draw what we found
draw_boxes('people.jpg', v_boxes, v_labels, v_scores)



