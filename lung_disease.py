import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image

# # Load the trained model
# model = load_model("model.h5")

# Function to preprocess input image
def preprocess_image(image_path, img_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = image.astype(np.float32) / 255.0  # Normalize
    return np.expand_dims(image, axis=0), image  # Return batch and original image

# Function to generate Grad-CAM heatmap
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to overlay Grad-CAM heatmap on original image
def overlay_gradcam(image, heatmap, alpha=0.4, colormap=cm.jet):
    # Convert image to uint8 (if it's not already)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # Remove alpha channel
    heatmap_colored = np.uint8(heatmap_colored * 255)

    # Ensure both are uint8
    superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return superimposed_img

# Function to make a prediction and generate Grad-CAM output
def predict_and_visualize(image_path, model, last_conv_layer_name="mixed10"):
    img_array, original_image = preprocess_image(image_path)
    prediction = model.predict(img_array)
    
    predicted_class = np.argmax(prediction[0])
    predicted_probability = prediction[0][predicted_class]  # Extract probability
    
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name, predicted_class)
    gradcam_image = overlay_gradcam(original_image, heatmap)
    
    return predicted_class, predicted_probability, gradcam_image

# Function to get Grad-CAM heatmap without overlaying on original image
def generate_gradcam_only(image_path, model, last_conv_layer_name="mixed10"):
    img_array, _ = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name, predicted_class)
    return predicted_class, heatmap

# Function to check if the given image is a lung X-ray
def is_lung_xray(image_path, model):
    img = image.load_img(image_path, target_size=(256, 256))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize input

    prediction = model.predict(img_array)  # Get probability
    lung_prob = prediction[0][0]  # Assuming output is [lung_prob]

    return lung_prob > 0.5  # Return True if it's a lung X-ray, False otherwise

