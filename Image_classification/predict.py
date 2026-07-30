import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
model = tf.keras.models.load_model("image_classifier_model.keras")
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
image_path = input("Enter the path to the image: ")
img=image.load_img(image_path, target_size=(32,32))

image_array=image.img_to_array(img)
image_array=image_array/255.0
image_array=np.expand_dims(image_array, axis=0)
predictions=model.predict(image_array)
predicted_class=np.argmax(predictions[0])
plt.imshow(img)
plt.title("Predicted: " + class_names[predicted_class])
plt.axis("off")
plt.show()
confidence = np.max(predictions[0])*100
print("Predicted Class:", class_names[predicted_class])
print("Confidence:", confidence)
