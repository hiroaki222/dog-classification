import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import xml.etree.ElementTree as ET

# アノテーションを処理
def parse(file):
    tree = ET.parse(file)
    root = tree.getroot()

    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    obj = root.find('object')
    name = obj.find('name').text
    xmin = int(obj.find('bndbox/xmin').text)
    ymin = int(obj.find('bndbox/ymin').text)
    xmax = int(obj.find('bndbox/xmax').text)
    ymax = int(obj.find('bndbox/ymax').text)
    objects = {
        'name': name,
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax
    }

    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }

# 前処理*
def preprocess(image, target):
    image = image.resize(target)
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    return image

# モデル生成
def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


anno_dir = 'data/annotations/Annotation/n02113624-toy_poodle'
img_dir = 'data/images/Images/n02113624-toy_poodle'

imgs = []
labels = []
image_files = os.listdir(img_dir)
anno_files = os.listdir(anno_dir)
classes = 10
target = (224, 224)

# データセット生成
for i in anno_files:
    anno_path = anno_dir + '/' + i

    anno_data = parse(anno_path)
    filename = anno_data['filename']

    img_path = img_dir + '/' + filename + '.jpg'
    img = PIL.Image.open(img_path)

    name = anno_data['objects']['name']
    xmin = anno_data['objects']['xmin']
    ymin = anno_data['objects']['ymin']
    xmax = anno_data['objects']['xmax']
    ymax = anno_data['objects']['ymax']

    img_cropped = img.crop((xmin, ymin, xmax, ymax))
    image_preprocessed = preprocess(img_cropped, target)

    imgs.append(image_preprocessed)
    labels.append(name)
dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
# 学習とテストに分ける
num = int(len(dataset) * 0.8)
train_dataset = dataset.take(num)
test_dataset = dataset.skip(num)

input_shape = target + (3,)

model = create_model(input_shape, classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_loss_history = []
train_accuracy_history = []
class TrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs['loss']
        train_accuracy = logs['accuracy']
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
callbacks = TrainingCallback()

model.fit(train_dataset, epochs=10, callbacks=[callbacks])
test_loss, test_accuracy = model.evaluate(test_dataset)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_history)
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig("lossaccuracy.png")