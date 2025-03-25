import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from skimage.transform import resize
from joblib import dump, load
from PIL import Image


# Функция для загрузки изображений и извлечения признаков
def load_data(data_dir):
    images = []
    labels = []
    classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            try:
                # Открытие и предобработка изображения
                img = Image.open(img_path).convert('RGB')
                img = resize(np.array(img), (64, 64), anti_aliasing=True)

                # Извлечение цветовых гистограмм
                hist_r = np.histogram(img[:, :, 0], bins=32, range=(0, 1))[0]
                hist_g = np.histogram(img[:, :, 1], bins=32, range=(0, 1))[0]
                hist_b = np.histogram(img[:, :, 2], bins=32, range=(0, 1))[0]

                # Объединение признаков
                feature_vector = np.concatenate([hist_r, hist_g, hist_b])
                images.append(feature_vector)
                labels.append(class_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return np.array(images), np.array(labels)


def process_image(image_path):
    """
    Предобработка изображения для модели классификации цветов

    Параметры:
    image_path (str): Путь к файлу изображения

    Возвращает:
    numpy.array: Вектор признаков (цветовые гистограммы)
    """
    try:
        # 1. Загрузка изображения
        img = Image.open(image_path).convert('RGB')

        # 2. Конвертация в numpy array и нормализация
        img_array = np.array(img) / 255.0

        # 3. Изменение размера с антиалиасингом
        resized_img = resize(img_array,
                             (64, 64),
                             anti_aliasing=True,
                             preserve_range=False)

        # 4. Извлечение цветовых гистограмм
        # Настройки должны точно соответствовать обучению!
        bins = 32
        range_values = (0, 1)

        hist_r = np.histogram(resized_img[:, :, 0], bins=bins, range=range_values)[0]
        hist_g = np.histogram(resized_img[:, :, 1], bins=bins, range=range_values)[0]
        hist_b = np.histogram(resized_img[:, :, 2], bins=bins, range=range_values)[0]

        # 5. Объединение и возврат признаков
        return np.concatenate([hist_r, hist_g, hist_b])

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# Загрузка данных
data_dir = 'flowers'
X, y = load_data(data_dir)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Создание и обучение модели
pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
)

pipeline.fit(X_train, y_train)

# Оценка модели
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 1. Сохраняем модель
dump(pipeline, 'flower_classifier.joblib')

model = load('flower_classifier.joblib')


def predict_flower(image_path):
    # Загрузка модели

    features = process_image(image_path)

    if features is not None:
        # Преобразование в формат для модели (2D array)
        features = features.reshape(1, -1)

        # Предсказание
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)

        print(f"Predicted class: {prediction[0]}")
        print("Class probabilities:")
        for class_name, prob in zip(model.classes_, probabilities[0]):
            print(f"{class_name}: {prob:.2f}")
    else:
        print("Prediction failed due to processing error")


# Рандомная картинка из датасета
predict_flower('img.jpg')

features = process_image("img.jpg")
print(f"Feature vector shape: {features.shape}")  # Должно быть (96,)
