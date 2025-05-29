import cv2
import albumentations as A
import os

# Definir transformaciones avanzadas para iluminación, sombras y cámaras
transform = A.Compose([
    # Simulación de iluminación
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
    A.RandomGamma(gamma_limit=(70, 130), p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=0.5),

    # Simulación de sombras parciales
    #A.CoarseDropout(max_holes=4, max_height=5, max_width=5, fill_value=0, p=0.2),

    # Simulación de calidad de cámara
    A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
    A.Defocus(radius=(2, 5), p=0.3),  # Simula desenfoque de lentes
    A.MotionBlur(blur_limit=5, p=0.4),

    # Transformaciones geométricas y ruido
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), translate_percent=(0.1, 0.2), shear=(-5, 5), p=0.5),

    # Perspectiva ajustada
    A.Perspective(scale=(0.01, 0.03), keep_size=True, pad_mode=cv2.BORDER_CONSTANT, p=0.5),
])

# Directorio de entrada y salida
input_dir = "imagenes_originales"
output_dir = "imagenes_aumentadas"
os.makedirs(output_dir, exist_ok=True)

# Crear 30 variaciones por imagen
num_variations = 30

# Recorrer todas las imágenes del directorio de entrada
for file_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, file_name)
    image = cv2.imread(img_path)

    if image is not None:
        # Crear una carpeta para cada imagen
        img_name = os.path.splitext(file_name)[0]  # Obtener el nombre sin la extensión
        img_output_dir = os.path.join(output_dir, img_name)
        os.makedirs(img_output_dir, exist_ok=True)

        # Generar 30 variaciones
        for i in range(num_variations):
            augmented = transform(image=image)
            aug_image = augmented["image"]

            # Guardar cada variación en la carpeta correspondiente
            output_path = os.path.join(img_output_dir, f"{img_name}_aug_{i + 1}.jpg")
            cv2.imwrite(output_path, aug_image)

print("Se generaron las variaciones (30 por imagen) con iluminación, sombras y simulaciones de cámara.")
