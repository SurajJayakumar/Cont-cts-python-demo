from deepface import DeepFace
import os
import shutil
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from rsa import verify
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace",
          "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
faces = RetinaFace.extract_faces(
    img_path=r"C:\Users\suraj\Saved Games\ML\news_preview_mob_image__preview_404.jpg", align=True)
print("%d faces Found!" % len(faces))
if len(faces) == 0:
    quit()
i = 0

for face in faces:
    i += 1
    plt.imsave(r"C:/Users/suraj/Saved Games/ML ps/images/face%d.jpg" % i, face)
    directory = "C:\\Users\\suraj\\Saved Games\\ML ps\\contacts"
    flag = 0
    result= DeepFace.verify(img1_path=r"C:/Users/suraj/Saved Games/ML ps/images/face{}.jpg".format(i), img2_path = r'C:\Users\suraj\Saved Games\ML ps\contacts\zayn_malik\zayn_malik.jpg', model_name = models[1], distance_metric = 'cosine', model = None, enforce_detection = False, detector_backend = 'opencv', align = True, prog_bar = True, normalization = 'base')
    print(result)
    plt.imshow(face)
    plt.show()
