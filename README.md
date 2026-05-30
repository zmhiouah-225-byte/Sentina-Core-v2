Si vous n'avez pas de caméra ou de piscine sous la main, vous pouvez tester le code directement sur le fichier vidéo mp4 inclus dans ce dépôt :

1. Ouvrez le fichier `main.py`.
2. Modifiez la ligne de la source vidéo (cv2.VideoCapture) :
   **Remplacez :** `cap = cv2.VideoCapture(0)`
   **Par :** `cap = cv2.VideoCapture('nom_de_votre_video.mp4')`
3. Lancez le script. L'IA va analyser la vidéo automatiquement !
