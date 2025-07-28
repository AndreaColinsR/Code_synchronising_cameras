# Descripción general
Este repositorio contiene el código preliminar para la reconstruir la marcha en 3D del Proyecto SUSESO

## Cómo usarlo
0. Imprima un a imagen de un tablero de ajedrez. Recomendamos imprimir [este ejemplo.](https://raw.githubusercontent.com/MarkHedleyJones/markhedleyjones.github.io/master/media/calibration-checkerboard-collection/Checkerboard-A3-35mm-8x6.pdf) Asegúrese de que al imprimir, la opción de "ajustar al tamaño de la página" está deseleccionada.
   
1. Descargue los archivos:
   1. Calibrate_both_cameras.ipynb
   2. Record_calibration_video.ipynb
   3. Record_video.ipynb
   4. bodypose3d.py
   5. show_3d_pose.py
   6. utils.py

2. En la carpeta donde está su código, cree las carpeta "Videos" y la carpeta "Calibration". En estas carpetas quedaran guardados los videos y la información de la calibración de las cámaras.

3. Calibración:
   - Para poder hacer la reconstrucción en 3D es crucial calibrar las dos cámaras que se utilizarán.
   - Para esto, use la cámara integrada del laptop y una camara USB.
   - Posicione las dos cámaras de manera tal que el sujeto siempre se vea desde las dos cámaras.
   - Tome su tablero de ajedrez y asegurese de que no se doble, por ejemplo, adhierala a un libro.
   - Abra el notebook `Record_calibration_video.ipynb`
   - Presione el botón para correr todo el código.
   - Esto automáticamente activará las cámaras y capturara dos videos (uno por cámara) sincronisados temporalmente. El video por defecto dura 7 segundos y se detiene.
   - Durante el video coloque el tablero de ajedrez en frente de las dos cámaras y trate de moverlo lo más posible para que cubra el área por donde caminara el sujeto.
   - Por favor cuide que el tablero de ajedrez se vea desde las dos cámaras.
    Puede ver un ejemplo de un video de calibración en la carpeta Calibration. El video Calib_video1contat.avi muestra los dos videos sincronizados.
   Si quiere grabar un nuevo video para reemplazar el anterior, sólo corra el programa de nuevo. Si quiere grabar otro video de calibración como backup, cambie la linea de codigo  `Nvideo = '1'` a `Nvideo = '2'` u otro número. 

**Por favor no mueva las cámaras una vez que grabó los videos de calibración. Si mueve las cámaras entonces tendrá que grabar un nuevo video de calibración.**


  4. Para grabar un video de un paciente use el notebook `Record_video.ipynb`. Abra el notebook presione Run y el codigo automaticamente registrara un video de XX segundos desde las dos camaras.
  5. Para grabar un nuevo video, cambie el numero del video en el código (Sorry! Cambiaré esto en versiones futuras). Segundo segmento cambiar `Nvideo= '6'` por otro número. Si no realiza este paso, se sobreescribirá el video anterior. 



# Reconstrucción en 3D y visualización de resultados 

Para hacer la reconstrucción en 3D de un video ya grabado se debe usar mediapipe. Por simplicidad, es mejor tener todo en un ambiente virtual. Una vez instalado el ambiente virtual y mediapipe, las siguientes instrucciones deberían producir una ventana con la visualización del video y la reconstrucción en 3D. Por defecto, el siguiente código hará la reconstrucción del video 4 en la carpeta `Videos/Usach_1/`

```python
# open command prompt and activate virtual environment
.\Documents\Virtual_Env_Py\Scripts\activate

# Python version Python 3.9.0
# Medida pipe Name: mediapipe
#Version: 0.10.8
#Summary: MediaPipe is the simplest way for researchers and developers to build world-class ML solutions and applications for mobile, edge, cloud and the web.
#Home-page: https://github.com/google/mediapipe
#Author: The MediaPipe Authors

# Go to the folder with all the code. For example:
cd .\SUSESO\Code_synchronising_cameras

# This function does the 3D reconstruction
python bodypose3d.py

# This function plots the results
python show_3d_pose.py

```
