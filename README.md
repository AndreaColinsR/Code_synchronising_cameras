# Descripción general
Este repositorio contiene el código preliminar para la reconstruir la marcha en 3D del Proyecto SUSESO

## Cómo usarlo
0. Imprima un a imagen de un tablero de ajedrez. Recomendamos imprimir este ejemplo. Asegúrese de que al imprimir, la opción de "ajustar al tamaño de la página" está deseleccionada.
   
1. Descargue los archivos:
   1. Calibrate_both_cameras.ipynb
   2. Record_calibration_video.ipynb
   3. Record_video.ipynb
   4. bodypose3d.py
   5. show_3d_pose.py
   6. utils.py

2. Calibración:
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

  3. Para grabar un video de un paciente use el notebook Record_video.ipynb. Abra el notebook presione Run y el codigo automaticamente registrara un video de XX segundos desde las dos camaras
  4. Para grabar un nuevo video, cambie el numero del video en el código. Sorry! 
