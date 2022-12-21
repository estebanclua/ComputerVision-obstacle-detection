import pyzed.sl as sl
import cv2
import numpy as np

# Cria Objetos ZED
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.sdk_verbose = True
init_params.camera_fps = 30

# Abre a câmera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    # Fecha aplicação, caso ocorra erro
    exit()

# Define os parametros do módulo de Object Detection
detection_parameters = sl.ObjectDetectionParameters()
detection_parameters.image_sync = True
detection_parameters.enable_tracking = True
detection_parameters.enable_mask_output = True

if detection_parameters.enable_tracking:
    zed.enable_positional_tracking()

print("Object Detection: Loading Module...")
err = zed.enable_object_detection(detection_parameters)
if err != sl.ERROR_CODE.SUCCESS:
    print("Error {}, exit program".format(err))
    zed.close()
    exit()

# Coloca o limiar de confiança para objetos detectados em 40%
detection_parameters_runtime = sl.ObjectDetectionRuntimeParameters()
detection_parameters_runtime.detection_confidence_threshold = 40

# Inicializa variavel para ojetos e para imagens
objects = sl.Objects()
image = sl.Mat()

# Loop para realizar a detecção de objetos a cada frame
while zed.grab() == sl.ERROR_CODE.SUCCESS:
    err = zed.retrieve_objects(objects, detection_parameters_runtime)

    # Obtém a imagem do frame capturado da camera esquerda
    zed.retrieve_image(image, sl.VIEW.LEFT)
    imageOCV = image.get_data()

    if objects.is_new:

        # Conta o numero de objetos detectados
        print("{} Object(s) detected".format(len(objects.object_list)))

        if len(objects.object_list):

            # Mostra a posição dos objetos detectados para debug do código
            for i in range(len(objects.object_list)):
                first_object = objects.object_list[i]
                position = first_object.position
                print(" 3D position : [{0},{1},{2}]".format(position[0], position[1], position[2]))

                # Recebe as coordenadas para as caixas que envolvem os objetos
                bounding_box_2d = first_object.bounding_box_2d
                print(" Bounding box 2D :")

                # Realiza o cálculo de distância entre os objetos e a ZED, em mm
                distanciaX = position[0] ** 2
                distanciaY = position[1] ** 2
                distanciaZ = position[2] ** 2
                distancia = distanciaX + distanciaY + distanciaZ
                distancia = np.sqrt(distancia)

                # Inicializa a variável para guardar os pontos das caixas
                points = []
                contador = 0

                # Para cada caixa, serão mostrados seus pontos para debug
                for it in bounding_box_2d:
                    print(" " + str(it), end='')
                    if contador < 3:
                        points.append(it)
                        contador += 1
                    else:
                        contador = 0
                        x1 = int(points[0][0])
                        y1 = int(points[0][1])
                        x2 = int(points[2][0])
                        y2 = int(points[2][1])

                        xT = int(points[0][0])
                        yT = int(points[0][1])

                        # Calcula a distância em metros e inicializa a variável da cor
                        distancia = distancia / 1000
                        distancia = round(distancia, 2)
                        color = (0, 0, 0)

                        # Define a cor de acordo com a distância do objeto
                        if (distancia <= 0.8):
                            color = (0, 0, 255)
                        elif (distancia <= 1.2):
                            color = (0, 255, 255)
                        else:
                            color = (0, 255, 0)

                        # Desenha a caixa em torno do objeto e coloca um texto indicando a distância do mesmo
                        cv2.rectangle(imageOCV, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                        cv2.putText(imageOCV, "Distancia: " + str(distancia) + " m", (xT, yT), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, color, 2)
                        points.clear()

            # Mostra o frame capturado
            cv2.imshow("ZED", imageOCV)
            cv2.waitKey(10)

# Desabilita a deteção de objetos e fecha a câmera
zed.disable_object_detection()
zed.close()
