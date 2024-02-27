// Importações das bibliotecas necessárias
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StyleSheet, Text, View, Dimensions, Platform } from 'react-native';
import {BarCodeScanningResult, Camera, CameraType, Point} from 'expo-camera';
import { StatusBar } from 'expo-status-bar';
import { bundleResourceIO, cameraWithTensors } from "@tensorflow/tfjs-react-native";
import * as tf from "@tensorflow/tfjs";
import { ExpoWebGLRenderingContext } from "expo-gl";
import {BarCodeScanner} from "expo-barcode-scanner";

type BoundingBox = [number, number, number, number]; // [x1, y1, x2, y2]

type ProcessedOutput = Array<[number, number, number, number, string, number]>; // [x1, y1, x2, y2, label, probability]

// Carrega o modelo de machine learning e seu binário correspondente
const modelJson = require("./assets/model/test_yolo/web_model_v2/model.json");
const modelBin1 = require("./assets/model/test_yolo/web_model_v2/group1-shard1of3.bin");
const modelBin2 = require("./assets/model/test_yolo/web_model_v2/group1-shard2of3.bin");
const modelBin3 = require("./assets/model/test_yolo/web_model_v2/group1-shard3of3.bin");

// Cria um componente de câmera que pode interagir com tensores do TensorFlow
const TensorCamera = cameraWithTensors(Camera);

// Define constantes baseadas na plataforma (iOS ou Android)
const IS_IOS = Platform.OS === 'ios';
const CAM_PREVIEW_WIDTH = Dimensions.get('window').width;
const CAM_PREVIEW_HEIGHT = CAM_PREVIEW_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

export default function App() {
  // Estados para gerenciar permissões da câmera, prontidão do modelo e saída do modelo
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [modelReady, setModelReady] = useState<boolean>(false);
  const [modelOutput, setModelOutput] = useState<string>('');
  const [qrCodeData, setQrCodeData] = useState<string>('Dados do QrCode: não identificado.');

  // Solicita permissão para a câmera e prepara o modelo na montagem do componente
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
    prepareModel();
  }, []);

  // Carrega o modelo TensorFlow.js
  const prepareModel = async () => {
    try {
      await tf.ready();
      model.current = await tf.loadGraphModel(bundleResourceIO(modelJson, [modelBin1,modelBin2,modelBin3]));
      console.log("Carregou o modelo")
      setModelReady(true);
    } catch (error) {
      console.error("Error loading model: ", error);
    }
  };

  // Ref para manter a referência ao modelo
  const model = useRef<tf.GraphModel | null>(null);

  const yolo_classes = ['circle'] 

  /**
   * Calback é executado toda vez que a leitura do QRCode é bem sucedida
   **/
  const handleBarCodeScanned = useCallback(async (
    scanningResult: BarCodeScanningResult,
  ) => {

    const cornerPoints: Point[] = scanningResult.cornerPoints
    setQrCodeData(`Dados do QrCode: ${scanningResult.data}`)
  }, [modelReady])

  function iou(box1: BoundingBox, box2: BoundingBox): number {
    return intersection(box1, box2) / union(box1, box2);
  }

  function union(box1: BoundingBox, box2: BoundingBox): number {
      const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
      const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
      const box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
      const box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);
      return box1_area + box2_area - intersection(box1, box2);
  }

  function intersection(box1: BoundingBox, box2: BoundingBox): number {
      const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
      const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
      const x1 = Math.max(box1_x1, box2_x1);
      const y1 = Math.max(box1_y1, box2_y1);
      const x2 = Math.min(box1_x2, box2_x2);
      const y2 = Math.min(box1_y2, box2_y2);
      return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  }

  function process_output(output: tf.Tensor, img_width: number, img_height: number): ProcessedOutput {
    const outputData = output.dataSync(); // Convert TensorFlow tensor to array synchronously

    let boxes: ProcessedOutput = [];
    for (let index = 0; index < 8400; index++) {
        const [class_id, prob] = [...Array(80).keys()]
            .map(col => [col, outputData[8400 * (col + 4) + index]])
            .reduce((accum, item) => item[1] > accum[1] ? item : accum, [0, 0]);
        if (prob < 0.5) {
            continue;
        }
        const label = yolo_classes[class_id];
        const xc = outputData[index];
        const yc = outputData[8400 + index];
        const w = outputData[2 * 8400 + index];
        const h = outputData[3 * 8400 + index];
        const x1 = (xc - w / 2) / 640 * img_width;
        const y1 = (yc - h / 2) / 640 * img_height;
        const x2 = (xc + w / 2) / 640 * img_width;
        const y2 = (yc + h / 2) / 640 * img_height;
        boxes.push([x1, y1, x2, y2, label, prob]);
    }

    boxes = boxes.sort((box1, box2) => box2[5] - box1[5]);
    const result: ProcessedOutput = [];
    while (boxes.length > 0) {
        const currentBox = boxes[0];
        const [x1, y1, x2, y2] = currentBox;
        result.push(currentBox);
        const slicedBox: BoundingBox = [x1, y1, x2, y2];
        boxes = boxes.filter(box => iou(slicedBox, box.slice(0, 4) as BoundingBox) < 0.7);
    }
    return result;
}


  // Callback que lida com o stream da câmera e faz previsões
  const handleCameraStream = useCallback(async (
    images: IterableIterator<tf.Tensor3D>,
    updatePreview: () => void,
    gl: ExpoWebGLRenderingContext
  ) => {
    const loop = async () => {
      // Executa o loop somente se o modelo estiver pronto
      if (modelReady) {
        const imageTensor = images.next().value;
        if (imageTensor) {
          const imageTensorPrep = tf.tidy(() => {
            // Processamento do tensor da imagem
            const imageTensor = images.next().value as tf.Tensor3D;
            const imageFloat = tf.cast(imageTensor, "float32");
            const normalizedImage = imageFloat.div(tf.scalar(255));
            const inputImage = tf.expandDims(normalizedImage, 0);
            imageTensor.dispose();
            return inputImage;
          });
          // Faz a previsão usando o modelo
          const prediction = model.current?.predict(imageTensorPrep) as tf.Tensor
          const processedOutput = process_output(prediction, 640, 640);
          console.log(processedOutput);
          setModelOutput(`Processed Output: ${JSON.stringify(processedOutput)}`);
          model.current?.disposeIntermediateTensors();
          tf.dispose([imageTensorPrep, prediction]);
        }
      }
      // Solicita o próximo frame para processamento
      requestAnimationFrame(loop);
    };

    loop();
  }, [modelReady]);

  // Verifica o estado da permissão da câmera antes de renderizar
  if (hasPermission === null) {
    return <View />;
  }
  if (!hasPermission) {
    return <Text>No access to camera</Text>;
  }

  // Renderiza a câmera e a saída do modelo
  return (
    <View style={styles.container}>
      {modelReady && (
        <TensorCamera
          // Configurações do componente da câmera
          barCodeScannerSettings={{
            barCodeTypes: [BarCodeScanner.Constants.BarCodeType.qr],
          }}
          onBarCodeScanned={handleBarCodeScanned}
          ratio={'16:9'}
          pictureSize={"1280x720"}
          style={styles.camera}
          autorender={false}
          type={CameraType.back}
          // Propriedades relacionadas ao tensor
          resizeWidth={640}
          resizeHeight={640}
          useCustomShadersToResize={false}
          cameraTextureHeight={640}
          cameraTextureWidth={640}
          resizeDepth={3}
          rotation={0}
          onReady={handleCameraStream}
          // Outras propriedades
        />)}
      <Text style={styles.qrCodeText}>{qrCodeData}</Text>
      <Text style={styles.predictionText}>{modelOutput}</Text>
      <StatusBar style="auto" />
    </View>
  );
}

// Estilos do componente
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  camera: {
    flex: 1,
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
  },
  predictionText: {
    color: 'white',
    fontSize: 18,
    position: 'absolute',
    bottom: 10,
    left: 10,
  },
  qrCodeText: {
    color: 'white',
    fontSize: 18,
    position: 'absolute',
    top: "60%", // Move o texto do QR Code para cima
    left: 10,
  },
});
