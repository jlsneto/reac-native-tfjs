// Importações das bibliotecas necessárias
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StyleSheet, Text, View, Dimensions, Platform } from 'react-native';
import { Camera, CameraType } from 'expo-camera';
import { StatusBar } from 'expo-status-bar';
import { bundleResourceIO, cameraWithTensors } from "@tensorflow/tfjs-react-native";
import * as tf from "@tensorflow/tfjs";
import { ExpoWebGLRenderingContext } from "expo-gl";

// Carrega o modelo de machine learning e seu binário correspondente
const modelJson = require("./assets/model/marks3/model.json");
const modelBin1 = require("./assets/model/marks3/group1-shard1of1.bin");

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
      model.current = await tf.loadGraphModel(bundleResourceIO(modelJson, [modelBin1]));
      console.log("Carregou o modelo")
      setModelReady(true);
    } catch (error) {
      console.error("Error loading model: ", error);
    }
  };

  // Ref para manter a referência ao modelo
  const model = useRef<tf.GraphModel | null>(null);

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
          const prediction = model.current?.predict(imageTensorPrep) as tf.Tensor;
          const pred2 = await prediction.data();
          // Atualiza o estado com a saída do modelo
          setModelOutput(`Prediction: ${prediction.toString()}`);
          // Libera recursos do tensor
          model.current?.disposeIntermediateTensors();
          tf.dispose([imageTensorPrep, prediction, pred2]);
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
          ratio={'16:9'}
          pictureSize={"1280x720"}
          style={styles.camera}
          autorender={false}
          type={CameraType.back}
          // Propriedades relacionadas ao tensor
          resizeWidth={384}
          resizeHeight={640}
          useCustomShadersToResize={false}
          cameraTextureHeight={384}
          cameraTextureWidth={640}
          resizeDepth={3}
          rotation={0}
          onReady={handleCameraStream}
          // Outras propriedades
        />)}
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
});
