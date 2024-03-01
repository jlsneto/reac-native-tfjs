// Importações das bibliotecas necessárias
import React, {useState, useEffect, useRef, useCallback} from 'react';
import {StyleSheet, Text, View, Dimensions, Platform, TouchableOpacity, Image} from 'react-native';
import {BarCodeScanningResult, Camera, CameraType, Point} from 'expo-camera';
import {StatusBar} from 'expo-status-bar';
import {bundleResourceIO, cameraWithTensors} from "@tensorflow/tfjs-react-native";
import * as tf from "@tensorflow/tfjs";
import {ExpoWebGLRenderingContext} from "expo-gl";
import {BarCodeScanner} from "expo-barcode-scanner";
import Loading from "./Loading";

const localImageSource = require('./assets/CaptureButton.png')
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
  const [qrCodeData, setQrCodeData] = useState<string>('Dados do QrCode: não identificado.');
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  let camera: Camera | null = null
  // Solicita permissão para a câmera e prepara o modelo na montagem do componente
  useEffect(() => {
    (async () => {
      const {status} = await Camera.requestCameraPermissionsAsync();
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

  /**
   * Calback é executado toda vez que a leitura do QRCode é bem sucedida
   **/
  const handleBarCodeScanned = useCallback(async (
    scanningResult: BarCodeScanningResult,
  ) => {

    const cornerPoints: Point[] = scanningResult.cornerPoints
    setQrCodeData(`Dados do QrCode: ${scanningResult.data}`)
  }, [modelReady])

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
    return <View/>;
  }
  if (!hasPermission) {
    return <Text>No access to camera</Text>;
  }

  const uploadImage = (imageUrl: string) => {
    const formData: FormData = new FormData();
    formData.append('image', {
      uri: imageUrl,
      type: 'image/jpeg',
      name: 'cropped_image.jpg',
    });
    fetch(
      'https://e918-2804-1b1-220c-153c-bd53-1972-7d00-6c34.ngrok-free.app/scanner',
      {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      },
    )
      .then(response => response.json())
      .then(data => {
        setCapturedImage(data["image_marked"])
        setIsLoading(false)
        // console.log('Resposta da API:', data);
      })
      .catch(error => {
        console.log('Erro ao enviar a imagem para a API:', error);
      });
  };
  const cleanImage = async () => {
    setCapturedImage(null)
  }
  const takePicture = async () => {
    try {
      console.log('LOGCAT_TAG: Trying to use Native Module.....')

      camera?.takePictureAsync({base64: true}).then(picture => {

        if (picture.base64) {
          console.log("Tirou foto")
          console.log(picture.uri)
          setIsLoading(true)
          uploadImage(picture.uri)

          // const predictions = model.current?.predict(tf.expandDims(tf.image.rgbToGrayscale(imageTensor), 0)) as tf.Tensor
          // console.log("predicionts shape", predictions.shape);
        }
      })
    } catch (error) {
      console.error('An camera error occurred:', error)
    }
  }

  // Renderiza a câmera e a saída do modelo
  return (
    <View style={styles.container}>
      <Loading isLoading={isLoading} statusText="Loading..." />
      {(capturedImage && !isLoading) ? (
        <View>
        <Image
          source={{ uri: capturedImage }}
          style={styles.capturedImage}
        />
          <View style={styles.buttonContainer}>
            <View style={styles.button}>
              <TouchableOpacity
                onPress={cleanImage}
                style={{display: 'flex'}}
              >
                <View style={{padding: 2}}>
                  <Image source={localImageSource}/>
                </View>
                <Text style={styles.text}>Escanear</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      ) : (modelReady && !isLoading) && (
        <Camera
          // Configurações do componente da câmera
          barCodeScannerSettings={{
            barCodeTypes: [BarCodeScanner.Constants.BarCodeType.qr],
          }}
          onBarCodeScanned={handleBarCodeScanned}
          ratio={'16:9'}
          // pictureSize={"1280x720"}
          style={styles.camera}
          type={CameraType.back}
          ref={r => {
            camera = r
          }}

        >
          <View style={styles.buttonContainer}>
            <View style={styles.button}>
              <TouchableOpacity
                onPress={takePicture}
                style={{display: 'flex'}}
              >
                <View style={{padding: 2}}>
                  <Image source={localImageSource}/>
                </View>
                <Text style={styles.text}>Escanear</Text>
              </TouchableOpacity>
            </View>
          </View>
          <Text style={styles.qrCodeText}>{qrCodeData}</Text>
          <Text style={styles.predictionText}>{modelOutput}</Text>
          <StatusBar style="auto" />
        </Camera>)}
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
  buttonContainer: {
    flex: 1,
    alignSelf: 'center',
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 24,
  },
  button: {
    flex: 2,
    alignSelf: 'flex-end',
    alignItems: 'center',
  },
  text: {
    fontSize: 16,
    color: '#0C326F',
  },
  capturedImage: {
    // width: "100%",
    top: 50,
    // width: "50%",
    height: 700,
    // padding: 100,
    margin: 20
  },
});
