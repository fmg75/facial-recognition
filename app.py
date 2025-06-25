import streamlit as st
import cv2
import os
import pickle
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import platform
import tempfile
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

class FaceRecognitionSystem:
    def __init__(self, caracteristicas_dict=None):
        """Inicializar el sistema de reconocimiento facial"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=True,
            device=self.device
        )
        
        self.caracteristicas = caracteristicas_dict or {}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.load_default_dictionary()
        
    def load_default_dictionary(self):
        """Cargar diccionario por defecto al inicializar"""
        default_path = "feature_dict.pkl"
        if os.path.exists(default_path):
            try:
                with open(default_path, "rb") as f:
                    self.caracteristicas = pickle.load(f)
                print(f"Diccionario por defecto cargado con {len(self.caracteristicas)} personas")
            except Exception as e:
                print(f"Error cargando diccionario por defecto: {e}")
                self.caracteristicas = {}
        else:
            print("No se encontró diccionario por defecto")

    def load_caracteristicas_from_file(self, file_content):
        """Cargar diccionario de características desde archivo subido"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, "rb") as f:
                self.caracteristicas = pickle.load(f)
            
            os.unlink(tmp_file_path)
            return True, f"Diccionario cargado exitosamente con {len(self.caracteristicas)} personas"
        except Exception as e:
            return False, f"Error cargando diccionario: {str(e)}"
    
    def get_embedding(self, face_tensor):
        """Obtener embedding de un rostro"""
        with torch.no_grad():
            face_tensor = face_tensor.to(self.device)
            embedding = self.model(face_tensor.unsqueeze(0))
            return embedding
    
    def recognize_face(self, face_tensor):
        """Reconocer rostro comparando con diccionario"""
        if not self.caracteristicas:
            return "Desconocido", 0.0
            
        face_embedding = self.get_embedding(face_tensor)
        
        distances = []
        for label, stored_embedding in self.caracteristicas.items():
            stored_embedding = stored_embedding.to(self.device)
            distance = torch.dist(face_embedding, stored_embedding).item()
            distances.append((label, distance))
        
        best_match = min(distances, key=lambda x: x[1])
        label, distance = best_match
        
        similarity = max(0, int(100 - 17.14 * distance))
        
        if similarity > 60:
            return label, similarity
        else:
            return "Desconocido", similarity
    
    def recognize_faces_in_frame(self, frame):
        """Reconocer rostros en un frame"""
        try:
            if not self.caracteristicas:
                return []
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            faces, probs = self.mtcnn.detect(pil_image)
            
            if faces is None or len(faces) == 0:
                return []
            
            recognized_faces = []
            
            for i, (face_box, prob) in enumerate(zip(faces, probs)):
                if prob > 0.85:
                    try:
                        face_tensor = self.mtcnn(pil_image, save_path=None)
                        
                        if face_tensor is not None:
                            if face_tensor.dim() == 4 and face_tensor.shape[0] > 1:
                                current_face = face_tensor[i] if i < face_tensor.shape[0] else face_tensor[0]
                            else:
                                current_face = face_tensor[0] if face_tensor.dim() == 4 else face_tensor
                            
                            label, similarity = self.recognize_face(current_face)
                            
                            recognized_faces.append({
                                'box': face_box.astype(int),
                                'label': label,
                                'similarity': similarity,
                                'prob': prob * 100
                            })
                            
                    except Exception as e:
                        recognized_faces.append({
                            'box': face_box.astype(int),
                            'label': "Error",
                            'similarity': 0,
                            'prob': prob * 100
                        })
                        continue
            
            return recognized_faces
        except Exception as e:
            return []
    
    def draw_face_info(self, frame, faces_info):
        """Dibujar información de rostros en el frame"""
        for face in faces_info:
            x1, y1, x2, y2 = face['box']
            label = face['label']
            similarity = face['similarity']
            
            color = (0, 0, 255) if label == "Desconocido" else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            text = f"{label} ({similarity}%)" if similarity > 0 else label
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            text_bg_x1 = x1 + (x2 - x1 - text_size[0]) // 2 - 5
            text_bg_x2 = text_bg_x1 + text_size[0] + 10
            text_bg_y1 = y1 - 30
            text_bg_y2 = y1
            
            if text_bg_x1 < 0:
                text_bg_x1 = 0
            if text_bg_x2 > frame.shape[1]:
                text_bg_x2 = frame.shape[1]
            
            cv2.rectangle(frame, 
                         (text_bg_x1, text_bg_y1), 
                         (text_bg_x2, text_bg_y2), 
                         color, -1)
            
            text_x = text_bg_x1 + (text_bg_x2 - text_bg_x1 - text_size[0]) // 2
            text_y = y1 - 10
            
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def process_image_for_recognition(self, image):
        """Procesar imagen subida para reconocimiento"""
        try:
            # Convertir imagen PIL a array numpy
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Asegurar formato BGR para OpenCV
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Si es RGB, convertir a BGR
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_array
            
            # Reconocer rostros
            faces_info = self.recognize_faces_in_frame(image_bgr)
            
            # Dibujar información en la imagen
            result_image = self.draw_face_info(image_bgr.copy(), faces_info)
            
            # Convertir de BGR a RGB para mostrar en Streamlit
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            return result_image_rgb, faces_info
            
        except Exception as e:
            st.error(f"Error procesando imagen: {str(e)}")
            return None, []


# Clase para el procesamiento de video en tiempo real
class VideoProcessor:
    def __init__(self, face_system):
        self.face_system = face_system
        self.last_capture = None
        self.capture_flag = False
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Si hay diccionario cargado, procesar reconocimiento
        if self.face_system.caracteristicas:
            faces_info = self.face_system.recognize_faces_in_frame(img)
            img = self.face_system.draw_face_info(img, faces_info)
            
            # Capturar frame si se solicita
            if self.capture_flag:
                self.last_capture = img.copy()
                self.capture_flag = False
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def capture_frame(self):
        self.capture_flag = True
        time.sleep(0.5)  # Esperar un poco para capturar
        return self.last_capture if self.last_capture is not None else None


def capture_camera_frame():
    """Capturar un frame de la cámara usando OpenCV (para uso local)"""
    try:
        # Intentar diferentes índices de cámara
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                break
        else:
            return None, "No se encontró ninguna cámara disponible"
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Intentar capturar algunos frames para estabilizar la cámara
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
        
        cap.release()
        
        if ret and frame is not None:
            return frame, None
        else:
            return None, "No se pudo capturar el frame de la cámara"
    except Exception as e:
        return None, f"Error accediendo a la cámara: {str(e)}"


def main():
    st.set_page_config(
        page_title="Reconocimiento Facial",
        page_icon="🎥",
        layout="wide"
    )
    
    # Inicializar estado de sesión
    if 'face_system' not in st.session_state:
        st.session_state.face_system = FaceRecognitionSystem()
        st.session_state.dict_loaded = False
        st.session_state.last_uploaded = None
        st.session_state.camera_mode = "upload"  # "upload", "camera_local", "camera_web"

    # Elementos estáticos
    st.title("🎥 Sistema de Reconocimiento Facial")
    st.markdown("---")
    
    # Detectar entorno
    try:
        is_cloud = any([
            os.getenv('STREAMLIT_SHARING_MODE'),
            os.getenv('STREAMLIT_CLOUD'),
            'streamlit.app' in os.getenv('STREAMLIT_SERVER_ADDRESS', ''),
            'herokuapp.com' in os.getenv('STREAMLIT_SERVER_ADDRESS', ''),
        ])
        is_local = not is_cloud
        
        # Intentar detectar disponibilidad de cámara local
        camera_local_available = is_local
        if is_local:
            try:
                cap = cv2.VideoCapture(0)
                camera_local_available = cap.isOpened()
                if cap.isOpened():
                    cap.release()
            except:
                camera_local_available = False
                
    except:
        is_local = True
        camera_local_available = True
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Información del sistema
        st.subheader("🖥️ Sistema")
        st.write(f"**SO:** {platform.system()}")
        st.write(f"**Dispositivo:** {st.session_state.face_system.device}")
        st.write(f"**OpenCV:** {cv2.__version__}")
        st.write(f"**Entorno:** {'Local' if is_local else 'Cloud/Desplegado'}")
        st.write(f"**Cámara local:** {'Sí' if camera_local_available else 'No'}")
        
        # Cargar diccionario .pkl
        st.subheader("📁 Cargar Diccionario")
        uploaded_file = st.file_uploader(
            "Selecciona un archivo .pkl",
            type=['pkl'],
            help="Archivo con las características faciales entrenadas",
            key="pkl_uploader"
        )
        
        # Auto-cargar cuando se sube un archivo
        if uploaded_file is not None:
            if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
                st.session_state.last_uploaded = uploaded_file.name
                with st.spinner("Cargando diccionario..."):
                    success, message = st.session_state.face_system.load_caracteristicas_from_file(
                        uploaded_file.read()
                    )
                    if success:
                        st.success(message)
                        st.session_state.dict_loaded = True
                    else:
                        st.error(message)
                        st.session_state.dict_loaded = False
        
        # Mostrar información del diccionario
        if st.session_state.face_system.caracteristicas:
            total_personas = len(st.session_state.face_system.caracteristicas)
            st.success(f"✅ Diccionario cargado correctamente")
            
            if total_personas > 10:
                st.subheader(f"👥 Personas Registradas (10 de {total_personas})")
                for persona in list(st.session_state.face_system.caracteristicas.keys())[:10]:
                    st.write(f"• **{persona}**")
                st.info(f"Y {total_personas - 10} personas más...")
            else:
                st.subheader(f"👥 Personas Registradas ({total_personas})")
                for persona in st.session_state.face_system.caracteristicas.keys():
                    st.write(f"• **{persona}**")
        else:
            st.warning("⚠️ No hay diccionario cargado")
        
        st.write(f"**Personas totales:** {len(st.session_state.face_system.caracteristicas)}")
    
    # Solo mostrar si hay diccionario cargado
    if st.session_state.face_system.caracteristicas:
        # Selector de modo
        st.subheader("📷 Modo de Captura")
        
        # Crear columnas para los botones
        if is_local and camera_local_available:
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2 = st.columns(2)
        
        with col1:
            upload_mode = st.button("📁 Subir Imagen", use_container_width=True)
        
        with col2:
            camera_web_mode = st.button("🌐 Cámara Web", use_container_width=True,
                                      help="Cámara web que funciona en cualquier entorno")
        
        if is_local and camera_local_available:
            with col3:
                camera_local_mode = st.button("📸 Cámara Local", use_container_width=True,
                                            help="Captura directa con OpenCV (solo local)")
        
        # Actualizar modo según botón presionado
        if upload_mode:
            st.session_state.camera_mode = "upload"
        elif camera_web_mode:
            st.session_state.camera_mode = "camera_web"
        elif is_local and camera_local_available and 'camera_local_mode' in locals() and camera_local_mode:
            st.session_state.camera_mode = "camera_local"
        
        st.markdown("---")
        
        # Modo subir imagen
        if st.session_state.camera_mode == "upload":
            st.subheader("📁 Subir Imagen para Reconocimiento")
            
            uploaded_image = st.file_uploader(
                "Selecciona una imagen",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Formatos soportados: JPG, JPEG, PNG, BMP"
            )
            
            if uploaded_image is not None:
                try:
                    # Cargar imagen
                    image = Image.open(uploaded_image)
                    
                    # Mostrar imagen original
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📷 Imagen Original")
                        st.image(image, use_container_width=True)
                    
                    # Procesar imagen
                    with st.spinner("Procesando imagen..."):
                        result_image, faces_info = st.session_state.face_system.process_image_for_recognition(image)
                    
                    with col2:
                        st.subheader("🔍 Resultado del Reconocimiento")
                        if result_image is not None:
                            st.image(result_image, use_container_width=True)
                        else:
                            st.error("Error procesando la imagen")
                    
                    # Mostrar resultados detallados
                    if faces_info:
                        st.subheader("📊 Resultados Detallados")
                        for i, face in enumerate(faces_info, 1):
                            status = "✅ Reconocido" if face['similarity'] > 60 else "❓ Desconocido"
                            confidence = "Alta" if face['similarity'] > 80 else "Media" if face['similarity'] > 60 else "Baja"
                            
                            with st.expander(f"Rostro {i}: {face['label']} ({status})"):
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Similitud", f"{face['similarity']}%")
                                with col_b:
                                    st.metric("Confianza", confidence)
                                with col_c:
                                    st.metric("Detección", f"{face['prob']:.1f}%")
                    else:
                        st.info("No se detectaron rostros en la imagen")
                        
                except Exception as e:
                    st.error(f"Error cargando imagen: {str(e)}")
        
        # Modo cámara web (funciona en cualquier entorno)
        elif st.session_state.camera_mode == "camera_web":
            st.subheader("🌐 Cámara Web en Tiempo Real")
            
            # Configuración WebRTC
            RTC_CONFIGURATION = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })
            
            # Inicializar procesador de video
            if 'video_processor' not in st.session_state:
                st.session_state.video_processor = VideoProcessor(st.session_state.face_system)
            
            # Streamlit WebRTC
            webrtc_ctx = webrtc_streamer(
                key="face-recognition",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: st.session_state.video_processor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            if webrtc_ctx.video_processor:
                st.info("✅ Cámara web conectada. Los rostros se reconocen en tiempo real.")
                
                # Botón para capturar frame
                if st.button("📸 Capturar Frame Actual"):
                    captured_frame = st.session_state.video_processor.capture_frame()
                    if captured_frame is not None:
                        st.session_state.web_captured_frame = captured_frame
                        st.success("¡Frame capturado!")
                
                # Mostrar frame capturado y análisis
                if 'web_captured_frame' in st.session_state:
                    st.subheader("📷 Frame Capturado")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Imagen Capturada:**")
                        # Convertir BGR a RGB para mostrar
                        frame_rgb = cv2.cvtColor(st.session_state.web_captured_frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, use_container_width=True)
                    
                    with col2:
                        st.write("**Análisis de Rostros:**")
                        faces_info = st.session_state.face_system.recognize_faces_in_frame(
                            st.session_state.web_captured_frame
                        )
                        
                        if faces_info:
                            for i, face in enumerate(faces_info, 1):
                                status = "✅" if face['similarity'] > 60 else "❓"
                                st.write(f"{status} **{face['label']}**")
                                st.write(f"   Similitud: {face['similarity']}%")
                                st.write(f"   Detección: {face['prob']:.1f}%")
                                st.write("---")
                        else:
                            st.info("No se detectaron rostros en el frame")
            else:
                st.warning("⚠️ Conectando con la cámara web...")
                st.info("""
                **Para usar la cámara web:**
                1. Permite el acceso a la cámara cuando el navegador lo solicite
                2. Espera a que aparezca el video
                3. Los rostros se reconocerán automáticamente
                4. Usa 'Capturar Frame' para analizar un momento específico
                """)
        
        # Modo cámara local (solo en entorno local)
        elif st.session_state.camera_mode == "camera_local" and is_local and camera_local_available:
            st.subheader("📸 Captura con Cámara Local")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("📷 Capturar Foto", use_container_width=True):
                    with st.spinner("Capturando imagen..."):
                        frame, error = capture_camera_frame()
                        
                        if frame is not None:
                            st.session_state.captured_frame = frame
                            st.success("¡Imagen capturada!")
                        else:
                            st.error(f"Error: {error}")
                            st.info("""
                            **Posibles soluciones:**
                            - Verifica que tu cámara esté conectada
                            - Cierra otras aplicaciones que usen la cámara
                            - Reinicia la aplicación
                            """)
            
            # Mostrar imagen capturada y procesarla
            if 'captured_frame' in st.session_state:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.subheader("📷 Imagen Capturada")
                    # Convertir BGR a RGB para mostrar
                    frame_rgb = cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, use_container_width=True)
                
                with col_b:
                    st.subheader("🔍 Resultado del Reconocimiento")
                    with st.spinner("Analizando rostros..."):
                        faces_info = st.session_state.face_system.recognize_faces_in_frame(
                            st.session_state.captured_frame
                        )
                        
                        if faces_info:
                            result_frame = st.session_state.face_system.draw_face_info(
                                st.session_state.captured_frame.copy(), faces_info
                            )
                            result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                            st.image(result_frame_rgb, use_container_width=True)
                        else:
                            st.image(frame_rgb, use_container_width=True)
                            st.info("No se detectaron rostros")
                
                # Resultados detallados
                if faces_info:
                    st.subheader("📊 Resultados de la Captura")
                    for i, face in enumerate(faces_info, 1):
                        status = "✅" if face['similarity'] > 60 else "❓"
                        st.write(f"{status} **{face['label']}** - Similitud: {face['similarity']}% - Detección: {face['prob']:.1f}%")
    
    else:
        st.warning("⚠️ Carga un diccionario .pkl primero para habilitar las funciones de reconocimiento")
        
        # Mostrar ejemplo de cómo usar
        with st.expander("📖 Cómo usar esta aplicación"):
            st.write("""
            **Pasos para usar el sistema:**
            
            1. **Cargar Diccionario**: Sube un archivo .pkl con las características faciales entrenadas
            2. **Seleccionar Modo**: 
               - **Subir Imagen**: Sube una foto desde tu dispositivo
               - **Cámara Web**: Usa la cámara en tiempo real (funciona en cualquier entorno)
               - **Cámara Local**: Captura directa con OpenCV (solo local)
            3. **Analizar**: La aplicación detectará y reconocerá rostros automáticamente
            
            **Formatos soportados:**
            - Diccionario: archivos .pkl
            - Imágenes: JPG, JPEG, PNG, BMP
            
            **Modos de cámara:**
            - **Cámara Web**: Reconocimiento en tiempo real usando WebRTC, funciona en todos los entornos
            - **Cámara Local**: Captura directa, solo disponible cuando se ejecuta localmente
            """)


if __name__ == "__main__":
    main()