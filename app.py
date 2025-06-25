import streamlit as st
import cv2
import os
import pickle
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
import tempfile
import threading
import queue
import time
import platform

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

        # Cargar diccionario por defecto si existe
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
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            # Cargar el diccionario
            with open(tmp_file_path, "rb") as f:
                self.caracteristicas = pickle.load(f)
            
            # Limpiar archivo temporal
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
            # Asegurar que tenemos un diccionario cargado
            if not self.caracteristicas:
                return []
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Detectar rostros con MTCNN
            faces, probs = self.mtcnn.detect(pil_image)
            
            if faces is None or len(faces) == 0:
                return []
            
            recognized_faces = []
            
            # Procesar cada rostro detectado
            for i, (face_box, prob) in enumerate(zip(faces, probs)):
                if prob > 0.85:  # Threshold de confianza
                    try:
                        # Extraer características del rostro
                        face_tensor = self.mtcnn(pil_image, save_path=None)
                        
                        if face_tensor is not None:
                            # Manejar múltiples rostros
                            if face_tensor.dim() == 4 and face_tensor.shape[0] > 1:
                                if i < face_tensor.shape[0]:
                                    current_face = face_tensor[i]
                                else:
                                    current_face = face_tensor[0]
                            else:
                                # Un solo rostro o tensor 3D
                                if face_tensor.dim() == 4:
                                    current_face = face_tensor[0]
                                else:
                                    current_face = face_tensor
                            
                            # Reconocer rostro
                            label, similarity = self.recognize_face(current_face)
                            
                            recognized_faces.append({
                                'box': face_box.astype(int),
                                'label': label,
                                'similarity': similarity,
                                'prob': prob * 100
                            })
                            
                    except Exception as e:
                        print(f"Error procesando rostro {i}: {e}")
                        # Agregar como desconocido si hay error
                        recognized_faces.append({
                            'box': face_box.astype(int),
                            'label': "Error",
                            'similarity': 0,
                            'prob': prob * 100
                        })
                        continue
            
            return recognized_faces
            
        except Exception as e:
            print(f"Error general en recognize_faces_in_frame: {e}")
            return []
    
    def draw_face_info(self, frame, faces_info):
        """Dibujar información de rostros en el frame - VERSIÓN CORREGIDA"""
        for face in faces_info:
            # Obtener coordenadas correctas (x1, y1, x2, y2)
            x1, y1, x2, y2 = face['box']
            label = face['label']
            similarity = face['similarity']
            
            # Color según el estado
            if label == "Desconocido":
                color = (0, 0, 255)  # Rojo
            else:
                color = (0, 255, 0)  # Verde
            
            # Dibujar rectángulo CORREGIDO usando (x1, y1) y (x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Texto
            text = f"{label} ({similarity}%)" if similarity > 0 else label
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Calcular posición centrada para el fondo del texto
            text_bg_x1 = x1 + (x2 - x1 - text_size[0]) // 2 - 5
            text_bg_x2 = text_bg_x1 + text_size[0] + 10
            text_bg_y1 = y1 - 30
            text_bg_y2 = y1
            
            # Asegurar que el fondo del texto no se salga de la imagen
            if text_bg_x1 < 0:
                text_bg_x1 = 0
            if text_bg_x2 > frame.shape[1]:
                text_bg_x2 = frame.shape[1]
            
            # Dibujar fondo del texto centrado
            cv2.rectangle(frame, 
                         (text_bg_x1, text_bg_y1), 
                         (text_bg_x2, text_bg_y2), 
                         color, -1)
            
            # Calcular posición centrada para el texto
            text_x = text_bg_x1 + (text_bg_x2 - text_bg_x1 - text_size[0]) // 2
            text_y = y1 - 10
            
            # Dibujar texto centrado
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def save_capture(self, frame, recognized_faces):
        """Guardar captura con nombres de múltiples rostros"""
        if not recognized_faces:
            return False, "No se detectaron rostros para capturar"
        
        try:
            # Crear directorio si no existe
            capture_dir = 'capturas_reconocimiento'
            os.makedirs(capture_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Obtener nombres reconocidos (sin duplicados)
            recognized_names = []
            for face in recognized_faces:
                if face['label'] not in ["Desconocido"] and face['similarity'] > 60:
                    recognized_names.append(face['label'])
            
            # Generar nombre del archivo
            if recognized_names:
                unique_names = list(set(recognized_names))
                if len(unique_names) == 1:
                    # Un solo rostro reconocido
                    filename = f'{capture_dir}/{unique_names[0]}.jpg'
                else:
                    # Múltiples rostros - unir nombres con guion bajo
                    joined_names = '_'.join(sorted(unique_names))
                    filename = f'{capture_dir}/{joined_names}.jpg'
            else:
                # No hay rostros reconocidos
                filename = f'{capture_dir}/Desconocido_{timestamp}.jpg'
            
            # Guardar imagen
            success = cv2.imwrite(filename, frame)
            
            if success and os.path.exists(filename):
                return True, f"Imagen guardada: {filename}"
            else:
                return False, "Error al guardar la imagen"
                
        except Exception as e:
            return False, f"Error inesperado: {str(e)}"

class CameraThread:
    """Clase para manejar la cámara en un hilo separado con detección mejorada"""
    def __init__(self, face_system):
        self.face_system = face_system
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=5)
        self.running = False
        self.thread = None
        self.cap = None
        self.frame_count = 0
        
    def _detect_camera_index(self):
        """Detectar el índice correcto de la cámara"""
        possible_indices = [0, 1, 2, -1]  # Indices comunes
        
        for index in possible_indices:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    print(f"Cámara encontrada en índice: {index}")
                    return index
        
        # Intentar con diferentes backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_GSTREAMER]
        for backend in backends:
            for index in [0, 1]:
                try:
                    cap = cv2.VideoCapture(index, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None:
                            print(f"Cámara encontrada con backend {backend} en índice: {index}")
                            return index, backend
                except:
                    continue
        
        return None
    
    def start(self):
        """Iniciar la cámara y el hilo con detección mejorada"""
        if self.running:
            return False, "La cámara ya está activa"
        
        # Detectar cámara
        camera_info = self._detect_camera_index()
        
        if camera_info is None:
            return False, "No se encontró ninguna cámara disponible"
        
        # Abrir cámara
        if isinstance(camera_info, tuple):
            index, backend = camera_info
            self.cap = cv2.VideoCapture(index, backend)
        else:
            index = camera_info
            self.cap = cv2.VideoCapture(index)
        
        # Verificar que se abrió correctamente
        if not self.cap.isOpened():
            return False, f"No se pudo acceder a la cámara en índice {index}"
        
        # Configurar resolución y FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Probar captura
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            return False, "La cámara no puede capturar imágenes"
        
        print(f"Cámara configurada: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        
        self.running = True
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()
        return True, f"Cámara iniciada correctamente (índice: {index})"
    
    def stop(self):
        """Detener la cámara y el hilo"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        
        # Limpiar colas
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
    
    def _camera_loop(self):
        """Bucle principal de la cámara con manejo de errores mejorado"""
        consecutive_errors = 0
        max_errors = 10
        
        while self.running and consecutive_errors < max_errors:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_errors += 1
                    print(f"Error capturando frame: {consecutive_errors}/{max_errors}")
                    time.sleep(0.1)
                    continue
                
                # Reset error counter si la captura fue exitosa
                consecutive_errors = 0
                
                frame = cv2.flip(frame, 1)
                self.frame_count += 1
                
                # Procesar reconocimiento cada 15 frames (más frecuente)
                if self.frame_count % 15 == 0:
                    try:
                        recognized_faces = self.face_system.recognize_faces_in_frame(frame)
                        
                        # Poner resultado en la cola (no bloqueante)
                        try:
                            self.result_queue.put_nowait({
                                'faces': recognized_faces,
                                'frame_count': self.frame_count
                            })
                        except queue.Full:
                            # Si la cola está llena, sacar el más viejo
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put_nowait({
                                    'faces': recognized_faces,
                                    'frame_count': self.frame_count
                                })
                            except queue.Empty:
                                pass
                    except Exception as e:
                        print(f"Error en reconocimiento: {e}")
                
                # Poner frame en la cola (no bloqueante)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Si la cola está llena, sacar el frame más viejo
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                consecutive_errors += 1
                print(f"Error general en camera_loop: {e} ({consecutive_errors}/{max_errors})")
                time.sleep(0.1)
        
        if consecutive_errors >= max_errors:
            print("Demasiados errores consecutivos, cerrando cámara")
            self.running = False
    
    def get_latest_frame(self):
        """Obtener el frame más reciente"""
        frame = None
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        return frame
    
    def get_latest_results(self):
        """Obtener los resultados más recientes"""
        results = None
        while not self.result_queue.empty():
            try:
                results = self.result_queue.get_nowait()
            except queue.Empty:
                break
        return results

def detect_available_cameras():
    """Detectar cámaras disponibles en el sistema"""
    available_cameras = []
    
    # Probar índices comunes
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(f"Cámara {i}")
            cap.release()
    
    return available_cameras

def main():
    st.set_page_config(
        page_title="Reconocimiento Facial en Tiempo Real",
        page_icon="🎥",
        layout="wide"
    )
    
    # Inicializar estado de sesión
    if 'face_system' not in st.session_state:
        st.session_state.face_system = FaceRecognitionSystem()
        st.session_state.camera_thread = CameraThread(st.session_state.face_system)
        st.session_state.last_results = None
        st.session_state.camera_active = False
        st.session_state.capture_message = None
        st.session_state.dict_loaded = False
        st.session_state.last_uploaded = None

    # Elementos estáticos
    st.title("🎥 Sistema de Reconocimiento Facial")
    st.markdown("---")
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Información del sistema
        st.subheader("🖥️ Sistema")
        st.write(f"**SO:** {platform.system()}")
        st.write(f"**Dispositivo:** {st.session_state.face_system.device}")
        st.write(f"**OpenCV:** {cv2.__version__}")
        
        # Detectar cámaras disponibles
        if st.button("🔍 Detectar Cámaras"):
            with st.spinner("Detectando cámaras..."):
                cameras = detect_available_cameras()
                if cameras:
                    st.success(f"Cámaras encontradas: {', '.join(cameras)}")
                else:
                    st.error("No se encontraron cámaras")
        
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
        
        # Mostrar información del diccionario optimizada para grandes datasets
        if st.session_state.face_system.caracteristicas:
            total_personas = len(st.session_state.face_system.caracteristicas)
            st.success(f"✅ Diccionario cargado correctamente")
            
            # Mostrar solo las primeras 10 personas si hay muchas
            if total_personas > 10:
                st.subheader(f"👥 Personas Registradas (10 de {total_personas})")
                personas = list(st.session_state.face_system.caracteristicas.keys())[:10]
                for persona in personas:
                    st.write(f"• **{persona}**")
                st.info(f"Y {total_personas - 10} personas más...")
            else:
                st.subheader(f"👥 Personas Registradas ({total_personas})")
                for persona in st.session_state.face_system.caracteristicas.keys():
                    st.write(f"• **{persona}**")
        else:
            st.warning("⚠️ No hay diccionario cargado")
        
        st.write(f"**Personas totales:** {len(st.session_state.face_system.caracteristicas)}")
        st.write(f"**Estado cámara:** {'🟢 Activa' if st.session_state.camera_thread.running else '🔴 Inactiva'}")
    
    # Área principal - Crear placeholders para contenido dinámico
    col1, col2 = st.columns([2, 1])
    
    with col1:
        camera_header = st.subheader("📷 Cámara en Tiempo Real")
        camera_placeholder = st.empty()  # Para el video
        
        # Controles
        col_controls = st.columns(4)
        with col_controls[0]:
            start_camera = st.button("🎥 Iniciar Cámara", type="primary")
        with col_controls[1]:
            stop_camera = st.button("⏹️ Detener Cámara")
        with col_controls[2]:
            capture_photo = st.button("📸 Capturar Foto")
        with col_controls[3]:
            clear_captures = st.button("🗑️ Limpiar Capturas")
        
        # Mensaje de captura
        capture_message = st.empty()
    
    with col2:
        results_header = st.subheader("📊 Resultados")
        results_placeholder = st.empty()  # Para resultados
        
        captures_header = st.subheader("📁 Capturas Guardadas")
        captures_placeholder = st.empty()  # Para lista de capturas
    
    # Control de cámara
    if start_camera:
        if not st.session_state.face_system.caracteristicas:
            st.error("⚠️ Carga un diccionario .pkl antes de usar la cámara")
        else:
            with st.spinner("Iniciando cámara..."):
                success, message = st.session_state.camera_thread.start()
                if success:
                    st.session_state.camera_active = True
                    st.success(message)
                else:
                    st.error(message)
    
    if stop_camera:
        st.session_state.camera_thread.stop()
        st.session_state.camera_active = False
        st.info("Cámara detenida")
        camera_placeholder.empty()
        results_placeholder.empty()
    
    # Limpiar capturas
    if clear_captures:
        try:
            capture_dir = 'capturas_reconocimiento'
            if os.path.exists(capture_dir):
                files = [f for f in os.listdir(capture_dir) if f.endswith(('.jpg', '.png'))]
                for filename in files:
                    os.remove(os.path.join(capture_dir, filename))
                st.success(f"Se eliminaron {len(files)} capturas")
            else:
                st.info("No hay capturas para eliminar")
        except Exception as e:
            st.error(f"Error al limpiar capturas: {e}")
    
    # Bucle de actualización cuando la cámara está activa
    while st.session_state.camera_active:
        # Obtener frame más reciente
        frame = st.session_state.camera_thread.get_latest_frame()
        
        if frame is not None:
            # Obtener resultados más recientes
            new_results = st.session_state.camera_thread.get_latest_results()
            if new_results:
                st.session_state.last_results = new_results['faces']
            
            # Usar los últimos resultados disponibles
            current_faces = st.session_state.last_results or []
            
            # Actualizar sección de resultados
            with results_placeholder.container():
                if current_faces:
                    results_text = "**Rostros Detectados:**\n"
                    for i, face in enumerate(current_faces, 1):
                        status = "✅" if face['similarity'] > 60 else "❓"
                        results_text += f"{status} {face['label']} ({face['similarity']}%)\n"
                    st.markdown(results_text)
                else:
                    st.markdown("**Buscando rostros...**")
            
            # Dibujar información en el frame
            frame_with_faces = st.session_state.face_system.draw_face_info(frame, current_faces)
            
            # Capturar foto si se presionó el botón
            if capture_photo:
                if current_faces:
                    success, message = st.session_state.face_system.save_capture(frame_with_faces, current_faces)
                    st.session_state.capture_message = (success, message)
                else:
                    st.session_state.capture_message = (False, "No hay rostros detectados para capturar")
            
            # Mostrar mensaje de captura si existe
            if st.session_state.capture_message:
                success, message = st.session_state.capture_message
                if success:
                    capture_message.success(message)
                else:
                    capture_message.error(message)
                # Limpiar mensaje después de mostrar
                st.session_state.capture_message = None
            
            # Mostrar frame
            frame_rgb = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Actualizar sección de capturas
            with captures_placeholder.container():
                try:
                    capture_dir = 'capturas_reconocimiento'
                    if os.path.exists(capture_dir):
                        files = [f for f in os.listdir(capture_dir) if f.endswith(('.jpg', '.png'))]
                        if files:
                            st.markdown(f"**Total: {len(files)} capturas**")
                            # Mostrar miniaturas de las últimas 3 capturas
                            cols = st.columns(3)
                            for i, file in enumerate(sorted(files, reverse=True)[:3]):
                                with cols[i % 3]:
                                    img_path = os.path.join(capture_dir, file)
                                    st.image(img_path, caption=file, width=100)
                        else:
                            st.markdown("**No hay capturas guardadas**")
                    else:
                        st.markdown("**No hay capturas guardadas**")
                except Exception as e:
                    st.markdown(f"**Error:** {str(e)}")
            
            # Control de FPS sin recargar toda la página
            time.sleep(0.05)  # ~20 FPS
        else:
            # Si no hay frame, esperar un poco antes de reintentar
            time.sleep(0.1)
    
    # Mostrar instrucciones cuando la cámara no está activa
    if not st.session_state.camera_active:
        with col1:
            st.markdown("""
            ### 📋 Instrucciones de Uso:
            
            1. **Detectar Cámaras**: Usa el botón para verificar qué cámaras están disponibles
            2. **Cargar Diccionario**: Sube un archivo .pkl con las características faciales entrenadas
            3. **Iniciar Cámara**: Activa la cámara web para reconocimiento en tiempo real
            4. **Capturar Foto**: Guarda una imagen cuando se detecten rostros
            
            ### 🔧 Solución de Problemas:
            - **Cámara no detectada**: Verifica que no esté siendo usada por otra aplicación
            - **Permisos**: Asegúrate de que la aplicación tenga permisos para acceder a la cámara
            - **Múltiples cámaras**: El sistema probará diferentes índices automáticamente
            - **Drivers**: Verifica que los drivers de la cámara estén actualizados
            
            ### 🎯 Características:
            - ✅ Detección automática de cámaras
            - ✅ Múltiples backends de OpenCV
            - ✅ Manejo robusto de errores
            - ✅ Reconocimiento en tiempo real
            - ✅ Interfaz fluida sin parpadeos
            - ✅ Optimizado para grandes diccionarios
            """)

if __name__ == "__main__":
    main()