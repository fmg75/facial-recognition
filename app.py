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
        st.subheader("📷 Cámara en Vivo")
        
        # Mostrar el widget de la cámara
        img_file_buffer = st.camera_input("Toma una foto para reconocimiento facial")
        
        if img_file_buffer is not None:
            # Convertir la imagen capturada a formato PIL
            image = Image.open(img_file_buffer)
            
            # Procesar imagen
            with st.spinner("Procesando imagen..."):
                result_image, faces_info = st.session_state.face_system.process_image_for_recognition(image)
            
            # Mostrar solo el resultado del reconocimiento
            st.subheader("🔍 Resultado del Reconocimiento")
            if result_image is not None:
                st.image(result_image, use_container_width=True)
            else:
                st.error("Error procesando la imagen")
            
            # Mostrar mensaje simple si no se detectaron rostros
            if not faces_info:
                st.info("No se detectaron rostros en la imagen")
    
    else:
        st.warning("⚠️ Carga un diccionario .pkl primero para habilitar las funciones de reconocimiento")

if __name__ == "__main__":
    main()