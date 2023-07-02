import os
import shutil
import dlib
import cv2

class ModelPreprocessing:
    def __init__(self, logger):
        self.logger = logger
        self.detector = dlib.get_frontal_face_detector()

    def crop_face_from_path(self, input_path, crop_path) : 
        crop_num = 0 
        for root, dirs, files in os.walk(input_path):
            for file_name in files: 
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path    = os.path.join(root, file_name) 
                    cropped_image = self.extract_face_and_shoulders(image_path, scale_x=2.8, scale_y=3)
                    if cropped_image is not None : 
                        crop_num +=1 
                        cropped_image_save_path = os.path.join(crop_path, file_name) 
                        self.save_cropped_image(cropped_image_save_path,  cropped_image) 
                    else :
                        self.copy_image_to_folder(crop_path, image_path)   
        return crop_num

    def extract_face_and_shoulders(self, image_path, scale_x=2.8, scale_y=3):
        try : 
            image = cv2.imread(image_path) 
            gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) != 1 : 
                return None 
            else : 
                face = faces[0] 
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                padding_x = int(w * (scale_x - 1) / 2)
                padding_y = int(h * (scale_y - 1) / 2)
                # 扩展裁剪区域
                x -= padding_x
                y -= padding_y
                w += 2 * padding_x
                h += 2 * padding_y
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                # 裁剪出脸部和肩膀区域
                cropped_face = image[y:y+h, x:x+w]
                return cropped_face
        except Exception as e:
            self.logger.error(f"截取头像失败:{e},{image_path}")  
            return None 

    def copy_image_to_folder(self, destination_folder, source_file):
        try:        
            shutil.copy2(source_file, destination_folder)
        except Exception as e:
            self.logger.error(f"Error copying file:{e}") 

    def save_cropped_image(self, cropped_image_path, cropped_image): 
        try:
            cv2.imwrite(cropped_image_path, cropped_image)
        except Exception as e:
            self.logger.error(f"Error saving cropped image: {e}")     

