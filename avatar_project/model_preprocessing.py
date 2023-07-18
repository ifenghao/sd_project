import os
import shutil
import dlib
import cv2

class ModelPreprocessing:
    def __init__(self, logger):
        self.logger = logger
        self.detector = dlib.get_frontal_face_detector()

    def crop_face_from_path(self, input_path, crop_path, scales=[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8]) : 
        crop_num = 0 
        for root, dirs, files in os.walk(input_path):
            for file_name in files: 
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path    = os.path.join(root, file_name)
                    for scale in scales:
                        cropped_image = self.extract_square_face_and_shoulders(image_path, scale=scale) 
                        scale_crop_path = crop_path + str(scale)
                        if cropped_image is not None : 
                            crop_num +=1
                            os.makedirs(scale_crop_path, exist_ok=True) 
                            self.save_cropped_image(os.path.join(scale_crop_path, file_name),  cropped_image) 
                        else :
                            self.copy_image_to_folder(scale_crop_path, image_path)   
        return crop_num
    
    def crop_face_from_path_auto_scale(self, input_path, crop_path, scales_num=10, min_scale=0.8) : 
        copy_num = 0
        crop_num = 0
        for root, dirs, files in os.walk(input_path):
            for file_name in files: 
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path    = os.path.join(root, file_name)
                    image, face = self.get_face_detect_result(image_path)
                    if face is None:
                        self.copy_image_to_folder(crop_path, image_path)
                        copy_num += 1
                    else:
                        max_scale = self.get_max_face_scale(image, face)
                        scale_step = (max_scale - min_scale) / scales_num
                        for i in range(scales_num):
                            cropped_image = self.get_square_face_at_scale(image, face, min_scale + i * scale_step)
                            scale_crop_path = crop_path + str(i)
                            os.makedirs(scale_crop_path, exist_ok=True)
                            self.save_cropped_image(os.path.join(scale_crop_path, file_name),  cropped_image)
                            crop_num +=1 
        return copy_num, crop_num

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
    
    def extract_square_face_and_shoulders(self, image_path, scale=3):
        try : 
            image = cv2.imread(image_path) 
            gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) != 1 : 
                return None 
            else : 
                face = faces[0] 
                left, top, w, h = face.left(), face.top(), face.width(), face.height()
                side = max(w, h)
                # 扩展裁剪区域
                padding = int(side * (scale - 1) / 2)
                left_bound = left - padding
                top_bound = top - padding
                right_bound = left + side + padding
                bottom_bound = top + side + padding
                max_diff = 0
                if left_bound < 0:
                    max_diff = max(max_diff, -left_bound)
                if top_bound < 0:
                    max_diff = max(max_diff, -top_bound)
                if right_bound > image.shape[1] - 1:
                    max_diff = max(max_diff, right_bound - image.shape[1] + 1)
                if bottom_bound > image.shape[0] - 1:
                    max_diff = max(max_diff, bottom_bound - image.shape[0] + 1)
                left_bound += max_diff
                top_bound += max_diff
                right_bound -= max_diff
                bottom_bound -= max_diff
                # 裁剪出脸部和肩膀区域
                cropped_face = image[top_bound:bottom_bound, left_bound:right_bound]
                return cropped_face
        except Exception as e:
            self.logger.error(f"截取头像失败:{e},{image_path}")
            return None

    def get_face_detect_result(self, image_path, min_face_reso=32):
        try : 
            image = cv2.imread(image_path) 
            gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) != 1 : 
                return None, None 
            else:
                face = faces[0]
                w, h = face.width(), face.height()
                if w <= min_face_reso and h <= min_face_reso:
                    return None, None
                return image, face
        except Exception as e:
            self.logger.error(f"截取头像失败:{e},{image_path}")
            return None, None
        
    def get_max_face_scale(self, image, face):
        left, top, w, h = face.left(), face.top(), face.width(), face.height()
        image_width, image_height = image.shape[1], image.shape[0]
        side = max(w, h)
        left_max_pad = left
        right_max_pad = image_width - 1 - (left + side)
        top_max_pad = top
        bottom_max_pad = image_height - 1 - (top + side)
        max_pad_min = min(left_max_pad, right_max_pad, top_max_pad, bottom_max_pad)
        max_scale = 2 * max_pad_min / side + 1
        return max_scale
    
    def get_square_face_at_scale(self, image, face, scale):
        left, top, w, h = face.left(), face.top(), face.width(), face.height()
        side = max(w, h)
        # 扩展裁剪区域
        padding = int(side * (scale - 1) / 2)
        left_bound = left - padding
        top_bound = top - padding
        right_bound = left + side + padding
        bottom_bound = top + side + padding
        cropped_face = image[top_bound:bottom_bound, left_bound:right_bound]
        return cropped_face 

    def copy_image_from_path(self, input_path, copy_path):
        copy_num = 0
        for root, dirs, files in os.walk(input_path):
            for file_name in files: 
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    copy_num += 1
                    image_path    = os.path.join(root, file_name)
                    self.copy_image_to_folder(copy_path, image_path)
        return copy_num
    
    def copy_image_to_folder(self, destination_folder, source_file):
        try:        
            shutil.copy2(source_file, destination_folder)
        except Exception as e:
            self.logger.error(f"Error copying file:{e}") 

    def save_cropped_image(self, cropped_image_path, cropped_image): 
        try:
            cv2.imwrite(cropped_image_path, cropped_image,[cv2.IMWRITE_JPEG_QUALITY, 100])
        except Exception as e:
            self.logger.error(f"Error saving cropped image: {e}")     

