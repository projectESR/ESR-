import cv2
import numpy as np
from PIL import Image
import io

class BloodImageValidator:
    """
    Validates whether an image is actually a blood test card.
    Uses heuristic-based checks rather than ML.
    Slightly above lenient threshold.
    """
    
    def __init__(self):
        # Color ranges for blood test sections (HSV)
        # Red/Pink range (Anti-A)
        self.red_lower = np.array([0, 50, 50])
        self.red_upper = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])
        
        # Yellow/Cream range (Anti-B)
        self.yellow_lower = np.array([15, 30, 100])
        self.yellow_upper = np.array([40, 200, 255])
        
        # Blue/Teal range (Anti-D/Rh)
        self.blue_lower = np.array([90, 50, 50])
        self.blue_upper = np.array([130, 255, 255])
    
    def validate(self, image_bytes):
        """
        Main validation function.
        Returns: (is_valid: bool, reason: str, confidence: float)
        Slightly above lenient: needs to fail 2+ checks to reject
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_cv is None:
                return False, "Image could not be decoded", 0.0
            
            # Run all checks
            checks = {
                'color_presence': self._check_color_presence(img_cv),
                'background_ratio': self._check_background_ratio(img_cv),
                'pattern_detection': self._check_pattern_detection(img_cv),
                'section_segmentation': self._check_section_segmentation(img_cv),
                'contrast_ratio': self._check_contrast_ratio(img_cv),
                'not_face': self._check_not_face(img_cv)
            }
            
            # Count failures
            failures = sum(1 for v in checks.values() if not v)
            confidence = (len(checks) - failures) / len(checks)
            
            # Slightly above lenient: reject only if 2+ checks fail
            if failures >= 2:
                failed_checks = [k for k, v in checks.items() if not v]
                return False, f"Failed checks: {', '.join(failed_checks)}", confidence
            
            return True, "Valid blood test card image", confidence
            
        except Exception as e:
            return False, f"Validation error: {str(e)}", 0.0
    
    def _check_color_presence(self, img_cv):
        """Check if image contains red, yellow, and blue color ranges"""
        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Red mask (handle wraparound)
        mask_red1 = cv2.inRange(img_hsv, self.red_lower, self.red_upper)
        mask_red2 = cv2.inRange(img_hsv, self.red_lower2, self.red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Yellow mask
        mask_yellow = cv2.inRange(img_hsv, self.yellow_lower, self.yellow_upper)
        
        # Blue mask
        mask_blue = cv2.inRange(img_hsv, self.blue_lower, self.blue_upper)
        
        # Check if each color exists (at least 1% of image pixels)
        total_pixels = img_cv.shape[0] * img_cv.shape[1]
        threshold = total_pixels * 0.01
        
        red_present = cv2.countNonZero(mask_red) > threshold
        yellow_present = cv2.countNonZero(mask_yellow) > threshold
        blue_present = cv2.countNonZero(mask_blue) > threshold
        
        # Need at least 2 of 3 colors (lenient)
        colors_found = sum([red_present, yellow_present, blue_present])
        return colors_found >= 2
    
    def _check_background_ratio(self, img_cv):
        """Verify significant white/light background"""
        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # White/light areas: low saturation, high value
        mask_light = cv2.inRange(img_hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
        
        light_ratio = cv2.countNonZero(mask_light) / (img_cv.shape[0] * img_cv.shape[1])
        
        # Need at least 30% light background
        return light_ratio > 0.30
    
    def _check_pattern_detection(self, img_cv):
        """Check for characteristic oval/circular patterns"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for circular/oval shapes (circularity > 0.5)
        circular_shapes = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Too small
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if 0.5 < circularity < 1.1:  # Roughly circular
                circular_shapes += 1
        
        # Need at least 1 circular pattern
        return circular_shapes >= 1
    
    def _check_section_segmentation(self, img_cv):
        """Attempt to find 3 distinct sections"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count substantial contours (area > 500)
        substantial_contours = sum(1 for cnt in contours if cv2.contourArea(cnt) > 500)
        
        # Need at least 2-3 sections
        return substantial_contours >= 2
    
    def _check_contrast_ratio(self, img_cv):
        """Check contrast between colored areas and background"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Get min and max brightness
        min_val = np.min(gray)
        max_val = np.max(gray)
        contrast = max_val - min_val
        
        # Need reasonable contrast (not a flat image)
        return contrast > 50
    
    def _check_not_face(self, img_cv):
        """Simple check to reject obvious faces/selfies"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Load cascade classifier for face detection
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # If faces detected, reject
            return len(faces) == 0
        except:
            # If cascade fails, pass this check
            return True


# Usage example:
if __name__ == "__main__":
    validator = BloodImageValidator()
    
    with open('test_image.jpg', 'rb') as f:
        image_bytes = f.read()
    
    is_valid, reason, confidence = validator.validate(image_bytes)
    print(f"Valid: {is_valid}")
    print(f"Reason: {reason}")
    print(f"Confidence: {confidence:.2%}")