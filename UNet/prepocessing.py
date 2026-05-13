import cv2
import numpy as np

class MacenkoNormalizer:
    def __init__(self, alpha=1, beta=0.15):
        """
        Initialize Macenko parameters.
        alpha: Percentile for angle limits (removes outliers).
        beta: Optical Density (OD) threshold to ignore background/glass.
        """
        self.alpha = alpha
        self.beta = beta
        self.HERef = None
        self.maxCRef = None

    def _get_stain_vectors(self, I):
        I = I.astype(np.float64)
        
        # Convert RGB to Optical Density (OD)
        OD = -np.log10((I + 1) / 256.0)
        
        # Remove background/white pixels based on the beta threshold
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        
        if len(ODhat) == 0: 
            return None, None

        # Compute eigenvectors using Singular Value Decomposition (SVD)
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        
        # Project data onto the two main eigenvectors
        That = ODhat.dot(eigvecs[:, 1:3])
        
        # Find the robust extreme angles for Hematoxylin and Eosin
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)
        
        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
        
        # Ensure the correct order: Hematoxylin first, Eosin second
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
        if HE[0, 0] > HE[0, 1]:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T
            
        # Calculate maximum concentrations
        Y = np.reshape(OD, (-1, 3)).T
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
        return HE, maxC

    def fit(self, target_img):
        """Extract and save the ideal stain vectors from a reference RGB image."""
        target_img = target_img.reshape((-1, 3))
        self.HERef, self.maxCRef = self._get_stain_vectors(target_img)

    def transform(self, img):
        """Apply the reference stain vectors to normalize the input RGB image."""
        if self.HERef is None or self.maxCRef is None:
            raise ValueError("You must call fit() before transform().")
            
        h, w, c = img.shape
        img_reshaped = img.reshape((-1, 3))
        
        # Get stain vectors of the current image
        HE, maxC = self._get_stain_vectors(img_reshaped)
        
        # If the image is entirely background, return it unaltered
        if HE is None: 
            return img
            
        # Compute concentrations of the current image
        Y = -np.log10((img_reshaped.astype(np.float64) + 1) / 256.0)
        C = np.linalg.lstsq(HE, Y.T, rcond=None)[0]
        
        # Normalize concentrations against the reference maximums
        maxC = np.where(maxC == 0, 1e-6, maxC) 
        C = C * (self.maxCRef / maxC)[:, np.newaxis]
        
        # Reconstruct the image back to RGB space
        Inorm = np.multiply(256, np.exp(-self.HERef.dot(C)))
        return np.clip(Inorm, 0, 255).astype(np.uint8).reshape((h, w, c))


class HistologyPreprocessor:
    def __init__(self, color_normalizer=None):
        """
        Initialize the pipeline. 
        color_normalizer: An already fitted instance of MacenkoNormalizer.
        """
        self.color_normalizer = color_normalizer

    def apply_clahe(self, image):
        """Enhance contrast safely by applying CLAHE only to the Lightness channel (LAB space)."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    def apply_denoising(self, image):
        """Apply a median filter to remove scanning noise while preserving cell edges."""
        return cv2.medianBlur(image, 3)

    def __call__(self, image):
        """
        Execute the full preprocessing pipeline.
        Order: Denoising -> Stain Normalization (if provided) -> Contrast Enhancement.
        """
        img = self.apply_denoising(image)
        
        if self.color_normalizer is not None:
            try:
                img = self.color_normalizer.transform(img)
            except np.linalg.LinAlgError:
                # Fallback: skip normalization if linear algebra fails for this specific tile
                pass 
                
        img = self.apply_clahe(img)
        return img