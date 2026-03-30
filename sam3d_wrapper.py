# SAM3D inference wrapper for GIC integration
# Handles: model loading, per-frame inference, local->camera->world coordinate transforms

import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from plyfile import PlyData


def rotation_6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    
    Args:
        r6d: (6,) or (B, 6) tensor
    Returns:
        (3, 3) or (B, 3, 3) rotation matrix
    """
    squeeze = r6d.dim() == 1
    if squeeze:
        r6d = r6d.unsqueeze(0)
    
    a1 = r6d[:, 0:3]
    a2 = r6d[:, 3:6]
    
    # Gram-Schmidt orthogonalization
    b1 = a1 / a1.norm(dim=-1, keepdim=True)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    
    R = torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)
    if squeeze:
        R = R.squeeze(0)
    return R


class SAM3DWrapper:
    """
    Wraps SAM3D inference model for use in GIC pipeline.
    
    Usage:
        wrapper = SAM3DWrapper(config_path="checkpoints/hf/pipeline.yaml")
        xyz_world = wrapper.reconstruct_frame(image, mask, cam_R, cam_T)
    """
    
    def __init__(
        self, 
        config_path: str = "checkpoints/hf/pipeline.yaml",
        sam3d_repo_path: str = None,
        device: str = "cuda",
    ):
        """
        Args:
            config_path: path to SAM3D pipeline.yaml
            sam3d_repo_path: path to sam3d-objects repo root (added to sys.path)
            device: cuda or cpu
        """
        self.device = device
        
        # Add sam3d repo to path if needed
        if sam3d_repo_path is not None:
            sys.path.insert(0, sam3d_repo_path)
            # Also add notebook/ for inference utilities
            notebook_path = str(Path(sam3d_repo_path) / "notebook")
            if notebook_path not in sys.path:
                sys.path.insert(0, notebook_path)
        
        # Load SAM3D model
        from inference import Inference
        self.model = Inference(config_path, compile=False)
        print(f"[SAM3D] Model loaded from {config_path}")
    
    def run_inference(self, image: np.ndarray, mask: np.ndarray, seed: int = 42) -> Dict:
        """
        Run SAM3D inference on a single RGBA image + mask.
        
        Args:
            image: RGBA image, shape (H, W, 4), uint8
            mask: binary mask, shape (H, W), uint8 or bool
        Returns:
            dict with 'gs' (Gaussian splat object), and possibly 'pose' if model outputs it
        """
        output = self.model(image, mask, seed=seed)
        return output
    
    def extract_xyz_from_output(self, output: Dict) -> torch.Tensor:
        """
        Extract xyz positions from SAM3D output.
        The output is in SAM3D's local/normalized space.
        
        Args:
            output: dict from run_inference
        Returns:
            xyz: (N, 3) tensor in local space
        """
        gs = output["gs"]
        
        # SAM3D's Gaussian object - try common attribute names
        # The exact attribute depends on SAM3D's Gaussian class implementation
        if hasattr(gs, 'xyz'):
            xyz = gs.xyz
        elif hasattr(gs, '_xyz'):
            xyz = gs._xyz
        elif hasattr(gs, 'get_xyz'):
            xyz = gs.get_xyz
        else:
            # Fallback: save to ply and read back
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
                gs.save_ply(f.name)
                xyz = self._read_xyz_from_ply(f.name)
        
        if isinstance(xyz, np.ndarray):
            xyz = torch.from_numpy(xyz).float()
        
        return xyz.to(self.device)
    
    def _read_xyz_from_ply(self, ply_path: str) -> torch.Tensor:
        """Read xyz positions from a .ply file."""
        plydata = PlyData.read(ply_path)
        xyz = np.stack([
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ], axis=1)
        return torch.from_numpy(xyz).float()
    
    def extract_pose_from_output(self, output: Dict) -> Dict:
        """
        Extract pose parameters from SAM3D output.
        
        Returns dict with:
            'rotation': (3, 3) rotation matrix
            'scale': (3,) per-axis scale
            'translation': (3,) translation vector
        """
        # 6D rotation -> 3x3 matrix
        rot_6d = output['6drotation_normalized'].squeeze()  # (6,)
        R = rotation_6d_to_matrix(rot_6d)  # (3, 3)
        
        # Scale (per-axis)
        scale = output['scale'].squeeze()  # (3,)
        
        # Translation (scaled)
        translation = output['translation'].squeeze()  # (3,)
        translation_scale = output['translation_scale'].squeeze()  # scalar
        translation = translation * translation_scale
        
        return {
            'rotation': R,
            'scale': scale,
            'translation': translation,
        }
    
    def local_to_camera(
        self, 
        xyz_local: torch.Tensor, 
        pose: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Transform from SAM3D local space to camera space using pose.
        
        Args:
            xyz_local: (N, 3) points in local space (centered, from gs.get_xyz)
            pose: dict with 'rotation' (3,3), 'scale' (3,), 'translation' (3,)
                  from extract_pose_from_output(). None = identity.
        Returns:
            xyz_camera: (N, 3) points in camera space
        """
        if pose is None:
            return xyz_local
        
        R = pose['rotation'].to(xyz_local.device)       # (3, 3)
        scale = pose['scale'].to(xyz_local.device)       # (3,)
        translation = pose['translation'].to(xyz_local.device)  # (3,)
        
        # Apply: scale -> rotate -> translate
        xyz_scaled = xyz_local * scale.unsqueeze(0)      # per-axis scale
        xyz_rotated = xyz_scaled @ R.T                    # rotate
        xyz_camera = xyz_rotated + translation.unsqueeze(0)  # translate
        
        return xyz_camera
    
    def camera_to_world(
        self, 
        xyz_camera: torch.Tensor, 
        cam_R: np.ndarray, 
        cam_T: np.ndarray,
    ) -> torch.Tensor:
        """
        Transform from camera space to world space.
        Uses GIC's camera convention where:
            pw2pc: pc = pw @ R_w2c.T + t_w2c  (world -> camera)
        So the inverse (camera -> world) is:
            pw = (pc - t_w2c) @ R_w2c
        
        In GIC, cam_R stored in Camera.R is already transposed (R = np.transpose(rotmat)),
        and cam_T is the translation vector.
        
        Args:
            xyz_camera: (N, 3) points in camera space
            cam_R: (3, 3) rotation from dataset (GIC convention: transposed rotation)
            cam_T: (3,) translation from dataset
        Returns:
            xyz_world: (N, 3) points in world space
        """
        R = torch.from_numpy(cam_R).float().to(xyz_camera.device)
        T = torch.from_numpy(cam_T).float().to(xyz_camera.device)
        
        # GIC convention: pw2pc does pc = pw @ R.T.T + T = pw @ R + T (since R is already transposed)
        # Inverse: pw = (pc - T) @ R.T
        # But we need to double check with scene/cameras.py:
        #   R_w2c = R.T   (R is stored transposed)
        #   pc = pw @ R_w2c.T + t_w2c = pw @ R + T
        # So: pw = (pc - T) @ R^{-1} = (pc - T) @ R.T (since R is orthogonal after transpose)
        xyz_world = (xyz_camera - T.unsqueeze(0)) @ R.T
        
        return xyz_world
    
    def reconstruct_frame(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        cam_R: np.ndarray,
        cam_T: np.ndarray,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Full pipeline: image+mask -> SAM3D -> local -> camera -> world space xyz.
        
        Args:
            image: RGBA image (H, W, 4) uint8
            mask: binary mask (H, W)
            cam_R: (3, 3) camera rotation (GIC convention)
            cam_T: (3,) camera translation
            seed: random seed for SAM3D
        Returns:
            xyz_world: (N, 3) world-space point cloud
        """
        # Step 1: SAM3D inference
        output = self.run_inference(image, mask, seed=seed)
        xyz_local = self.extract_xyz_from_output(output)
        
        # Step 2: Extract pose and transform local -> camera
        pose = self.extract_pose_from_output(output)
        xyz_camera = self.local_to_camera(xyz_local, pose)
        
        # Step 3: camera -> world (extrinsics)
        xyz_world = self.camera_to_world(xyz_camera, cam_R, cam_T)
        
        return xyz_world
    
    def reconstruct_frame_full(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        cam_R: np.ndarray,
        cam_T: np.ndarray,
        seed: int = 42,
    ) -> Dict:
        """
        Like reconstruct_frame but also returns the SAM3D output and pose.
        Used when we need to save the splat for rendering.
        """
        output = self.run_inference(image, mask, seed=seed)
        xyz_local = self.extract_xyz_from_output(output)
        pose = self.extract_pose_from_output(output)
        xyz_camera = self.local_to_camera(xyz_local, pose)
        xyz_world = self.camera_to_world(xyz_camera, cam_R, cam_T)
        
        return {
            'xyz_world': xyz_world,
            'xyz_local': xyz_local,
            'xyz_camera': xyz_camera,
            'pose': pose,
            'output': output,
        }
    
    def reconstruct_frame_from_ply(
        self,
        ply_path: str,
        cam_R: np.ndarray,
        cam_T: np.ndarray,
        pose_7d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        If SAM3D outputs are pre-computed as .ply files, load and transform.
        Useful for debugging or when SAM3D is run separately.
        """
        xyz_local = self._read_xyz_from_ply(ply_path).to(self.device)
        xyz_camera = self.local_to_camera(xyz_local, pose_7d)
        xyz_world = self.camera_to_world(xyz_camera, cam_R, cam_T)
        return xyz_world