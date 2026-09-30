import torch
import torch.nn.functional as F

from typing import Optional, Union
from openfold.utils.rigid_utils import Rigid, Rotation


class EuclideanTransformation(Rigid):
    def __matmul__(self, other: Rigid):
        return self.from_rigid(self.compose(other))

    @classmethod
    def from_rigid(cls, rigid: Rigid):
        return cls(rigid._rots, rigid._trans)

    @classmethod
    def from_3_points(cls, *args, **kwargs):
        return cls.from_rigid(super().from_3_points(*args, **kwargs))

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> "EuclideanTransformation":
        return EuclideanTransformation(
            rots=self._rots.to(device, dtype),
            trans=self._trans.to(device, dtype),
        )

    @staticmethod
    def _rot_axis(angle: Union[float, torch.Tensor], axis: int):
        # Convert float into tensor
        if type(angle) is not torch.Tensor:
            angle = torch.tensor(angle)
            if len(angle.shape) == 0:
                angle = angle.unsqueeze(-1)

        # Angle representation can be either 1D or 2D
        assert angle.shape[-1] < 3, \
            f"Undefined angle representation! (dims={angle.shape[-1]})"

        # Convert radians (1D) into unit circle (2D) representation
        if angle.shape[-1] == 1:
            angle = torch.cat((torch.cos(angle), torch.sin(angle)), dim=-1)

        # Normalize 2D angle representation
        angle = F.normalize(angle, dim=-1)

        # Create rotation matrix
        rots = angle.new_zeros(angle.shape[:-1] + (3, 3))
        rots[..., axis, axis] = 1.
        rots[..., (axis + 1) % 3, (axis + 1) % 3] = angle[..., 0]
        rots[..., (axis + 2) % 3, (axis + 2) % 3] = angle[..., 0]
        rots[..., (axis + 2) % 3, (axis + 1) % 3] = angle[..., 1]
        rots[..., (axis + 1) % 3, (axis + 2) % 3] = -angle[..., 1]

        return EuclideanTransformation(
            rots=Rotation(rot_mats=rots, quats=None), trans=None
        )

    @staticmethod
    def _trans_axis(dist: Union[float, torch.Tensor], axis: int):
        # Convert float to tensor
        if type(dist) is not torch.Tensor:
            dist = torch.tensor(dist)

        # Create translation vector
        trans = torch.zeros(dist.shape + (3,))
        trans[..., axis] = dist

        return EuclideanTransformation(rots=None, trans=trans)

    @staticmethod
    def rot_x(angle: Union[float, torch.Tensor]):
        return EuclideanTransformation._rot_axis(angle, axis=0)

    @staticmethod
    def rot_y(angle: Union[float, torch.Tensor]):
        return EuclideanTransformation._rot_axis(angle, axis=1)

    @staticmethod
    def rot_z(angle: Union[float, torch.Tensor]):
        return EuclideanTransformation._rot_axis(angle, axis=2)

    @staticmethod
    def trans_x(dist: Union[float, torch.Tensor]):
        return EuclideanTransformation._trans_axis(dist, axis=0)

    @staticmethod
    def trans_y(dist: Union[float, torch.Tensor]):
        return EuclideanTransformation._trans_axis(dist, axis=1)

    @staticmethod
    def trans_z(dist: Union[float, torch.Tensor]):
        return EuclideanTransformation._trans_axis(dist, axis=2)
