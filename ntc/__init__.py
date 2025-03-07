import torch
class NeuralTransformationCache(torch.nn.Module):
    def __init__(self, model, xyz_bound_min, xyz_bound_max):
        super(NeuralTransformationCache, self).__init__()
        self.model = model
        self.register_buffer('xyz_bound_min',xyz_bound_min)
        self.register_buffer('xyz_bound_max',xyz_bound_max)
        
    def dump(self, path):
        torch.save(self.state_dict(),path)                                   #모델 저장
        
    def get_contracted_xyz(self, xyz):
        with torch.no_grad():
            contracted_xyz=(xyz-self.xyz_bound_min)/(self.xyz_bound_max-self.xyz_bound_min) #0,1 사이로 정규화
            return contracted_xyz
        
    def forward(self, xyz:torch.Tensor):
        contracted_xyz=self.get_contracted_xyz(xyz)                          # Shape: [N, 3] 의 좌표 텐서
        
        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)                 # MASK : 유효한 좌표인지를 나타내는 Bool 마스크 
        mask = mask.all(dim=1)
        
        ntc_inputs=torch.cat([contracted_xyz[mask]],dim=-1)                  # 유효한 좌표만을 선택하여 입력 
        resi=self.model(ntc_inputs)
        
        masked_d_xyz=resi[:,:3]                                              # 변환된 위치 0,1,2
        masked_d_rot=resi[:,3:7]                                             # 변환된 회전 3,4,5,6 (쿼터니언)
        # masked_d_opacity=resi[:,7:None]
        
        d_xyz = torch.full((xyz.shape[0], 3), 0.0, dtype=torch.half, device="cuda") # 모든 좌표의 기본값을 초기화
        d_rot = torch.full((xyz.shape[0], 4), 0.0, dtype=torch.half, device="cuda")
        d_rot[:, 0] = 1.0
        # d_opacity = self._origin_d_opacity.clone()

        d_xyz[mask] = masked_d_xyz                                           # 마스크 된 값 (유효한 값)만 업데이트
        d_rot[mask] = masked_d_rot
        
        return mask, d_xyz, d_rot 
        
        
