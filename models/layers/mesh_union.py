import torch


class MeshUnion:
    def __init__(self, n, device=torch.device('cpu')):
        self.__size = n
        self.rebuild_features = self.rebuild_features_average
        self.groups = torch.eye(n, device=device)

    def union(self, source, target):
        self.groups[target, :] += self.groups[source, :]

    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        return self.groups[edge_key, :]

    def get_occurrences(self):
        return torch.sum(self.groups, 0)

    def get_groups(self, tensor_mask):
        self.groups = torch.clamp(self.groups, 0, 1)
        return self.groups[tensor_mask, :]

    def rebuild_features_average(self, features, mask, target_edges):
        self.prepare_groups(features, mask)
        
        # 确保数据类型一致并且在相同设备上
        features_squeezed = features.squeeze(-1)
        
        # 一次性确保设备和数据类型匹配，减少重复转换
        if self.groups.device != features_squeezed.device or self.groups.dtype != features_squeezed.dtype:
            self.groups = self.groups.to(device=features_squeezed.device, dtype=features_squeezed.dtype)
        
        # 检查维度兼容性
        if features_squeezed.shape[-1] != self.groups.shape[0]:
            raise ValueError(f"Matrix multiplication dimension mismatch: "
                           f"features {features_squeezed.shape} vs groups {self.groups.shape}")
        
        # 直接使用矩阵乘法，移除 try-catch 以避免性能开销
        fe = torch.matmul(features_squeezed, self.groups)
        
        # 预计算 occurrences 并复用
        occurrences = torch.sum(self.groups, 0)
        if occurrences.numel() > 0:
            occurrences = torch.clamp(occurrences, min=1e-8)
            fe = fe / occurrences.unsqueeze(0)  # 使用 unsqueeze 而不是 expand
        
        # 优化 padding 操作
        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            fe = torch.nn.functional.pad(fe, (0, padding_b), mode='constant', value=0)
        
        return fe

    def prepare_groups(self, features, mask):
        # 直接在 GPU 上创建 tensor_mask，避免 CPU->GPU 转换
        if not isinstance(mask, torch.Tensor):
            tensor_mask = torch.from_numpy(mask).to(device=self.groups.device, dtype=torch.bool)
        else:
            tensor_mask = mask.to(device=self.groups.device, dtype=torch.bool)
        
        self.groups = torch.clamp(self.groups[tensor_mask, :], 0, 1).transpose_(1, 0)
        padding_a = features.shape[1] - self.groups.shape[0]
        if padding_a > 0:
            # 使用 F.pad 代替 ConstantPad2d，更高效
            self.groups = torch.nn.functional.pad(self.groups, (0, 0, 0, padding_a), mode='constant', value=0)
                        