# model/kv_image_store.py
import torch
import hashlib
import numpy as np
import time
from typing import List, Optional
from collections import OrderedDict


class PatchKVStorage:
    def __init__(self, device="cuda"):
        self.device = device

        # 将 patch_k/v 合并为 tensor list（GPU 存储或按需搬到 gpu）
        self.patch_k = []        # list of torch.Tensor (dim_k,)
        self.patch_v = []        # list of torch.Tensor (dim_v,)
        self.patch_hash = []     # list of str

        # images: image_id -> {'feature': tensor, 'patch_hashes': [...], 'patch_idxs': [...]}
        self.images = OrderedDict()

        # 简单 LRU 限制（可选）
        self.max_patches = 2000000

    def _hash_bytes(self, b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def quantize_and_hash(self, emb: torch.Tensor, nbits=8) -> str:
        arr = emb.detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(arr) + 1e-6
        arr = arr / norm
        q = np.round((arr + 1.0) * (2**nbits / 2 - 1)).astype(np.uint8)
        return self._hash_bytes(q.tobytes())

    def find_closest_image(self, image_feature: torch.Tensor, topk=1):
        # linear scan (可后续换 FAISS)
        if not self.images:
            return None

        f = image_feature.detach().cpu().numpy()
        best_id = None
        best_sim = -1.0

        for img_id, meta in self.images.items():
            sim = float(
                np.dot(f, meta['feature'].numpy()) /
                ((np.linalg.norm(f) + 1e-6) * (np.linalg.norm(meta['feature'].numpy()) + 1e-6))
            )
            if sim > best_sim:
                best_sim = sim
                best_id = img_id

        return best_id, best_sim

    def add_patch(self, k: torch.Tensor, v: torch.Tensor, phash: str):
        # 如果存在相同 hash, 复用 index
        if phash in self.patch_hash:
            idx = self.patch_hash.index(phash)
            # ensure storage has k,v
            if self.patch_k[idx] is None:
                self.patch_k[idx] = k.detach().to(self.device)
                self.patch_v[idx] = v.detach().to(self.device)
            return idx

        idx = len(self.patch_hash)
        self.patch_hash.append(phash)
        self.patch_k.append(k.detach().to(self.device))
        self.patch_v.append(v.detach().to(self.device))
        return idx

    def add_image(self, image_id: str, image_feature: torch.Tensor, patch_embs: torch.Tensor, model_compute_kv):
        # model_compute_kv(patch_emb) -> (k, v) tensors (no pos)
        patch_idxs = []
        patch_hashes = []

        for p in patch_embs:
            h = self.quantize_and_hash(p)
            patch_hashes.append(h)

            if h in self.patch_hash:
                idx = self.patch_hash.index(h)
            else:
                k, v = model_compute_kv(p)  # user-provided callable
                idx = self.add_patch(k, v, h)

            patch_idxs.append(idx)

        self.images[image_id] = {
            'feature': image_feature.detach().cpu(),
            'patch_hashes': patch_hashes,
            'patch_idxs': patch_idxs,
            'timestamp': time.time()
        }

        return patch_idxs

    def get_or_build_kv_for_image(
        self,
        image_id,
        image_feature,
        patch_embs,
        model_compute_kv,
        sim_thresh=0.7
    ):
        """
        返回:
            K_tensor [num_patches, dim_k],
            V_tensor [num_patches, dim_v],
            pos_ids [num_patches],
            metadata
        """
        cand = self.find_closest_image(image_feature)
        patch_idxs_out = []

        if cand is not None:
            cand_id, sim = cand
            if sim >= sim_thresh:
                stored = self.images[cand_id]
                # 对每个 patch hash 试图匹配
                for i, p in enumerate(patch_embs):
                    h = self.quantize_and_hash(p)
                    if h in stored['patch_hashes']:
                        idx_in_stored = stored['patch_hashes'].index(h)
                        global_idx = stored['patch_idxs'][idx_in_stored]
                        patch_idxs_out.append(global_idx)
                        continue

                    # not matched -> compute new k, v
                    k, v = model_compute_kv(p)
                    new_idx = self.add_patch(k, v, h)
                    patch_idxs_out.append(new_idx)

                # register this image mapping
                self.images[image_id] = {
                    'feature': image_feature.detach().cpu(),
                    'patch_hashes': [self.patch_hash[i] for i in patch_idxs_out],
                    'patch_idxs': patch_idxs_out,
                    'timestamp': time.time()
                }
            else:
                # no sufficiently close candidate -> build all
                patch_idxs_out = self.add_image(image_id, image_feature, patch_embs, model_compute_kv)
        else:
            patch_idxs_out = self.add_image(image_id, image_feature, patch_embs, model_compute_kv)

        # assemble K,V tensors in patch order
        Ks = [self.patch_k[i] for i in patch_idxs_out]
        Vs = [self.patch_v[i] for i in patch_idxs_out]

        K_tensor = torch.stack(Ks, dim=0).to(self.device)
        V_tensor = torch.stack(Vs, dim=0).to(self.device)
        pos_ids = torch.arange(K_tensor.shape[0], dtype=torch.long)  # default positions

        metadata = {'image_id': image_id, 'patch_idxs': patch_idxs_out}
        return K_tensor, V_tensor, pos_ids, metadata

    def kv_memory_bytes(self):
        total = 0
        for t in self.patch_k + self.patch_v:
            if t is None:
                continue
            total += t.numel() * t.element_size()
        return total
