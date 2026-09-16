import unittest
import numpy as np
import torch
from PIL import Image
from training.adapter import QueryAdapter
from training.augment import augment
from training.metrics import retrieval_metrics


class TrainingTests(unittest.TestCase):
    def test_adapter_starts_as_identity(self):
        features=torch.nn.functional.normalize(torch.randn(4,768),dim=-1)
        self.assertTrue(torch.allclose(QueryAdapter()(features),features,atol=1e-6))

    def test_metrics_include_false_accept(self):
        gallery=np.eye(5,dtype=np.float32)
        queries=gallery[[0,1,3]]
        result=retrieval_metrics(queries,gallery,[0,1,4])
        self.assertAlmostEqual(result['top1'],2/3)
        self.assertEqual(result['false_accepts'],1)
        self.assertEqual(result['accepted'],3)

    def test_augmentation_is_seeded(self):
        image=Image.new('RGB',(300,600),(100,120,200))
        self.assertEqual(augment(image,12).tobytes(),augment(image,12).tobytes())
        self.assertNotEqual(augment(image,12).tobytes(),augment(image,13).tobytes())
