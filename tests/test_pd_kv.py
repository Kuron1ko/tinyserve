import unittest
import torch
from pd.transport import pack_pages, unpack_pages


class KVTests(unittest.TestCase):
    def test_relocated_pages_across_all_layers_and_inference_tensors(self):
        with torch.inference_mode():
            source = [torch.arange(2*7*256*2*4, dtype=torch.float32).reshape(2,7,256,2,4) + 100000*l
                      for l in range(3)]
            destination = [torch.full_like(t, -1) for t in source]
        wire = pack_pages(source, [5, 1, 3])
        unpack_pages(destination, [0, 6, 2], wire)
        for src, dst in zip(source, destination):
            for old, new in zip([5,1,3], [0,6,2]):
                self.assertTrue(torch.equal(src[:,old], dst[:,new]))
            self.assertTrue(torch.all(dst[:,1] == -1))

    def test_mismatched_page_count_is_rejected(self):
        cache = [torch.zeros(2,4,256,1,8)]
        with self.assertRaises(ValueError):
            unpack_pages(cache, [1], pack_pages(cache, [1,2]))


if __name__ == "__main__":
    unittest.main()
