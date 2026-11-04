"""CPU model of Candidate C reduction routing (not GPU equivalence)."""
import math
import random
import unittest


def better(value, index, best_value, best_index):
    if math.isnan(value):
        return False
    if math.isnan(best_value):
        return True
    return value > best_value or (value == best_value and index < best_index)


def chunk_best(scores, base, valid, hr, sample_only, temp=0.8):
    """Model the per-thread two-row candidate followed by the block argmax."""
    bests = [(-math.inf, 2**31-1) for _ in range(256)]
    for tid in range(256):
        v, idx = bests[tid]
        for rep in range((hr + 255)//256):
            loc = tid + rep*256
            if loc >= hr:
                continue
            gi = base + loc
            if gi >= valid:
                continue
            cand = scores[gi]
            if better(cand, gi, v, idx):
                v, idx = cand, gi
        bests[tid] = (v, idx)
    best = (-math.inf, 2**31-1)
    for value, index in bests:
        if better(value, index, *best):
            best = (value, index)
    return best


def reference(scores, base, valid, hr):
    best = (-math.inf, 2**31-1)
    for gi in range(base, min(base+hr, valid)):
        if better(scores[gi], gi, *best):
            best = (scores[gi], gi)
    return best


class TileRoutingTests(unittest.TestCase):
    def test_vocab_geometry(self):
        self.assertEqual(100352 // 256, 392)
        self.assertEqual(100352 // 512, 196)
        self.assertEqual(100352 % 512, 0)

    def test_random_mask_ties_nans(self):
        rng = random.Random(5005)
        for case in range(800):
            valid = rng.randrange(1, 1025)
            scores = [rng.randrange(-100,100)/10 for _ in range(1024)]
            if case % 5 == 0:
                scores[0] = 10.0
                scores[1] = 10.0
            if case % 7 == 0:
                scores[2] = float('nan')
            for hr in (256, 512):
                for base in range(0, 1024, hr):
                    if base < valid:
                        self.assertEqual(reference(scores, base, valid, hr),
                                         chunk_best(scores, base, valid, hr, True))


if __name__ == '__main__':
    unittest.main()
