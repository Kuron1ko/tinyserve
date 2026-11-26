import unittest
import random
from pd.policy import Backpressure, DecodeMemoryPolicy, PrefillJob, PrefillPolicy


class PrefillTests(unittest.TestCase):
    def test_alternation_and_donation(self):
        p = PrefillPolicy(256, 1024)
        p.add(PrefillJob("s1", 80)); p.add(PrefillJob("s2", 90))
        p.add(PrefillJob("l1", 1024)); p.add(PrefillJob("l2", 1024))
        result = p.schedule()
        self.assertEqual([s.request_id for s in result], ["s1", "l1", "s2", "l2"])
        self.assertEqual(sum(s.count for s in result), 682)

    def test_active_limit_and_tail_to_short(self):
        p = PrefillPolicy(256, 256)
        for i in range(6): p.add(PrefillJob(str(i), 512))
        self.assertEqual(list(p.active), ["0", "1", "2", "3"])
        self.assertEqual(list(p.waiting), ["4", "5"])
        p.schedule()
        self.assertEqual(list(p.short), ["0"])
        self.assertEqual(list(p.active), ["1", "2", "3", "4"])

    def test_shrink_only_whole_pages(self):
        p = PrefillPolicy(512, 768)
        p.add(PrefillJob("short", 300)); p.add(PrefillJob("long", 1200))
        r = p.schedule()
        self.assertEqual([(s.request_id, s.count) for s in r], [("short", 300), ("long", 256)])

    def test_last_partial_page_is_not_padding_tokens(self):
        p = PrefillPolicy(256, 256)
        p.add(PrefillJob("a", 263))
        self.assertEqual(p.schedule()[0].count, 256)
        self.assertEqual(p.schedule()[0].count, 7)
        self.assertFalse(p.jobs)

    def test_cache_hits_reduce_remaining_work(self):
        p = PrefillPolicy(256, 512)
        p.add(PrefillJob("hit", 800, 768))
        self.assertEqual([(s.start, s.count) for s in p.schedule()], [(768, 32)])

    def test_cached_long_request_preserves_waiting_admission_order(self):
        p = PrefillPolicy(256, 512)
        for i in range(4):
            p.add(PrefillJob(str(i), 1024))
        p.add(PrefillJob("cached", 800, 768))
        self.assertEqual(list(p.waiting), ["cached"])
        p.remove("0")
        self.assertEqual(list(p.short), ["cached"])

    def test_memory_failure_donates_to_other_queue(self):
        p = PrefillPolicy(256, 512)
        p.add(PrefillJob("s", 80)); p.add(PrefillJob("l", 800))
        self.assertEqual([s.request_id for s in p.schedule(lambda rid, n: rid != "s")], ["l"])

    def test_request_runs_at_most_one_chunk_per_forward(self):
        p = PrefillPolicy(256, 4096)
        p.add(PrefillJob("l", 4096))
        self.assertEqual(len(p.schedule()), 1)

    def test_finite_mixed_workload_completes_without_losing_tokens(self):
        rng = random.Random(42)
        lengths = {str(i): rng.randint(1, 4096) for i in range(100)}
        computed = dict.fromkeys(lengths, 0)
        p = PrefillPolicy(512, 2048)
        for rid, length in lengths.items():
            p.add(PrefillJob(rid, length))
        for _ in range(2000):
            if not p.jobs:
                break
            plan = p.schedule()
            self.assertTrue(plan)
            self.assertLessEqual(sum(s.count for s in plan), 2048)
            self.assertEqual(len({s.request_id for s in plan}), len(plan))
            self.assertLessEqual(len(p.active), 4)
            for s in plan:
                self.assertEqual(s.start, computed[s.request_id])
                computed[s.request_id] += s.count
                if computed[s.request_id] < lengths[s.request_id]:
                    self.assertEqual(s.count % 256, 0)
        self.assertEqual(computed, lengths)


class MemoryTests(unittest.TestCase):
    def test_strict_floor_and_no_double_reservation(self):
        p = DecodeMemoryPolicy(8, 2)
        p.reserve("a", 1024, 256, 0); p.reserve("a", 1024, 256, 0)
        self.assertEqual([(e.kind, e.blocks) for e in p.service(0)], [("reserved", 5)])
        self.assertEqual(p.free, 3)
        p.reserve("b", 256, 0, 0)
        self.assertEqual(p.service(0), [])  # F == R + floor is not enough
        p.release("a")
        self.assertEqual(p.service(1)[0].request_id, "b")

    def test_expand_uses_floor_fifo(self):
        p = DecodeMemoryPolicy(6, 2)
        p.reserve("a", 256, 0, 0); p.reserve("b", 256, 0, 0)
        p.service(0)
        p.expand("b", 1); p.expand("a", 2)
        self.assertEqual([e.request_id for e in p.service(2)], ["b", "a"])
        self.assertEqual(p.free, 2)
        p.expand("a", 3); p.expand("b", 3)
        p.service(3)
        self.assertEqual(p.free, 0)

    def test_timeout_releases_capacity_without_preemption(self):
        p = DecodeMemoryPolicy(4, 0, timeout_s=5)
        p.reserve("a", 768, 0, 0); p.service(0)
        p.expand("a", 0); p.service(0)
        self.assertEqual(p.free, 0)
        p.expand("a", 1)
        p.reserve("b", 256, 0, 1)
        events = p.service(6)
        self.assertEqual([e.kind for e in events], ["expand_timeout", "reserved"])
        self.assertEqual(p.free, 3)

    def test_impossible_reservation_does_not_block_fifo_forever(self):
        p = DecodeMemoryPolicy(4, 1)
        p.reserve("large", 768, 0, 0); p.reserve("small", 256, 0, 0)
        self.assertEqual([e.kind for e in p.service(0)], ["reservation_too_large", "reserved"])

    def test_backpressure_resumes_only_after_draining(self):
        p = Backpressure(100, 50)
        self.assertFalse(p.update(100))
        self.assertTrue(p.update(101))
        self.assertTrue(p.update(70))
        self.assertFalse(p.update(50))


if __name__ == "__main__":
    unittest.main()
