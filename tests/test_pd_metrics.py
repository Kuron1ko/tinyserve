import unittest
from bench_pd import summarize


class MetricsTests(unittest.TestCase):
    def test_client_token_intervals_and_failure_denominator(self):
        rows = [dict(ok=True,token_times_s=[.1,.2,.5],dispatch_lag_s=0),
                dict(ok=True,token_times_s=[.2,.4],dispatch_lag_s=.001),
                dict(ok=False,token_times_s=[],dispatch_lag_s=0)]
        result = summarize(rows, 1)
        self.assertEqual(result["requests"],3)
        self.assertEqual(result["failed"],1)
        self.assertAlmostEqual(result["failure_rate"],1/3)
        self.assertAlmostEqual(result["ttft_ms"]["p50"],150)
        self.assertEqual(result["measured_intervals"],3)
        self.assertAlmostEqual(result["itl_ms"]["p99"],298)
        self.assertEqual(result["output_tokens_per_s"],5)


if __name__ == "__main__":
    unittest.main()
