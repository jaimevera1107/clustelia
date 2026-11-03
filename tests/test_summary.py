import unittest

from clustelia import __version__, summarize_clusters


class SummarizeClustersTests(unittest.TestCase):
    def test_standard_labels_and_noise(self):
        result = summarize_clusters([0, 0, 1, -1])
        self.assertEqual(result.n_samples, 4)
        self.assertEqual(result.n_clusters, 2)
        self.assertEqual(result.n_noise, 1)
        self.assertEqual(result.counts, {0: 2, 1: 1, -1: 1})
        self.assertEqual(result.shares[0], 0.5)

    def test_custom_noise_and_generator(self):
        result = summarize_clusters((x for x in ["A", "A", "B", "NA"]), noise_label="NA")
        self.assertEqual(result.n_clusters, 2)
        self.assertEqual(result.n_noise, 1)

    def test_empty(self):
        result = summarize_clusters([])
        self.assertEqual(result.n_samples, 0)
        self.assertEqual(result.n_clusters, 0)
        self.assertEqual(result.shares, {})

    def test_version(self):
        self.assertEqual(__version__, "0.0.0")


if __name__ == "__main__":
    unittest.main()
