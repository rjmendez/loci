"""Every query path uses the same quantized-search parameters.

The params were hand-built at two call sites and forgotten at a third —
probe_collection, which backs retrieval_selftest. A diagnostic that queries
without rescore/oversampling is measuring a different search path from the one
production runs, so a green selftest would not mean the real search works.

loci_memory is INT8-quantized (quantile=0.99, always_ram=True); it is the only
collection on the instance that is. rescore re-scores ANN candidates against the
original full-precision vectors, so omitting it changes what comes back.
"""
import unittest
from unittest import mock

import qdrant_ops


class SharedSearchParamsTest(unittest.TestCase):

    def test_the_params_are_rescore_and_oversampled(self):
        sp = qdrant_ops._quant_search_params()
        self.assertTrue(sp.quantization.rescore)
        self.assertEqual(sp.quantization.oversampling, 2.0)

    def test_one_instance_is_reused(self):
        self.assertIs(qdrant_ops._quant_search_params(),
                      qdrant_ops._quant_search_params())

    def test_probe_collection_passes_them_to_query_points(self):
        """The site that forgot them. This is the regression."""
        client = mock.MagicMock()
        client.get_collection.return_value.config.params.vectors = {
            "dense": mock.MagicMock(size=768)
        }
        client.count.return_value.count = 10
        client.query_points.return_value.points = []
        with mock.patch.object(qdrant_ops, "_collection_shape",
                               return_value={"named": True, "dense_dims": [768],
                                             "points": 10, "sparse": False}), \
             mock.patch.object(qdrant_ops, "_dense_vector_name", return_value="dense"):
            qdrant_ops.probe_collection([0.0] * 768, client, "loci_memory", 3)
        kw = client.query_points.call_args.kwargs
        self.assertIn("search_params", kw,
                      "probe_collection must query the same path production does")
        self.assertTrue(kw["search_params"].quantization.rescore)

    def test_no_call_site_builds_its_own_copy(self):
        """Three hand-built copies is how one of them drifted. Keep it to one."""
        src = open(qdrant_ops.__file__).read()
        built = src.count("QuantizationSearchParams(")
        self.assertEqual(built, 1,
                         f"{built} construction sites; the shared helper should be the only one")


if __name__ == "__main__":
    unittest.main()
