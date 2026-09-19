"""Package-import smoke test. Real tests (leakage-safety, graph correctness) land in Week 2."""

import ehrgraph
import ehrgraph.data
import ehrgraph.eval
import ehrgraph.graphs
import ehrgraph.models
import ehrgraph.netstats
import ehrgraph.viz


def test_package_imports():
    assert ehrgraph is not None
