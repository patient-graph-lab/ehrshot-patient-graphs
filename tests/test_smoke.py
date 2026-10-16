"""Every package and the notebook entry point import cleanly."""

import ehrgraph
import ehrgraph.data
import ehrgraph.eval
import ehrgraph.graphs
import ehrgraph.netstats
import ehrgraph.project
import ehrgraph.viz


def test_package_imports():
    assert ehrgraph.project.setup is not None
