from pathlib import Path

from ehrgraph.eval.publish import sanitize, scan_notebook


def _nb(*texts, html=None):
    outputs = [{"output_type": "stream", "name": "stdout", "text": [t]} for t in texts]
    if html:
        outputs.append({"output_type": "execute_result", "data": {"text/html": html}})
    return {"cells": [{"cell_type": "code", "source": "", "outputs": outputs, "metadata": {}}]}


def test_clean_aggregate_output_passes():
    nb = _nb(
        "6,998 nodes, prevalence 0.254", "homophily 0.209 | OMOP/320128 essential hypertension"
    )
    assert scan_notebook(nb) == []


def test_flags_patient_columns_ids_timestamps_and_long_tables():
    assert any("patient_id" in p for p in scan_notebook(_nb("patient_id  value")))
    assert any("id-shaped" in p for p in scan_notebook(_nb("node 115969158 has degree 4")))
    assert any("timestamp" in p for p in scan_notebook(_nb("prediction at 2016-01-01 23:59")))
    assert any("row table" in p for p in scan_notebook(_nb("...\n[6998 rows x 3 columns]")))
    assert any("column" in p for p in scan_notebook(_nb(html="<th>prediction_time</th>")))


def test_concept_ids_and_counts_are_not_mistaken_for_patient_ids():
    nb = _nb("OMOP/40511000119 code", "concept 1201005123", "18,698 links", "[12 rows x 4 columns]")
    assert scan_notebook(nb) == []


def test_exact_sensitive_tokens_are_caught():
    nb = _nb("group of 3 incl. 4471")
    assert scan_notebook(nb, sensitive={"4471"})  # short real ids are still caught exactly


def test_size_cap():
    assert any("bytes" in p for p in scan_notebook(_nb("ok"), size=6_000_000))


def test_sanitize_removes_home_path_and_execution_metadata(tmp_path):
    nb = _nb("trimmed tables : /home/someone/ehrshot-data")
    nb["cells"][0]["metadata"]["execution"] = {"iopub.status.busy": "2026-10-02T11:00:00"}
    clean = sanitize(nb, home=Path("/home/someone"))
    assert clean["cells"][0]["outputs"][0]["text"] == ["trimmed tables : ~/ehrshot-data"]
    assert "execution" not in clean["cells"][0]["metadata"]
    assert nb["cells"][0]["outputs"][0]["text"] == ["trimmed tables : /home/someone/ehrshot-data"]
