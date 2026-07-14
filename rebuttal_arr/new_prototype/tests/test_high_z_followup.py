from __future__ import annotations

from experiment.high_z_followup import select_high_z_cases


def test_previous_run_global_top_twenty_selection_is_frozen() -> None:
    selected = select_high_z_cases()
    assert len(selected) == 20
    assert [row["rank"] for row in selected] == list(range(1, 21))
    assert sum(row["watermark"] == "sweet" for row in selected) == 11
    assert sum(row["watermark"] == "wllm" for row in selected) == 9
    assert not [row for row in selected if row["watermark"] == "synthid"]
    assert selected[0]["config_key"] == "Llama31Instruct8B--sweet--humaneval_py--054"
    assert selected[0]["task_name"] == "humaneval_py/40"
    assert selected[0]["old_step100_z"] == 19.254816734512815
    assert selected[-1]["old_step100_z"] == 12.715653823235167
