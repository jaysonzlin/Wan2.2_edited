from pathlib import Path


SCRIPTS = (
    "submit_joint.sh",
    "submit_h200_i2v_lingbot_optim_requeue.sh",
    "submit_h200_i2v_requeue.sh",
    "submit_h200_ti2v_negative_requeue.sh",
    "submit_h200.sh",
    "submit_h200_requeue.sh",
    "submit_832x480.sh",
    "submit_lingbot_nodecay.sh",
    "submit_lingbot.sh",
)


def test_submission_scripts_use_the_current_lab_storage_prefix() -> None:
    scripts = [Path(script).read_text() for script in SCRIPTS]

    assert all("/net/holy-isilon/ifs/rc_labs" not in script for script in scripts)
    assert sum(script.count("/n/lab_storage") for script in scripts) == 27
