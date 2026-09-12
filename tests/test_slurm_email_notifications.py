from pathlib import Path


def test_every_slurm_submission_script_requests_lifecycle_email_notifications():
    for script in sorted(Path(".").glob("*.sh")):
        contents = script.read_text()
        if "#SBATCH" not in contents:
            continue

        assert "#SBATCH --mail-user=jlin3@college.harvard.edu" in contents
        assert "#SBATCH --mail-type=BEGIN,END,FAIL" in contents
