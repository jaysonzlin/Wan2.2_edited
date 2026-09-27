"""Train SimGen PC and bridge weights while keeping the Wan video model frozen."""

from joint_simgen import main


if __name__ == "__main__":
    main(freeze_video=True)
