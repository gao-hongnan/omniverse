"""Core functions for Git versioning."""
from __future__ import annotations

import logging
import subprocess
from typing import List

logger = logging.getLogger(__name__)

__all__ = ["check_git_status", "get_git_commit_hash", "commit_changes", "push_changes"]


def check_git_status(working_dir: str | None = None) -> bool:
    """
    Check the Git status of the working directory. If there are untracked or
    uncommitted changes, return False.

    Parameters
    ----------
    working_dir : str, optional
        The path of the working directory where the Git command should be executed,
        by default None. If None, it uses the current working directory.

    Returns
    -------
    bool
        True if there are no untracked or uncommitted changes in the working directory,
        False otherwise.
    """
    status_output = subprocess.check_output(["git", "status", "--porcelain"], cwd=working_dir).decode("utf-8").strip()
    return len(status_output) == 0


def log_message_if_working_dir_is_none(working_dir: str | None = None) -> None:
    """
    Log an informative message if no working directory is provided.

    Parameters
    ----------
    working_dir : str, optional
        The path of the working directory, by default None.
            If None, an info message is logged.
    """
    if working_dir is None:
        logger.info("Working directory not provided. Defaulting to current directory.")
    logger.info(f"Working directory: {working_dir}")


def get_git_commit_hash(
    working_dir: str | None = None, are_there_untracked_or_uncommitted: bool = False, short: bool = True
) -> str:
    """
    Get the current Git commit hash.

    If Git is not installed or the working directory is not a Git repository,
    the function returns "N/A".

    Parameters
    ----------
    working_dir : str, optional
        The path of the working directory where the Git command should be executed,
        by default None. If None, it uses the current working directory ".".
    are_there_untracked_or_uncommitted : Literal[True, False], optional
        Whether to check if there are untracked or uncommitted changes in the
        working directory, by default False.
    short : bool
        Whether to return the short commit hash (7 characters) or the full hash,
        by default True.

    Returns
    -------
    commit_hash : str
        The Git commit hash, or "N/A" if Git is not installed or the working
        directory is not a Git repository.
    """
    log_message_if_working_dir_is_none(working_dir)

    git_command = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]

    try:
        if are_there_untracked_or_uncommitted and not check_git_status(working_dir):
            error_message = (
                "There are untracked or uncommitted files in the working directory. "
                "Please commit or stash them before running training as the commit hash "
                "will be used to tag the model."
            )
            raise RuntimeError(error_message)

        commit_hash = subprocess.check_output(git_command, cwd=working_dir).decode("utf-8").strip()
    except FileNotFoundError:
        logger.exception("Git not found or the provided working directory doesn't exist.")
        commit_hash = "N/A"
    except subprocess.CalledProcessError:
        logger.exception("The provided directory is not a Git repository.")
        commit_hash = "N/A"

    return commit_hash


def commit_changes(
    commit_message: str,
    file_paths: List[str] | None = None,
    working_dir: str | None = None,
) -> None:
    """
    Commit changes to a Git repository.

    Parameters
    ----------
    commit_message : str
        The message to use for the commit.
    file_paths : Optional[List[str]], default=None
        List of file paths to be added to the commit. If not specified,
        all changes will be committed.
    working_dir : str | None, default=None
        The path of the working directory where the Git commands should be executed.
        If None, the commands will be executed in the current working directory.
    """
    log_message_if_working_dir_is_none(working_dir)

    try:
        # Add files to the staging area
        if file_paths is None:
            subprocess.run(["git", "add", "."], check=True, cwd=working_dir)
        else:
            for file_path in file_paths:
                subprocess.run(["git", "add", file_path], check=True, cwd=working_dir)

        # Commit the changes
        subprocess.run(["git", "commit", "-m", commit_message], check=True, cwd=working_dir)
    except subprocess.CalledProcessError as error:
        print(f"An error occurred while committing changes: {error}")


def push_changes(
    remote_name: str = "origin",
    branch_name: str = "master",
    working_dir: str | None = None,
) -> None:
    """
    Push committed changes to a remote repository.

    Parameters
    ----------
    remote_name : str, default='origin'
        The name of the remote repository to push to.
    branch_name : str, default='master'
        The name of the branch to push.
    working_dir : str, optional
        The path of the working directory where the Git command should be executed,
        by default None. If None, it uses the current working directory.
    """

    log_message_if_working_dir_is_none(working_dir)

    try:
        # Push the changes
        subprocess.run(["git", "push", remote_name, branch_name], check=True, cwd=working_dir)
    except subprocess.CalledProcessError as error:
        print(f"An error occurred while pushing changes: {error}")
