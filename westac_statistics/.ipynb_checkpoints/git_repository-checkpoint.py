import os
import subprocess

class GitRepo:
    """
    A class used to represent a Git Repository

    ...

    Attributes
    ----------
    path : str
        a string representing the local path of the git repository

    Methods
    -------
    tags():
        Returns a list of all tags in the repository.
    current_tag():
        Returns the current tag of the repository, if any.
    update():
        Updates the repository using 'git pull'.
    switch_to_tag(tag):
        Switches the repository to a specific tag.
    check_and_clone(gitlab_url, clone=False):
        Checks if the repository is from a specific GitLab link and clones the repository if the directory does not exist or is empty.
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path : str
            The local path of the git repository
        """
        self.path = path

    @property
    def tags(self):
        """Returns a list of all tags in the repository."""
        result = subprocess.run(['git', 'tag'], cwd=self.path, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f'Error getting tags: {result.stderr}')
        return result.stdout.splitlines()

    @property
    def current_tag(self):
        """Returns the current tag of the repository, if any."""
        result = subprocess.run(['git', 'describe', '--tags', '--exact-match'], cwd=self.path, capture_output=True, text=True)
        if result.returncode != 0:
            return None  # No tag
        return result.stdout.strip()

    def update(self):
        """Updates the repository using 'git pull'."""
        result = subprocess.run(['git', 'pull'], cwd=self.path, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f'Error updating repo: {result.stderr}')
        return result.stdout

    def switch_to_tag(self, tag):
        """
        Switches the repository to a specific tag.

        Parameters
        ----------
        tag : str
            The tag to switch to
        """
        if tag not in self.tags:
            raise Exception(f'Tag {tag} does not exist')
        result = subprocess.run(['git', 'checkout', 'tags/' + tag], cwd=self.path, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f'Error switching to tag {tag}: {result.stderr}')
        return result.stdout

    def check_and_clone(self, gitlab_url, clone=False):
        """
        Checks if the repository is from a specific GitLab link and clones the repository if the directory does not exist or is empty.

        Parameters
        ----------
        gitlab_url : str
            The GitLab URL to check against
        clone : bool, optional
            Whether to clone the repository if the directory does not exist or is empty (default is False)
        """
        # Check if the directory exists and is not empty
        if not os.path.exists(self.path) or not os.listdir(self.path):
            if clone:
                # Clone the repository
                result = subprocess.run(['git', 'clone', gitlab_url, self.path], capture_output=True, text=True)
                if result.returncode != 0:
                    raise Exception(f'Error cloning repo: {result.stderr}')
                return result.stdout
            else:
                raise Exception('Directory does not exist or is empty, and clone option is not set')

        # Check if the repository is from the specified GitLab link
        result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], cwd=self.path, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f'Error getting remote URL: {result.stderr}')
        if result.stdout.strip() != gitlab_url:
            raise Exception(f'Repository is not from {gitlab_url}, it is from {result.stdout.strip()}')