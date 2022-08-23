# elizabeth-homes

This the private repository for the project.

## Getting started

We use Devcontainers to have a consistent development environment. To get started first make sure that your system is ready for using VS Code Devcontainers. If you have never used Devcontainers follow the installation steps [as described on the official website](https://code.visualstudio.com/docs/remote/containers#_installation):


>  1. Install and configure [Docker](https://www.docker.com/get-started) for your operating system.
>
>      **Windows / macOS**:
>
>      1. Install [Docker Desktop for Windows/Mac](https://www.docker.com/products/docker-desktop).
>
>      2. If you are using WSL 2 on Windows, to ensure the [WSL 2 back-end](https://aka.ms/vscode-remote/containers/docker-wsl2) is enabled: Right-click on the Docker taskbar item and select **Settings**. Check **Use the WSL 2 based engine** and verify your distribution is enabled under **Resources > WSL Integration**.
>
>      3. When not using the WSL 2 back-end, right-click on the Docker task bar item, select **Settings** and update **Resources > File Sharing** with any locations your source code is kept. See [tips and tricks](/docs/remote/troubleshooting.md#container-tips) for troubleshooting.
>
>      **Linux**:
>
>      1. Follow the [official install instructions for Docker CE/EE for your distribution](https://docs.docker.com/install/#supported-platforms). If you are using Docker Compose, follow the [Docker Compose directions](https://docs.docker.com/compose/install/) as well.
>
>      2. Add your user to the `docker` group by using a terminal to run: `sudo usermod -aG docker $USER`
>
>      3. Sign out and back in again so your changes take effect.
>
>  2. Install [Visual Studio Code](https://code.visualstudio.com/) or [Visual Studio Code Insiders](https://code.visualstudio.com/insiders/).
>
>  3. Install the [Remote Development extension pack](https://aka.ms/vscode-remote/download/extension).

### Starting devcontainer

Once your system is ready, you can start the Devcontainer by:

1. First cloning this repo locally on your machine and then opening its folder in VS Code.
2. Running the Remote-Containers: Open Folder in Container... command from the Command Palette (F1) or quick actions Status bar item.

This should launch this project's Devcontainer.

> **Note**
> The starting of a Devcontainer may take a while the first time you launch it but will drastically speed up on second and further launches.

## Repo structure and git etiquette

All experimental code should be added under your personal `exp/<name>` folder. This way commits should not conflict with each other. Further please commit all your experimental results in your own git branch (e.g. `exp/<name>-creating-building-set`) and not the main branch. This separation will prevent not being able to commit and pushing because you haven't merged somebody else's commits.
