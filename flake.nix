{
  description = "Project dev environment (Python + uv)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Pick your Python version here:
        python = pkgs.python311;

        # Pick your Julia version here:
        # - julia-bin tends to track upstream Julia closely on unstable
        # - if you want a specific version, you can use pkgs.julia_110, pkgs.julia_19, etc (if available)
        # julia = pkgs.julia-bin;

        uv = pkgs.uv;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            uv
            pkgs.git
            pkgs.ripgrep
          ];

          # Nice defaults for project-local tooling
          env = {
            # Make uv put the venv in the repo (standard)
            UV_VENV = ".venv";
          };

          shellHook = ''
            echo "Python:  ${python.interpreter}"
            echo "uv:      $(uv --version)"

            # Create venv if missing, using the Nix-provided python interpreter
            if [ ! -d .venv ]; then
              uv venv --python ${python.interpreter}
            fi

            # Activate automatically when you enter
            source .venv/bin/activate

            # Optional: keep this fast by only syncing when lockfiles change
            if [ -f pyproject.toml ]; then
              # If you use uv.lock, this will install exactly what's locked
              uv sync || true
            fi
          '';
        };
      });
}
