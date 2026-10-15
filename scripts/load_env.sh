# Sourced by pixi on activation (pixi.toml): exports the variables of the gitignored .env, if there is one (e.g. the
# wandb project and entity; see .env.example)
if [ -f "$PIXI_PROJECT_ROOT/.env" ]; then
  set -a
  . "$PIXI_PROJECT_ROOT/.env"
  set +a
fi
