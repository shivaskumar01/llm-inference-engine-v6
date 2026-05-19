"""Download Llama 3.2 1B-Instruct weights into ./data.

Default repo is the gated meta-llama mirror (requires `huggingface-cli login`
and license acceptance). Pass --mirror to use the unsloth re-host that does
not require gating.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


REPOS = {
    "meta":    "meta-llama/Llama-3.2-1B-Instruct",
    "unsloth": "unsloth/Llama-3.2-1B-Instruct",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mirror", choices=sorted(REPOS), default="meta",
                   help="Which HF repo to pull from (default: meta).")
    p.add_argument("--out", default="./data/llama-3.2-1b-instruct",
                   help="Local directory to populate.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    repo = REPOS[args.mirror]
    print(f"Downloading {repo} to {out}/ ...")
    try:
        snapshot_download(repo_id=repo, local_dir=str(out))
    except Exception as e:  # noqa: BLE001 — surface to user with hint
        print(f"\nDownload failed: {e}", file=sys.stderr)
        if args.mirror == "meta":
            print("\nThe meta-llama repo is gated. Either run "
                  "`huggingface-cli login` and accept the license at "
                  "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct, "
                  "or retry with `--mirror unsloth`.", file=sys.stderr)
        return 1

    print(f"\nDone. Files in {out}/:")
    for f in sorted(out.iterdir()):
        size = f.stat().st_size if f.is_file() else 0
        print(f"  {f.name:40s} {size:>14,d} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
