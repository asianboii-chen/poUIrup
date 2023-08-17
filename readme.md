## PoUIrup ("Power up UI")

A (third) cross-platform rewrite attempt of Asianboii's UI, an objectively amazing keyboard and mouse macro kit.

### Usage

I will bring back the one-file executable later using PyInstaller. Stay tuned.

For now, run it the programmer's way:

```sh
git clone <this_repo>
cd <this_repo>
conda env create -f environment.yaml
python source/main.py &
```

### Development

- To export Python requirements, run `conda env export --from-history > environment.yaml`, then delete the last line ("`prefix: ...`") from the generated `environment.yaml`.

### Platform-Related Notes

- I try to make as much of the code cross-platform as possible, that is, able to run on multiple platforms with the same code.
  - Currently though, development and testing is done on macOS only.
