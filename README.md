AutoMerge
===

A tool for automatically finding the two best matching frames at the end of one video file and the beginning of another video file.

The tool supports four different image similarity metrics:
- [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) - Mean Squared Error
- [NRMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) - Normalized Root Mean Squared Error
- [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) - Peak Signal-to-Noise Ratio
- [SSIM](https://en.wikipedia.org/wiki/Structural_similarity) - Structural Similarity Index Measure

## Usage
`AutoMerge.py {options} {leading video} {following videos} {seconds} {method}`

Finds the best matching frames in the `{seconds}` last seconds of `{leading video}` and the `{seconds}` first seconds of `{following videos}`, using `{method}` as similarity measure.

- `{leading video}` is the path to the leading video.

- `{following videos}` is the path or a comma separated list of paths to one or several following videos.

- `{seconds}` is the number of seconds to search.

- `{method}` is the similarity measure to use. Valid options are: `mse`, `nrmse`, `psnr`, and `ssim`.

- `{options}` can be any combination  of the following:
  - `--verbose {integer}` where `{integer}` can be any of the following: 
    - 0 = nothing
    - 1 = stage of operation
    - 2 = threading and time
    - 3 = detailed processing
  - `--colour` or `--greyscale`: colour on / off (default off)
  - `--downscale` or `--no-downscale`: downscale on / off (default on)
  
`AutoMerge.py --help` shows this usage information.

## GUI

Alternatively the GUI can be used by running `GUI.py`. The GUI will always run with `{verbose}` set to 3.

![AutomergeScreenshot](https://user-images.githubusercontent.com/17293533/119658919-f5b4ba80-be2d-11eb-8250-5ede3fcad58d.png)
