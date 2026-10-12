# Dataset attribution

PneumoniaMNIST is distributed as part of MedMNIST and MedMNIST+ under the [Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Source and creator attribution:

- [MedMNIST+ release 3.0, including the 28, 64, 128, and 224 pixel PneumoniaMNIST files](https://doi.org/10.5281/zenodo.10519652). The linked record identifies the dataset creators and supplies the original license notice.
- [Original OCT and chest X-ray dataset, version 2](https://doi.org/10.17632/rscbjbr9sj.2). The linked record identifies its contributors and lists CC BY 4.0.

Processing in this experiment: select the fixed 100-image balanced training subset; retain the 624-image test split; export selected grayscale arrays as PNG files. For model requests, convert to RGB and apply bicubic resizing on a 896 × 896 canvas. Labels are preserved. The split manifests identify selected indices and include pixel checksums. The repository supplies code and manifests; the preparation command obtains the image arrays from the official dataset release.

The dataset license is separate from the repository's code license. Data are supplied without warranties under the [full CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/legalcode.en). No endorsement by the dataset creators is implied. Keep this notice with redistributed subsets or converted images and record any further changes. MedMNIST is a research benchmark and is not intended for clinical use.

Related publications requested by the dataset project:

- [MedMNIST v2, Scientific Data (2023)](https://doi.org/10.1038/s41597-022-01721-8).
- [MedMNIST classification benchmark, ISBI (2021)](https://doi.org/10.1109/ISBI48211.2021.9434062).
- [Original source-image study, Cell (2018)](https://doi.org/10.1016/j.cell.2018.02.010).
